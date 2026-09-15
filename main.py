from __future__ import annotations
import hashlib, hmac, json, math, os, time
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlencode
import numpy as np
import pandas as pd
import requests

BINGX_BASE_URL = "https://open-api.bingx.com"
BINGX_API_KEY = os.getenv('BINGX_API_KEY', '')
BINGX_SECRET_KEY = os.getenv('BINGX_SECRET_KEY', '')
SYMBOLS = ['ZECUSDT','USELESSUSDT','FETUSDT','HYPEUSDT','JTOUSDT','VETUSDT','XRPUSDT','ETHUSDT','UNIUSDT','INJUSDT','SEIUSDT','1000SHIBUSDT','DYDXUSDT']
INTERVAL = '30m'
NOTIONAL_USDT = float(os.getenv('PC_NOTIONAL_USDT', '60'))
LEVERAGE = int(os.getenv('PC_LEVERAGE', '20'))
PAPER = os.getenv('PC_PAPER', 'true').lower() == 'true'
LIVE_CONFIRM = os.getenv('PC_LIVE_CONFIRM', '') == 'I_UNDERSTAND_LIVE_TRADING'
STATE_PATH = Path(os.getenv('PC_STATE_PATH', '/data/purple_cloud_state.json'))
SCAN_SECONDS = int(os.getenv('PC_SCAN_SECONDS', '60'))
PERIOD = 20
ALPHA = 1.5
BPT = 0.2
SPT = 0.2


def format_symbol(symbol):
    s=symbol.upper().replace('-','')
    return f'{s[:-4]}-USDT' if s.endswith('USDT') else s


def api_payload(method, path, params=None):
    if not BINGX_API_KEY or not BINGX_SECRET_KEY:
        raise RuntimeError('BINGX_API_KEY/BINGX_SECRET_KEY missing')
    p={k:v for k,v in dict(params or {}).items() if v is not None}
    p['timestamp']=int(time.time()*1000); p['recvWindow']=5000
    query=urlencode(sorted(p.items()))
    sig=hmac.new(BINGX_SECRET_KEY.encode(),query.encode(),hashlib.sha256).hexdigest()
    url=f'{BINGX_BASE_URL}{path}?{query}&signature={sig}'
    r=requests.request(method,url,headers={'X-BX-APIKEY':BINGX_API_KEY,'User-Agent':'PurpleCloudBot'},timeout=20)
    r.raise_for_status(); data=r.json()
    if data.get('code') not in (0,'0',None): raise RuntimeError(f"BingX {data.get('code')}: {data.get('msg')}")
    return data.get('data')


def signed_get(path, params=None): return api_payload('GET',path,params)
def signed_post(path, params=None): return api_payload('POST',path,params)


def public_get(path, params=None):
    r=requests.get(f'{BINGX_BASE_URL}{path}',params=params or {},headers={'User-Agent':'PurpleCloudBot'},timeout=20)
    r.raise_for_status(); data=r.json()
    if data.get('code') not in (0,'0',None): raise RuntimeError(f"BingX {data.get('code')}: {data.get('msg')}")
    return data.get('data')


def get_klines(symbol, limit=250):
    rows=public_get('/openApi/swap/v3/quote/klines',{'symbol':format_symbol(symbol),'interval':INTERVAL,'limit':limit}) or []
    out=[]
    for x in rows:
        if isinstance(x,list) and len(x)>=6: out.append(x[:6])
        elif isinstance(x,dict): out.append([x.get('time') or x.get('openTime'),x.get('open'),x.get('high'),x.get('low'),x.get('close'),x.get('volume')])
    d=pd.DataFrame(out,columns=['time','open','high','low','close','volume'])
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d['time']=pd.to_datetime(pd.to_numeric(d['time'],errors='coerce'),unit='ms',utc=True)
    return d.dropna().sort_values('time').drop_duplicates('time').reset_index(drop=True)


def get_contract(symbol):
    data=public_get('/openApi/swap/v2/quote/contracts',{'symbol':format_symbol(symbol)}) or []
    if isinstance(data,dict): data=data.get('contracts',data.get('data',[data]))
    if isinstance(data,dict): data=[data]
    fs=format_symbol(symbol)
    c=next((x for x in data if x.get('symbol')==fs),None)
    if not c: raise RuntimeError(f'contract info missing for {fs}')
    if str(c.get('apiStateOpen','true')).lower()!='true': raise RuntimeError(f'{fs} API opening disabled')
    return c


def get_positions(symbol=None):
    data=signed_get('/openApi/swap/v2/user/positions',{'symbol':format_symbol(symbol)} if symbol else None) or []
    if isinstance(data,dict): data=data.get('positions',data.get('data',[]))
    return data if isinstance(data,list) else []


def live_position(symbol, side=None):
    found=[]
    for p in get_positions(symbol):
        try: amt=abs(float(p.get('positionAmt',0) or 0))
        except Exception: amt=0
        ps=str(p.get('positionSide','')).upper()
        if amt>0 and (side is None or ps==side): found.append((p,amt))
    return found


def is_isolated(p):
    v=p.get('isolated',False)
    if isinstance(v,bool): return v
    return str(v).strip().lower() in ('true','1','yes')


def ensure_hedge_mode():
    data=signed_get('/openApi/swap/v1/positionSide/dual') or {}
    dual=data.get('dualSidePosition') if isinstance(data,dict) else None
    if isinstance(dual,str): dual=dual.lower()=='true'
    if dual is not True: raise RuntimeError('BingX account is not in Hedge Mode; live trading aborted')


def set_cross_and_leverage(symbol):
    fs=format_symbol(symbol)
    mt=signed_get('/openApi/swap/v2/trade/marginType',{'symbol':fs}) or {}
    current=str(mt.get('marginType','')).upper() if isinstance(mt,dict) else ''
    if current!='CROSSED':
        signed_post('/openApi/swap/v2/trade/marginType',{'symbol':fs,'marginType':'CROSSED'})
    lev=signed_get('/openApi/swap/v2/trade/leverage',{'symbol':fs}) or {}
    for side,key in (('LONG','longLeverage'),('SHORT','shortLeverage')):
        try: current_lev=int(float(lev.get(key,0) or 0))
        except Exception: current_lev=0
        if current_lev!=LEVERAGE:
            signed_post('/openApi/swap/v2/trade/leverage',{'symbol':fs,'side':side,'leverage':LEVERAGE})


def order_qty(symbol, price):
    c=get_contract(symbol); prec=int(c.get('quantityPrecision',8)); minq=float(c.get('tradeMinQuantity',0) or 0); minusdt=float(c.get('tradeMinUSDT',0) or 0)
    raw=NOTIONAL_USDT/price; scale=10**prec; qty=math.floor(raw*scale)/scale
    if qty<minq: qty=minq
    if qty*price+1e-9 < minusdt: qty=math.ceil((minusdt/price)*scale)/scale
    if qty<=0: raise RuntimeError('calculated quantity <= 0')
    return f'{qty:.{prec}f}'


def place_market(symbol, position_side, quantity, opening):
    fs=format_symbol(symbol); q=abs(float(quantity))
    if q<=0: raise RuntimeError('order quantity <= 0')
    side=('BUY' if position_side=='LONG' else 'SELL') if opening else ('SELL' if position_side=='LONG' else 'BUY')
    cid=f"pc-{symbol.lower()}-{position_side.lower()}-{'o' if opening else 'c'}-{int(time.time()*1000)}"[:40]
    return signed_post('/openApi/swap/v2/trade/order',{'symbol':fs,'side':side,'positionSide':position_side,'type':'MARKET','quantity':format(q,'.12g'),'clientOrderId':cid})


def wait_side_flat(symbol, side, seconds=15):
    end=time.time()+seconds
    while time.time()<end:
        if not live_position(symbol,side): return True
        time.sleep(1)
    return False


def available_usdt():
    bal=signed_get('/openApi/swap/v3/user/balance') or []
    if isinstance(bal,dict): bal=bal.get('balance',bal.get('data',bal))
    if isinstance(bal,list): b=next((x for x in bal if x.get('asset')=='USDT'),{})
    else: b=bal if isinstance(bal,dict) else {}
    return float(b.get('availableMargin',b.get('availableBalance',0)) or 0)


def verify_account():
    print('[PC API] Account verification starting; orders only possible when PAPER=False AND live confirmation is set.')
    ensure_hedge_mode()
    bal=signed_get('/openApi/swap/v3/user/balance') or []
    if isinstance(bal,dict): bal=bal.get('balance',bal.get('data',bal))
    if isinstance(bal,list): b=next((x for x in bal if x.get('asset')=='USDT'),bal[0] if bal else {})
    else: b=bal if isinstance(bal,dict) else {}
    print(f"[PC API] BALANCE OK | asset={b.get('asset','USDT')} | equity={b.get('equity',b.get('balance','?'))} | available={b.get('availableMargin',b.get('availableBalance','?'))}")
    opened=[]
    for p in get_positions():
        try: amt=abs(float(p.get('positionAmt',0) or 0))
        except Exception: amt=0
        if amt>0: opened.append(f"{p.get('symbol')}:{p.get('positionSide')}:{amt}")
    print(f"[PC API] HEDGE MODE OK | POSITIONS OK | open={len(opened)}"+(f" | {'; '.join(opened[:20])}" if opened else ''))


def rma(s,n): return s.ewm(alpha=1/n,adjust=False).mean()
def atr(d,n):
    p=d.close.shift(1); tr=pd.concat([d.high-d.low,(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1); return rma(tr,n)
def vwma(x,v,n): return (x*v).rolling(n).sum()/v.rolling(n).sum().replace(0,np.nan)


def purple_cloud(d):
    d=d.copy().reset_index(drop=True); n1=int(np.ceil(PERIOD/4)); n2=int(np.ceil(PERIOD/2)); x2=atr(d,PERIOD)*ALPHA
    xh=d.close+x2; xl=d.close-x2; hl2=(d.high+d.low)/2
    a1=vwma(hl2*d.volume,d.volume,n1)/vwma(d.volume,d.volume,n1); a2=vwma(hl2*d.volume,d.volume,n2)/vwma(d.volume,d.volume,n2)
    a3=2*a1-a2; a4=vwma(a3,d.volume,PERIOD); b1=rma(d.close,PERIOD); a5=2*a4*b1/(a4+b1)
    buy=(a5<=xl)&(d.close>b1*(1+BPT*0.01)); sell=(a5>=xh)&(d.close<b1*(1-SPT*0.01))
    xs=np.zeros(len(d),dtype=int)
    for i in range(1,len(d)): xs[i]=1 if bool(buy.iloc[i]) else (-1 if bool(sell.iloc[i]) else xs[i-1])
    changed=pd.Series(xs).ne(pd.Series(xs).shift(1)); d['pc_buy']=buy&changed; d['pc_sell']=sell&changed
    return d


def load_state():
    try:
        if STATE_PATH.exists(): return json.loads(STATE_PATH.read_text())
    except Exception as e: print('[PC] state read error:',e)
    return {'positions':{},'last_signal_candle':{},'events':[]}
def save_state(st):
    STATE_PATH.parent.mkdir(parents=True,exist_ok=True); tmp=STATE_PATH.with_suffix('.tmp'); tmp.write_text(json.dumps(st,indent=2)); tmp.replace(STATE_PATH)
def emit(st,symbol,action,side,price,candle,note=''):
    mode='PAPER' if PAPER else ('LIVE' if LIVE_CONFIRM else 'LIVE_BLOCKED')
    ev={'ts':datetime.now(timezone.utc).isoformat(),'symbol':symbol,'action':action,'side':side,'price':price,'signal_candle':candle,'notional_usdt':NOTIONAL_USDT,'leverage':LEVERAGE,'mode':mode,'note':note}
    st['events'].append(ev); st['events']=st['events'][-1000:]; print('[PC EVENT]',json.dumps(ev))


def execute_live(st,symbol,signal,price,candle):
    if not LIVE_CONFIRM:
        emit(st,symbol,'LIVE_BLOCKED',signal,price,candle,'PC_LIVE_CONFIRM guard not enabled'); return
    ensure_hedge_mode()
    opposite='SHORT' if signal=='LONG' else 'LONG'
    same=live_position(symbol,signal); opp=live_position(symbol,opposite)
    if same and not opp:
        emit(st,symbol,'IGNORE_SAME_SIDE',signal,price,candle,'exchange position already matches signal'); return
    for p,amt in opp:
        place_market(symbol,opposite,amt,False)
        emit(st,symbol,'CLOSE_SENT',opposite,price,candle,f'qty={amt}')
    if opp and not wait_side_flat(symbol,opposite):
        raise RuntimeError(f'{opposite} did not close; reverse aborted')
    if live_position(symbol,signal):
        emit(st,symbol,'IGNORE_SAME_SIDE',signal,price,candle,'matching exchange position remains'); return
    if available_usdt() < (NOTIONAL_USDT/LEVERAGE)*1.10:
        raise RuntimeError('insufficient available margin for configured notional/leverage')
    set_cross_and_leverage(symbol)
    qty=order_qty(symbol,price)
    place_market(symbol,signal,qty,True)
    time.sleep(1)
    actual=live_position(symbol,signal)
    if not actual: raise RuntimeError('open order sent but position not confirmed')
    p,_=actual[0]
    if is_isolated(p): raise RuntimeError('position opened ISOLATED, expected CROSS')
    lev=int(float(p.get('leverage',0) or 0))
    if lev!=LEVERAGE: print(f'[PC LIVE] WARNING {format_symbol(symbol)} exchange leverage={lev}, requested={LEVERAGE}')
    emit(st,symbol,'OPEN_CONFIRMED',signal,float(p.get('avgPrice',price) or price),candle,f"qty={abs(float(p.get('positionAmt',0) or 0))} cross=True leverage={lev}")


def process_symbol(st,symbol):
    d=get_klines(symbol); now=pd.Timestamp.now(tz='UTC'); closed=d[d.time+pd.Timedelta(minutes=30)<=now].copy()
    if len(closed)<60: return
    pc=purple_cloud(closed); row=pc.iloc[-1]; candle=closed.iloc[-1].time.isoformat(); price=float(closed.iloc[-1].close)
    signal='LONG' if bool(row.pc_buy) else ('SHORT' if bool(row.pc_sell) else None)
    if not signal or st['last_signal_candle'].get(symbol)==candle: return
    st['last_signal_candle'][symbol]=candle; save_state(st)
    if not PAPER:
        execute_live(st,symbol,signal,price,candle); return
    old=st['positions'].get(symbol)
    if old and old['side']==signal: emit(st,symbol,'IGNORE_SAME_SIDE',signal,price,candle); return
    if old:
        pnl=NOTIONAL_USDT*((price/old['entry']-1) if old['side']=='LONG' else (old['entry']/price-1)); emit(st,symbol,'CLOSE',old['side'],price,candle,f'paper_pnl={pnl:.4f} USDT'); st['positions'].pop(symbol,None)
    st['positions'][symbol]={'side':signal,'entry':price,'opened_candle':candle,'notional_usdt':NOTIONAL_USDT,'leverage':LEVERAGE}; emit(st,symbol,'OPEN',signal,price,candle)


def main():
    mode='PAPER' if PAPER else ('LIVE' if LIVE_CONFIRM else 'LIVE_BLOCKED')
    print(f'[PC] NORMAL 30M | {len(SYMBOLS)} symbols | ${NOTIONAL_USDT} notional | {LEVERAGE}x | MODE={mode}')
    try: verify_account()
    except Exception as e: print(f'[PC API] VERIFY FAILED: {type(e).__name__}: {e}')
    if not PAPER and not LIVE_CONFIRM: print('[PC SAFETY] PC_PAPER=false but live confirmation is missing: NO orders will be sent.')
    while True:
        st=load_state()
        for symbol in SYMBOLS:
            try: process_symbol(st,symbol)
            except Exception as e: print('[PC]',symbol,'error:',type(e).__name__,e)
            save_state(st)
        time.sleep(SCAN_SECONDS)

if __name__=='__main__': main()
