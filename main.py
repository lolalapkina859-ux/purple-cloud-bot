from __future__ import annotations
import csv, hashlib, hmac, json, math, os, time
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlencode
import numpy as np
import pandas as pd
import requests

BINGX_BASE_URL = "https://open-api.bingx.com"
BINGX_API_KEY = os.getenv('BINGX_API_KEY', '')
BINGX_SECRET_KEY = os.getenv('BINGX_SECRET_KEY', '')
SYMBOLS = ['ZECUSDT','USELESSUSDT','FETUSDT','HYPEUSDT','JTOUSDT','VETUSDT','XRPUSDT','ETHUSDT','UNIUSDT','INJUSDT','SEIUSDT','1000SHIBUSDT','DYDXUSDT','NEARUSDT','FLOWUSDT']
INTERVAL = '30m'
NOTIONAL_USDT = float(os.getenv('PC_NOTIONAL_USDT', '60'))
LEVERAGE = int(os.getenv('PC_LEVERAGE', '20'))
SYMBOL_LEVERAGE = {'FLOWUSDT': 10}

def leverage_for(symbol):
    return min(LEVERAGE, SYMBOL_LEVERAGE.get(symbol.upper(), LEVERAGE))
PAPER = os.getenv('PC_PAPER', 'true').lower() == 'true'
LIVE_CONFIRM = os.getenv('PC_LIVE_CONFIRM', '') == 'I_UNDERSTAND_LIVE_TRADING'
STATE_PATH = Path(os.getenv('PC_STATE_PATH', '/data/purple_cloud_state.json'))
JOURNAL_PATH = Path(os.getenv('PC_JOURNAL_PATH', '/data/trade_journal.csv'))
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
    target_leverage=leverage_for(symbol)
    mt=signed_get('/openApi/swap/v2/trade/marginType',{'symbol':fs}) or {}
    current=str(mt.get('marginType','')).upper() if isinstance(mt,dict) else ''
    if current!='CROSSED':
        signed_post('/openApi/swap/v2/trade/marginType',{'symbol':fs,'marginType':'CROSSED'})
    lev=signed_get('/openApi/swap/v2/trade/leverage',{'symbol':fs}) or {}
    for side,key in (('LONG','longLeverage'),('SHORT','shortLeverage')):
        try: current_lev=int(float(lev.get(key,0) or 0))
        except Exception: current_lev=0
        if current_lev!=target_leverage:
            signed_post('/openApi/swap/v2/trade/leverage',{'symbol':fs,'side':side,'leverage':target_leverage})


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


def balance_record():
    """Return the USDT futures balance record without confusing numeric 'balance' with a nested object."""
    raw=signed_get('/openApi/swap/v3/user/balance') or {}
    data=raw
    # Some API/wrapper variants may nest the record under data/balance.
    # Only unwrap when the nested value is itself a dict/list.
    if isinstance(data,dict) and isinstance(data.get('data'),(dict,list)):
        data=data['data']
    if isinstance(data,dict) and isinstance(data.get('balance'),(dict,list)):
        data=data['balance']
    if isinstance(data,list):
        return next((x for x in data if isinstance(x,dict) and x.get('asset')=='USDT'), data[0] if data and isinstance(data[0],dict) else {})
    return data if isinstance(data,dict) else {}


def available_usdt():
    b=balance_record()
    value=b.get('availableMargin',b.get('availableBalance'))
    if value is None:
        raise RuntimeError(f"balance response missing available margin; keys={sorted(b.keys())}")
    return float(value)


def verify_account():
    print('[PC API] Account verification starting; orders only possible when PAPER=False AND live confirmation is set.')
    ensure_hedge_mode()
    b=balance_record()
    print(f"[PC API] BALANCE OK | asset={b.get('asset','USDT')} | balance={b.get('balance','?')} | equity={b.get('equity','?')} | available={b.get('availableMargin',b.get('availableBalance','?'))} | used={b.get('usedMargin','?')} | frozen={b.get('frozenMargin',b.get('freezedMargin','?'))}")
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


def journal_trade(symbol,side,entry_time,entry_price,exit_time,exit_price,notional,qty='',source='LIVE'):
    try:
        entry_price=float(entry_price); exit_price=float(exit_price); notional=float(notional)
        ret=(exit_price/entry_price-1) if side=='LONG' else (entry_price/exit_price-1)
        try:
            duration_hours=(pd.Timestamp(exit_time)-pd.Timestamp(entry_time)).total_seconds()/3600
        except Exception: duration_hours=''
        row={'symbol':symbol,'side':side,'entry_time':entry_time,'entry_price':entry_price,'exit_time':exit_time,'exit_price':exit_price,
             'pnl_usdt_est':notional*ret,'return_pct':ret*100,'duration_hours':duration_hours,'notional_usdt':notional,'qty':qty,'source':source}
        JOURNAL_PATH.parent.mkdir(parents=True,exist_ok=True); exists=JOURNAL_PATH.exists()
        with JOURNAL_PATH.open('a',newline='') as fh:
            w=csv.DictWriter(fh,fieldnames=list(row));
            if not exists: w.writeheader()
            w.writerow(row)
        print('[PC JOURNAL]',json.dumps(row))
    except Exception as e: print('[PC JOURNAL] write error:',e)


def execute_live(st,symbol,signal,price,candle):
    if not LIVE_CONFIRM:
        emit(st,symbol,'LIVE_BLOCKED',signal,price,candle,'PC_LIVE_CONFIRM guard not enabled'); return
    ensure_hedge_mode()
    opposite='SHORT' if signal=='LONG' else 'LONG'
    same=live_position(symbol,signal); opp=live_position(symbol,opposite)
    if same and not opp:
        emit(st,symbol,'IGNORE_SAME_SIDE',signal,price,candle,'exchange position already matches signal'); return
    for p,amt in opp:
        entry_px=float(p.get('avgPrice',0) or 0); entry_time=p.get('updateTime') or p.get('createTime') or ''
        place_market(symbol,opposite,amt,False)
        emit(st,symbol,'CLOSE_SENT',opposite,price,candle,f'qty={amt}')
        if entry_px>0:
            journal_trade(symbol,opposite,entry_time,entry_px,candle,price,NOTIONAL_USDT,amt,'LIVE')
    if opp and not wait_side_flat(symbol,opposite):
        raise RuntimeError(f'{opposite} did not close; reverse aborted')
    if live_position(symbol,signal):
        emit(st,symbol,'IGNORE_SAME_SIDE',signal,price,candle,'matching exchange position remains'); return
    target_leverage=leverage_for(symbol)
    available=available_usdt()
    required=(NOTIONAL_USDT/target_leverage)*1.10
    print(f'[PC MARGIN] {symbol} available={available:.4f} required={required:.4f} notional={NOTIONAL_USDT:.2f} leverage={target_leverage}x')
    if available < required:
        raise RuntimeError(f'insufficient available margin: available={available:.4f}, required={required:.4f}, leverage={target_leverage}x')
    set_cross_and_leverage(symbol)
    qty=order_qty(symbol,price)
    place_market(symbol,signal,qty,True)
    time.sleep(1)
    actual=live_position(symbol,signal)
    if not actual: raise RuntimeError('open order sent but position not confirmed')
    p,_=actual[0]
    if is_isolated(p): raise RuntimeError('position opened ISOLATED, expected CROSS')
    lev=int(float(p.get('leverage',0) or 0))
    if lev!=target_leverage: print(f'[PC LIVE] WARNING {format_symbol(symbol)} exchange leverage={lev}, requested={target_leverage}')
    open_px=float(p.get('avgPrice',price) or price); open_qty=abs(float(p.get('positionAmt',0) or 0))
    st['positions'][symbol]={'side':signal,'entry':open_px,'opened_candle':candle,'notional_usdt':NOTIONAL_USDT,'qty':open_qty,'leverage':target_leverage}
    emit(st,symbol,'OPEN_CONFIRMED',signal,open_px,candle,f"qty={open_qty} cross=True leverage={lev}")


def process_symbol(st,symbol):
    d=get_klines(symbol); now=pd.Timestamp.now(tz='UTC'); closed=d[d.time+pd.Timedelta(minutes=30)<=now].copy()
    if len(closed)<60: return
    pc=purple_cloud(closed); row=pc.iloc[-1]; candle=closed.iloc[-1].time.isoformat(); price=float(closed.iloc[-1].close)
    signal='LONG' if bool(row.pc_buy) else ('SHORT' if bool(row.pc_sell) else None)
    if not signal or st['last_signal_candle'].get(symbol)==candle: return
    st['last_signal_candle'][symbol]=candle; save_state(st)
    if not PAPER:
        old=st.get('positions',{}).get(symbol)
        if old and old.get('side')!=signal:
            journal_trade(symbol,old['side'],old.get('opened_candle',''),old.get('entry',price),candle,price,old.get('notional_usdt',NOTIONAL_USDT),old.get('qty',''),'LIVE_STATE')
            st['positions'].pop(symbol,None)
        execute_live(st,symbol,signal,price,candle); return
    old=st['positions'].get(symbol)
    if old and old['side']==signal: emit(st,symbol,'IGNORE_SAME_SIDE',signal,price,candle); return
    if old:
        pnl=NOTIONAL_USDT*((price/old['entry']-1) if old['side']=='LONG' else (old['entry']/price-1)); emit(st,symbol,'CLOSE',old['side'],price,candle,f'paper_pnl={pnl:.4f} USDT'); st['positions'].pop(symbol,None)
    st['positions'][symbol]={'side':signal,'entry':price,'opened_candle':candle,'notional_usdt':NOTIONAL_USDT,'leverage':LEVERAGE}; emit(st,symbol,'OPEN',signal,price,candle)


def main():
    mode='PAPER' if PAPER else ('LIVE' if LIVE_CONFIRM else 'LIVE_BLOCKED')
    print(f'[PC] NORMAL 30M | {len(SYMBOLS)} symbols | ${NOTIONAL_USDT} notional | {LEVERAGE}x | MODE={mode}')
    if not PAPER: verify_account()
    st=load_state()
    while True:
        for s in SYMBOLS:
            try: process_symbol(st,s)
            except Exception as e: print(f'[PC] {s} error: {e}')
        save_state(st); time.sleep(SCAN_SECONDS)

if __name__=='__main__':
    main()
