from __future__ import annotations
import hashlib, hmac, json, os, time
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
STATE_PATH = Path(os.getenv('PC_STATE_PATH', '/data/purple_cloud_state.json'))
SCAN_SECONDS = int(os.getenv('PC_SCAN_SECONDS', '60'))
PERIOD = 20
ALPHA = 1.5
BPT = 0.2
SPT = 0.2


def format_symbol(symbol):
    s=symbol.upper().replace('-','')
    return f'{s[:-4]}-USDT' if s.endswith('USDT') else s


def get_klines(symbol, limit=250):
    url=f'{BINGX_BASE_URL}/openApi/swap/v3/quote/klines'
    params={'symbol':format_symbol(symbol),'interval':INTERVAL,'limit':limit}
    r=requests.get(url,params=params,headers={'User-Agent':'PurpleCloudBot'},timeout=20)
    r.raise_for_status(); payload=r.json()
    if payload.get('code') not in (0,'0',None): raise RuntimeError(f"BingX {payload.get('code')}: {payload.get('msg')}")
    rows=[]
    for x in payload.get('data',[]):
        if isinstance(x,list) and len(x)>=6: rows.append(x[:6])
        elif isinstance(x,dict): rows.append([x.get('time') or x.get('openTime'),x.get('open'),x.get('high'),x.get('low'),x.get('close'),x.get('volume')])
    d=pd.DataFrame(rows,columns=['time','open','high','low','close','volume'])
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d['time']=pd.to_datetime(pd.to_numeric(d['time'],errors='coerce'),unit='ms',utc=True)
    return d.dropna().sort_values('time').drop_duplicates('time').reset_index(drop=True)


def signed_get(path, params=None):
    if not BINGX_API_KEY or not BINGX_SECRET_KEY:
        raise RuntimeError('BINGX_API_KEY/BINGX_SECRET_KEY missing')
    p=dict(params or {}); p['timestamp']=int(time.time()*1000)
    query=urlencode(sorted(p.items()))
    sig=hmac.new(BINGX_SECRET_KEY.encode(),query.encode(),hashlib.sha256).hexdigest()
    r=requests.get(f'{BINGX_BASE_URL}{path}?{query}&signature={sig}',headers={'X-BX-APIKEY':BINGX_API_KEY,'User-Agent':'PurpleCloudBot'},timeout=20)
    r.raise_for_status(); data=r.json()
    if data.get('code') not in (0,'0',None): raise RuntimeError(f"BingX {data.get('code')}: {data.get('msg')}")
    return data.get('data')


def verify_account_read_only():
    print('[PC API] Read-only account verification starting; NO orders will be sent.')
    if not BINGX_API_KEY or not BINGX_SECRET_KEY:
        print('[PC API] SKIP: BingX API credentials are not configured.')
        return
    try:
        bal=signed_get('/openApi/swap/v2/user/balance') or {}
        b=bal.get('balance',bal) if isinstance(bal,dict) else bal
        asset=b.get('asset','USDT') if isinstance(b,dict) else 'USDT'
        available=b.get('availableMargin',b.get('availableBalance','?')) if isinstance(b,dict) else '?'
        equity=b.get('equity',b.get('balance','?')) if isinstance(b,dict) else '?'
        print(f'[PC API] BALANCE OK | asset={asset} | equity={equity} | available={available}')
        pos=signed_get('/openApi/swap/v2/user/positions') or []
        if isinstance(pos,dict): pos=pos.get('positions',pos.get('data',[]))
        opened=[]
        for p in pos if isinstance(pos,list) else []:
            try:
                amt=float(p.get('positionAmt',p.get('positionAmount',0)) or 0)
                if abs(amt)>0: opened.append(f"{p.get('symbol','?')}:{p.get('positionSide',p.get('side','?'))}:{amt}")
            except Exception: pass
        print(f'[PC API] POSITIONS OK | open={len(opened)}' + (f" | {'; '.join(opened[:20])}" if opened else ''))
        print('[PC API] VERIFIED: credentials can read BingX account. Trading remains blocked while PAPER=True.')
    except Exception as e:
        print(f'[PC API] VERIFY FAILED: {type(e).__name__}: {e}')


def rma(s,n): return s.ewm(alpha=1/n,adjust=False).mean()
def atr(d,n):
    p=d.close.shift(1)
    tr=pd.concat([d.high-d.low,(d.high-p).abs(),(d.low-p).abs()],axis=1).max(axis=1)
    return rma(tr,n)
def vwma(x,v,n): return (x*v).rolling(n).sum()/v.rolling(n).sum().replace(0,np.nan)


def purple_cloud(d):
    d=d.copy().reset_index(drop=True)
    n1=int(np.ceil(PERIOD/4)); n2=int(np.ceil(PERIOD/2)); x2=atr(d,PERIOD)*ALPHA
    xh=d.close+x2; xl=d.close-x2; hl2=(d.high+d.low)/2
    a1=vwma(hl2*d.volume,d.volume,n1)/vwma(d.volume,d.volume,n1)
    a2=vwma(hl2*d.volume,d.volume,n2)/vwma(d.volume,d.volume,n2)
    a3=2*a1-a2; a4=vwma(a3,d.volume,PERIOD); b1=rma(d.close,PERIOD); a5=2*a4*b1/(a4+b1)
    buy=(a5<=xl)&(d.close>b1*(1+BPT*0.01)); sell=(a5>=xh)&(d.close<b1*(1-SPT*0.01))
    xs=np.zeros(len(d),dtype=int)
    for i in range(1,len(d)): xs[i]=1 if bool(buy.iloc[i]) else (-1 if bool(sell.iloc[i]) else xs[i-1])
    changed=pd.Series(xs).ne(pd.Series(xs).shift(1))
    d['pc_buy']=buy&changed; d['pc_sell']=sell&changed
    return d


def load_state():
    try:
        if STATE_PATH.exists(): return json.loads(STATE_PATH.read_text())
    except Exception as e: print('[PC] state read error:',e)
    return {'positions':{},'last_signal_candle':{},'events':[]}


def save_state(st):
    STATE_PATH.parent.mkdir(parents=True,exist_ok=True)
    tmp=STATE_PATH.with_suffix('.tmp'); tmp.write_text(json.dumps(st,indent=2)); tmp.replace(STATE_PATH)


def emit(st,symbol,action,side,price,candle,note=''):
    ev={'ts':datetime.now(timezone.utc).isoformat(),'symbol':symbol,'action':action,'side':side,'price':price,'signal_candle':candle,'notional_usdt':NOTIONAL_USDT,'leverage':LEVERAGE,'mode':'PAPER' if PAPER else 'LIVE_BLOCKED','note':note}
    st['events'].append(ev); st['events']=st['events'][-1000:]; print('[PC EVENT]',json.dumps(ev))


def process_symbol(st,symbol):
    d=get_klines(symbol)
    now=pd.Timestamp.now(tz='UTC'); closed=d[d.time+pd.Timedelta(minutes=30)<=now].copy()
    if len(closed)<60: return
    pc=purple_cloud(closed); row=pc.iloc[-1]; candle=closed.iloc[-1].time.isoformat(); price=float(closed.iloc[-1].close)
    signal='LONG' if bool(row.pc_buy) else ('SHORT' if bool(row.pc_sell) else None)
    if not signal or st['last_signal_candle'].get(symbol)==candle: return
    st['last_signal_candle'][symbol]=candle; old=st['positions'].get(symbol)
    if old and old['side']==signal: emit(st,symbol,'IGNORE_SAME_SIDE',signal,price,candle); return
    if old:
        pnl=NOTIONAL_USDT*((price/old['entry']-1) if old['side']=='LONG' else (old['entry']/price-1))
        emit(st,symbol,'CLOSE',old['side'],price,candle,f'paper_pnl={pnl:.4f} USDT'); st['positions'].pop(symbol,None)
    if PAPER:
        st['positions'][symbol]={'side':signal,'entry':price,'opened_candle':candle,'notional_usdt':NOTIONAL_USDT,'leverage':LEVERAGE}
        emit(st,symbol,'OPEN',signal,price,candle)
    else: emit(st,symbol,'LIVE_BLOCKED',signal,price,candle,'Real orders intentionally disabled during paper validation')


def main():
    print(f'[PC] NORMAL 30M | {len(SYMBOLS)} symbols | ${NOTIONAL_USDT} notional | {LEVERAGE}x | PAPER={PAPER}')
    verify_account_read_only()
    while True:
        st=load_state()
        for symbol in SYMBOLS:
            try: process_symbol(st,symbol)
            except Exception as e: print('[PC]',symbol,'error:',e)
            save_state(st)
        time.sleep(SCAN_SECONDS)

if __name__=='__main__': main()
