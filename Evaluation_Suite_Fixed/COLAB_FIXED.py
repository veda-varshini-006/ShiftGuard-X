#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X — NB120 | Real Bit-Exact Arithmetic-Path Codec Gate

**Status:** Experimental, not a novelty claim. This notebook repairs the gap in reconstructed NB119: NB119 only estimated hypothetical coding costs. NB120 measures real bytes with actual reversible streams.

**Research question:** Does encoding a target monetary column using IEEE-754 evaluation-path candidates improve total lossless storage cost when the decoder must also possess all side columns and selector information?

**Controls:** raw-target zlib; fixed left-to-right path; vendor-conditioned path trained on January and evaluated on February; oracle five-path with explicitly encoded selectors. All use the **same** bytewise XOR residual representation and zlib backend. This is a prototype, not a production entropy codec. Vendor decisions use only side information already available to the decoder. **Never infer the original vendor arithmetic.**

**Stopping rule:** If no candidate beats the raw target after including its fair side-information cost on both held-out datasets, stop this mechanism rather than adding more predictors. A positive result still does not establish originality.
"""

# %pip -q install pyarrow pandas numpy


import io, os, json, math, time, zlib, zipfile, hashlib, urllib.request, platform
import numpy as np, pandas as pd
from pathlib import Path
OUT=Path('/content/SGX_NB120_results'); OUT.mkdir(exist_ok=True)
DATASETS={'yellow_2024_01':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet', 'yellow_2024_02':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet', 'green_2024_02':'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-02.parquet'}
CHARGES=['fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge','congestion_surcharge','airport_fee']
NROWS=125000
print('Environment',platform.python_version(),np.__version__,pd.__version__)


"""
## 1 — Download official real-world data

For comparability with NB118, use at most 125,000 rows per file, and retain only rows with finite target and side columns. Parquet nulls become zero **only for optional missing charge fields**, not missing individual charge values. Every tested target is encoded as its original IEEE-754 64-bit word; no rounding or approximate reconstruction is permitted.
"""

import pyarrow.parquet as pq

def get_data(name,url):
    path=Path('/content')/(name+'.parquet')
    if not path.exists():
        print('Downloading',name,flush=True); urllib.request.urlretrieve(url,path)
    schema=pq.ParquetFile(path).schema_arrow.names
    use=['total_amount']+[x for x in CHARGES if x in schema]+(['VendorID'] if 'VendorID' in schema else [])
    frames=[]; got=0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32768,columns=use):
        df=batch.to_pandas(); frames.append(df); got+=len(df)
        if got>=NROWS: break
    df=pd.concat(frames,ignore_index=True).iloc[:NROWS].copy()
    for x in CHARGES:
        if x not in df: df[x]=0.0
    for x in CHARGES+['total_amount']:
        df[x]=pd.to_numeric(df[x],errors='coerce')
    df=df.loc[np.isfinite(df[CHARGES+['total_amount']].to_numpy(dtype=np.float64)).all(axis=1)].reset_index(drop=True)
    vendor=pd.to_numeric(df['VendorID'],errors='coerce').fillna(-1).to_numpy(dtype=np.int64) if 'VendorID' in df else np.full(len(df),-1,dtype=np.int64)
    return df[CHARGES].to_numpy(dtype='<f8'),df['total_amount'].to_numpy(dtype='<f8'),vendor
D={k:get_data(k,u) for k,u in DATASETS.items()}
print({k:len(v[1]) for k,v in D.items()})


"""
## 2 — Define five explicit IEEE-754 evaluation paths

These are **test candidates**, not verified reconstruction formulas for the TLC's billing software. Candidate ordering is fixed, and the same NumPy float64 arithmetic is used in encoder and decoder. A production codec must specify cross-platform IEEE-754 behavior and test other hardware.
"""

def candidates(a):
    # Each operation is a float64 add; no re-association by a compiler/JIT.
    out=[]
    orders=[list(range(8)),list(reversed(range(8))),[0,2,4,6,1,3,5,7],[7,5,3,1,6,4,2,0],[0,1,3,2,4,5,7,6]]
    for order in orders:
        acc=np.zeros(len(a),dtype=np.float64)
        for j in order: acc=np.add(acc,a[:,j],dtype=np.float64)
        out.append(acc.view(np.uint64).copy())
    return np.stack(out)

def rawbits(v): return np.ascontiguousarray(v,dtype='<f8').view(np.uint64)
def bzip(arr): return zlib.compress(np.ascontiguousarray(arr).tobytes(),level=9)
def unbzip(buf,dtype,count):
    a=np.frombuffer(zlib.decompress(buf),dtype=dtype)
    assert len(a)==count
    return a.copy()

def choose_vendor(train):
    a,y,v=train; p=candidates(a); target=rawbits(y)
    mapping={}
    for key in np.unique(v):
        ix=(v==key); scores=np.count_nonzero(p[:,ix]==target[ix],axis=1)
        mapping[int(key)]=int(np.argmax(scores))
    return mapping
VENDOR_RULE=choose_vendor(D['yellow_2024_01'])
print('January-frozen vendor rule:',VENDOR_RULE)


"""
## 3 — Fully reversible payload construction

XOR residuals are 64-bit words: `original_bits XOR predicted_bits`. No residual is discarded. Oracle selection transmits one selector byte per row before zlib; vendor selection transmits the January-frozen mapping in the header. Fixed prediction uses path zero. The header is stored as literal JSON bytes, and every payload byte is counted.
"""

def encode(a,y,v,mode,mapping=None):
    n=len(y); p=candidates(a); truth=rawbits(y)
    if mode=='fixed': idx=np.zeros(n,dtype=np.uint8)
    elif mode=='vendor': idx=np.array([mapping.get(int(k),0) for k in v],dtype=np.uint8)
    elif mode=='oracle':
        match=p==truth[None,:]
        idx=np.where(match.any(axis=0),match.argmax(axis=0),0).astype(np.uint8)
    else: raise ValueError(mode)
    prediction=p[idx,np.arange(n)]
    residual=np.bitwise_xor(truth,prediction).astype('<u8')
    header={'version':1,'mode':mode,'rows':n,'charges':CHARGES,'vendor_map':mapping if mode=='vendor' else None}
    hb=json.dumps(header,sort_keys=True,separators=(',',':')).encode()
    pieces=[hb,bzip(residual)]
    if mode=='oracle': pieces.append(bzip(idx))
    return pieces,{'header':len(hb),'residual':len(pieces[1]),'selector':len(pieces[2]) if mode=='oracle' else 0},idx

def decode(a,v,pieces):
    h=json.loads(pieces[0]); n=h['rows']; assert len(a)==len(v)==n
    mode=h['mode']; p=candidates(a)
    if mode=='fixed': idx=np.zeros(n,dtype=np.uint8)
    elif mode=='vendor': idx=np.array([h['vendor_map'].get(str(int(k)),0) for k in v],dtype=np.uint8)
    else: idx=unbzip(pieces[2],np.uint8,n)
    assert np.all(idx<5)
    residual=unbzip(pieces[1],'<u8',n)
    return np.bitwise_xor(p[idx,np.arange(n)],residual)

# Adversarial bit-pattern controls, including signed zero and NaN payloads.
a=np.zeros((7,8),dtype='<f8'); v=np.array([1,2,1,2,-1,1,2])
weird=np.array([0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0x0000000000000001,0x7fefffffffffffff],dtype='<u8')
for mode in ['fixed','vendor','oracle']:
    pieces,_,_=encode(a,weird.view('<f8'),v,mode,VENDOR_RULE)
    assert np.array_equal(decode(a,v,pieces),weird)
print('All adversarial bitwise round trips passed')


"""
## 4 — Fair measured size comparison

Report both **target-only** bytes (when all charge columns and VendorID are already available to the decoder) and **standalone** bytes (charge and VendorID side information included). The standalone control compresses raw target together with the same side columns, so there is no hidden free input. Charge-column and VendorID bytes are counted identically for each method. This measures a column-bundle prototype, not full Parquet compression.
"""

rows=[]; details=[]
for name,(a,y,v) in D.items():
    n=len(y); target=rawbits(y); side_charge=bzip(a.astype('<f8')); side_vendor=bzip(v.astype('<i8'))
    side=len(side_charge)+len(side_vendor)
    raw=bzip(target.astype('<u8'))
    baseline=len(raw)
    for mode in ['fixed','vendor','oracle']:
        mapping=VENDOR_RULE if mode=='vendor' else None
        t=time.perf_counter(); pieces,sizes,idx=encode(a,y,v,mode,mapping); encode_s=time.perf_counter()-t
        t=time.perf_counter(); restored=decode(a,v,pieces); decode_s=time.perf_counter()-t
        assert np.array_equal(restored,target),f'Bitwise roundtrip FAILED: {name}/{mode}'
        cost=sum(map(len,pieces))
        rows.append(dict(dataset=name,mode=mode,rows=n,raw_target_zlib_bytes=baseline,codec_target_bytes=cost,target_delta_bytes=cost-baseline,side_columns_zlib_bytes=side,raw_bundle_bytes=side+baseline,codec_bundle_bytes=side+cost,bundle_delta_bytes=cost-baseline,exact_roundtrip=True,encode_seconds=encode_s,decode_seconds=decode_s,**sizes))
        details.append(dict(dataset=name,mode=mode,selected_counts=np.bincount(idx,minlength=5).tolist(),exact_matches=int(np.count_nonzero(candidates(a)[idx,np.arange(n)]==target))))
res=pd.DataFrame(rows); display(res)
res.to_csv(OUT/'measured_codec_results.csv',index=False)
(OUT/'selection_details.json').write_text(json.dumps(details,indent=2))


"""
## 5 — Scientific decision and reproducibility

Do not claim a new compression algorithm solely from a positive byte delta. **Even the oracle codec uses encoder-only knowledge and explicitly transmits selectors**, while vendor mode has a frozen training rule. This notebook does not benchmark the published FPC/ALP implementations and cannot establish state-of-the-art performance.
"""

held=res[res.dataset.isin(['yellow_2024_02','green_2024_02'])]
by=held.pivot(index='dataset',columns='mode',values='bundle_delta_bytes')
print('Held-out bytes relative to raw baseline (negative is better):'); display(by)
passes=bool((by[['vendor','oracle']]<0).any(axis=1).all())
operational=bool((by['vendor']<0).all())
decision={'notebook':'NB120','source':'official NYC TLC parquet','jan_vendor_training_only':True,'holdout_datasets':['yellow_2024_02','green_2024_02'],'bitwise_roundtrip_all':bool(res.exact_roundtrip.all()),'vendor_improves_both_holdouts':operational,'some_path_method_improves_each_holdout':passes,'novelty_established':False,'recommendation':('Proceed to NB121 rigorous baseline and ablation audit' if operational else 'Stop or redesign this arithmetic-path selection mechanism; do not claim novelty or compression gain')}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
(OUT/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'datasets':DATASETS,'rows_limit':NROWS,'vendor_rule':VENDOR_RULE},indent=2))
(OUT/'README.md').write_text('ShiftGuard-X NB120: measured, reversible arithmetic-path codec prototype. Compare measured_codec_results.csv. All size claims refer to zlib-compressed binary column bundles, not original Parquet. No novelty established.\n')
print(json.dumps(decision,indent=2))
archive='/content/SGX_NB120_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for f in OUT.iterdir(): z.write(f,arcname=f.name)
print('Saved',archive)
from google.colab import files
files.download(archive)
print('If automatic download is blocked: open the Colab Files sidebar and download SGX_NB120_results.zip manually.')

