#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X — NB121 | Fair Baselines and Vendor-Rule Ablation

**Research gate:** NB120 showed small held-out improvements for a January-trained vendor-conditioned path. This notebook tests whether that improvement is actually caused by IEEE-754 path diversity, or simply by generic prediction and zlib-friendly residuals.

**No novelty claim.** We use identical input rows, predictor-dependent XOR residuals, zlib level 9, and counted JSON headers. A previous-value predictor provides a non-arithmetic baseline. All methods are bit-exact. The target-only comparison assumes the eight charge columns and VendorID are already decoded; the standalone bundle counts those side columns identically across methods. We do not claim a fair comparison to full Parquet or published FPC/ALP implementations.

**Important:** NB120 chose vendor paths by exact-match counts, not compressed bytes. Here we also learn a vendor rule using *actual compressed training residual size* and compare it with fixed paths and a vendor-label permutation control. February data is held out.
"""

# %pip -q install numpy pandas pyarrow


import os, json, zlib, time, zipfile, platform, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
OUT=Path('/content/SGX_NB121_results'); OUT.mkdir(exist_ok=True)
DATASETS={'yellow_2024_01':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet','yellow_2024_02':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet','green_2024_02':'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-02.parquet'}
CHARGES=['fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge','congestion_surcharge','airport_fee']
NROWS=125000

def load(name,url):
    path=Path('/content')/(name+'.parquet')
    if not path.exists():
        print('Downloading',name,flush=True); urllib.request.urlretrieve(url,path)
    pf=pq.ParquetFile(path); cols=pf.schema_arrow.names
    use=['total_amount']+[x for x in CHARGES if x in cols]+(['VendorID'] if 'VendorID' in cols else [])
    frames=[]; count=0
    for batch in pf.iter_batches(batch_size=32768,columns=use):
        frames.append(batch.to_pandas()); count+=len(frames[-1]);
        if count>=NROWS: break
    df=pd.concat(frames,ignore_index=True).iloc[:NROWS].copy()
    for col in CHARGES:
        if col not in df: df[col]=0.0
    for col in CHARGES+['total_amount']:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    mask=np.isfinite(df[CHARGES+['total_amount']].to_numpy(dtype=np.float64)).all(axis=1)
    df=df.loc[mask].reset_index(drop=True)
    v=pd.to_numeric(df['VendorID'],errors='coerce').fillna(-1).to_numpy(dtype=np.int64) if 'VendorID' in df else np.full(len(df),-1,dtype=np.int64)
    return df[CHARGES].to_numpy(dtype='<f8'),df.total_amount.to_numpy(dtype='<f8'),v
D={k:load(k,u) for k,u in DATASETS.items()}
print('Rows:',{k:len(x[1]) for k,x in D.items()})


def bits(y): return np.ascontiguousarray(y,dtype='<f8').view('<u8')
def compress(x): return zlib.compress(np.ascontiguousarray(x).tobytes(),9)
def paths(a):
    orders=[list(range(8)),list(reversed(range(8))),[0,2,4,6,1,3,5,7],[7,5,3,1,6,4,2,0],[0,1,3,2,4,5,7,6]]
    out=[]
    for order in orders:
        s=np.zeros(len(a),dtype=np.float64)
        for j in order: s=np.add(s,a[:,j],dtype=np.float64)
        out.append(s.view('<u8').copy())
    return np.stack(out)

def prev_predictor(y):
    # Decoder can reproduce this only from previously decoded target words.
    pred=np.empty(len(y),dtype='<u8'); pred[0]=0; pred[1:]=bits(y)[:-1]
    return pred

def map_by_matches(train):
    a,y,v=train; p=paths(a); target=bits(y); mapping={}
    for vendor in np.unique(v):
        m=(v==vendor); mapping[int(vendor)]=int(np.argmax(np.count_nonzero(p[:,m]==target[m],axis=1)))
    return mapping

def map_by_bytes(train):
    a,y,v=train; p=paths(a); target=bits(y); mapping={}
    for vendor in np.unique(v):
        m=(v==vendor)
        mapping[int(vendor)]=min(range(5),key=lambda j:len(compress(np.bitwise_xor(target[m],p[j,m]).astype('<u8'))))
    return mapping

MATCH_RULE=map_by_matches(D['yellow_2024_01'])
BYTE_RULE=map_by_bytes(D['yellow_2024_01'])
print('Exact-match rule:',MATCH_RULE,'Compressed-byte rule:',BYTE_RULE)


"""
## Methodology

Fixed0–Fixed4: five candidate arithmetic orders. Vendor-match and vendor-bytes: vendor-conditioned choices trained **only on January yellow taxis**. Vendor-permuted: swap the learned choices for vendor IDs 1 and 2 to test whether the association matters. Previous-value predictor: conventional temporal XOR prediction (decoder uses prior decoded word). Zero-predictor: compress original bit patterns with the same residual packaging. Raw-zlib: plain compressed target without a codec header. Every header, residual, and selector is included in codec bytes. No oracle selection is needed for this causal ablation.

A path-based predictor's side columns are always included in the bundle; previous-value and zero methods are charged the same side columns for comparability. The baseline remains raw-target zlib plus those same side columns. This isolates the target coding effect, **not** whole-file compression.
"""

def header(mode,n,rule=None):
    return json.dumps({'version':121,'mode':mode,'rows':n,'vendor_map':rule},sort_keys=True,separators=(',',':')).encode()

def encode_decode(a,y,v,mode,rule=None):
    n=len(y); truth=bits(y); p=paths(a)
    if mode.startswith('fixed'):
        j=int(mode[5:]); pred=p[j]; index=np.full(n,j,dtype=np.uint8)
    elif mode.startswith('vendor'):
        index=np.fromiter((rule.get(int(k),0) for k in v),dtype=np.uint8,count=n)
        pred=p[index,np.arange(n)]
    elif mode=='previous':
        pred=prev_predictor(y); index=None
    elif mode=='zero':
        pred=np.zeros(n,dtype='<u8'); index=None
    else: raise ValueError(mode)
    residual=np.bitwise_xor(truth,pred).astype('<u8')
    hb=header(mode,n,rule if mode.startswith('vendor') else None)
    rb=compress(residual)
    # Independent decode. Previous predictor is reconstructed sequentially.
    h=json.loads(hb); rec=np.frombuffer(zlib.decompress(rb),dtype='<u8').copy()
    assert len(rec)==n
    if mode.startswith('fixed'): restored=np.bitwise_xor(p[int(mode[5:])],rec)
    elif mode.startswith('vendor'):
        idx=np.fromiter((h['vendor_map'].get(str(int(k)),0) for k in v),dtype=np.uint8,count=n)
        restored=np.bitwise_xor(p[idx,np.arange(n)],rec)
    elif mode=='zero': restored=rec
    else:
        restored=np.empty(n,dtype='<u8'); prev=np.uint64(0)
        for i in range(n):
            restored[i]=np.bitwise_xor(prev,rec[i]); prev=restored[i]
    assert np.array_equal(restored,truth),f'Bit-exact round trip failed: {mode}'
    return len(hb)+len(rb),len(hb),len(rb)

# Adversarial reconstruction: signed zero, infinities, NaN payload, subnormal, largest finite.
weird=np.array([0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,1,0x7fefffffffffffff],dtype='<u8')
a=np.zeros((len(weird),8),dtype='<f8'); v=np.array([1,2,1,2,1,2,1],dtype=np.int64)
for m in [*(f'fixed{i}' for i in range(5)),'vendor_match','vendor_bytes','vendor_permuted','previous','zero']:
    rule= MATCH_RULE if m=='vendor_match' else BYTE_RULE if m=='vendor_bytes' else {1:BYTE_RULE.get(2,0),2:BYTE_RULE.get(1,0)} if m=='vendor_permuted' else None
    encode_decode(a,weird.view('<f8'),v,m if not m.startswith('vendor_') else 'vendor_'+m.split('_',1)[1],rule)
print('All adversarial bit-pattern tests passed')


ROWS=[]
permuted={1:BYTE_RULE.get(2,0),2:BYTE_RULE.get(1,0)}
methods=[*(f'fixed{i}' for i in range(5)),'vendor_match','vendor_bytes','vendor_permuted','previous','zero']
for name,(a,y,v) in D.items():
    side=len(compress(a.astype('<f8')))+len(compress(v.astype('<i8')))
    raw=len(compress(bits(y)))
    for mode in methods:
        rule=MATCH_RULE if mode=='vendor_match' else BYTE_RULE if mode=='vendor_bytes' else permuted if mode=='vendor_permuted' else None
        t=time.perf_counter(); size,hb,rb=encode_decode(a,y,v,mode,rule); elapsed=time.perf_counter()-t
        ROWS.append(dict(dataset=name,mode=mode,rows=len(y),target_bytes=size,raw_target_zlib_bytes=raw,delta_vs_raw_bytes=size-raw,side_columns_zlib_bytes=side,bundle_bytes=side+size,raw_bundle_bytes=side+raw,header_bytes=hb,residual_bytes=rb,roundtrip=True,encode_plus_decode_seconds=elapsed))
res=pd.DataFrame(ROWS)
res.to_csv(OUT/'ablation_results.csv',index=False)
display(res.pivot(index='mode',columns='dataset',values='target_bytes'))


held=res[res.dataset.isin(['yellow_2024_02','green_2024_02'])]
p=held.pivot(index='dataset',columns='mode',values='target_bytes')
fixed_best=p[[f'fixed{i}' for i in range(5)]].min(axis=1)
comparison=pd.DataFrame({'best_fixed_path':fixed_best,'vendor_match':p.vendor_match,'vendor_bytes':p.vendor_bytes,'vendor_permuted':p.vendor_permuted,'previous':p.previous,'zero':p.zero,'raw_zlib':held.drop_duplicates('dataset').set_index('dataset').raw_target_zlib_bytes})
comparison['vendor_bytes_gain_vs_best_fixed']=comparison.best_fixed_path-comparison.vendor_bytes
comparison['vendor_bytes_gain_vs_previous']=comparison.previous-comparison.vendor_bytes
comparison.to_csv(OUT/'holdout_decision_table.csv')
display(comparison)
# A narrow predeclared go/no-go gate: vendor-byte rule must beat both fixed and previous on BOTH holdouts.
gate=bool(((comparison.vendor_bytes_gain_vs_best_fixed>0)&(comparison.vendor_bytes_gain_vs_previous>0)).all())
decision={'notebook':'NB121','exact_roundtrip_all':bool(res.roundtrip.all()),'january_training_only':True,'heldout_datasets':['yellow_2024_02','green_2024_02'],'vendor_bytes_beats_best_fixed_and_previous_both_holdouts':gate,'novelty_established':False,'recommendation':('Proceed to targeted novelty/prior-art audit and broader datasets, without claiming novelty' if gate else 'Do not extend vendor path-selection as the main novelty claim; reconsider algorithmic direction')}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
(OUT/'rules.json').write_text(json.dumps({'match_rule':MATCH_RULE,'byte_rule':BYTE_RULE,'permuted_rule':permuted},indent=2))
(OUT/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'datasets':DATASETS,'rows_limit':NROWS},indent=2))
(OUT/'README.md').write_text('NB121: measured zlib XOR residual baselines and vendor ablations; full IEEE-754 word roundtrips. Target-column bundles, NOT original Parquet sizes. No novelty claim.\n')
print(json.dumps(decision,indent=2))
archive='/content/SGX_NB121_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for f in OUT.iterdir():z.write(f,arcname=f.name)
print('Results saved:',archive)
try:
    from google.colab import files
    files.download(archive)
except ImportError:
    print('Outside Colab: download results manually from',archive)

