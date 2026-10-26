#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X — NB127 | Specialized Float Codec Provenance & Correctness Gate

**Objective:** Advance NB126 with an independently implemented Chimp-family baseline and verify reproducible access to the **official ALP and Chimp source repositories**. **This notebook does NOT claim to benchmark official ALP**: official C++ ALP primitives need a separate, audited stream serialization wrapper, and the official Chimp Java test harness does not provide a documented Python standalone interface. Instead, an independently maintained `floatbungler` Chimp-family implementation is benchmarked and explicitly labeled *third-party*, not *official*. This avoids misleading measurements.

**Previous evidence:** NB125–NB126 established pcodec as the best measured working codec on the taxi subsets; original NB126 ZIP should remain preserved. NB127 makes no novelty claim.

**Principles:** Identical input binary64 words; measured standalone codec payload sizes plus explicit per-file headers; bitwise reconstruction mandatory; skip unsupported or incorrect codecs rather than report them as successful.

**Upstream reference:** https://github.com/cwida/ALP and https://github.com/panagiotisl/chimp; third-party Chimp implementation https://github.com/miikka/floatbungler. Full official-implementation head-to-head still requires separately verified format wrappers.
"""

# %pip -q install numpy pandas pyarrow zstandard pcodec floatbungler


import os, sys, json, time, zlib, zipfile, hashlib, importlib, subprocess, platform
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
import zstandard as zstd
OUT=Path('/content/SGX_NB127_results');OUT.mkdir(exist_ok=True)
URLS={
 'yellow_jan_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet',
 'yellow_feb_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet',
 'green_feb_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-02.parquet'}
MAX_VALUES=125000
print(platform.platform(),sys.version, 'numpy',np.__version__)


import urllib.request

def load(name,url):
    path=Path('/content')/(name+'.parquet')
    if not path.exists():urllib.request.urlretrieve(url,path)
    pf=pq.ParquetFile(path)
    vals=[];seen=0
    for batch in pf.iter_batches(batch_size=32768,columns=['total_amount']):
        x=np.asarray(batch.to_pandas(),dtype='<f8');vals.append(x);seen+=len(x)
        if seen>=MAX_VALUES:break
    data=np.concatenate(vals)[:MAX_VALUES]
    # Follow earlier finite-only experimental cohort, avoiding unstated null-to-NaN payload assumptions.
    data=np.ascontiguousarray(data[np.isfinite(data)],dtype='<f8')
    return data
D={k:load(k,v) for k,v in URLS.items()}
print({k:len(v) for k,v in D.items()})


"""
## Optional upstream provenance audit

The cell below attempts to pin official repositories to specific commits. It **does not falsely substitute the Java Chimp unit tests or a DuckDB database file for a fair standalone codec benchmark**. Network/build errors are recorded rather than silently ignored.
"""

repos={'official_ALP':'https://github.com/cwida/ALP.git','official_Chimp':'https://github.com/panagiotisl/chimp.git'}
upstreams={}
for name,url in repos.items():
    folder=Path('/content')/name
    try:
        if not folder.exists():
            r=subprocess.run(['git','clone','-q','--depth','1',url,str(folder)],capture_output=True,text=True,timeout=150)
            if r.returncode:raise RuntimeError(r.stderr[-500:])
        r=subprocess.run(['git','-C',str(folder),'rev-parse','HEAD'],capture_output=True,text=True,timeout=15,check=True)
        upstreams[name]={'status':'source_available','git_commit':r.stdout.strip(),'codec_benchmarked':False}
    except Exception as e: upstreams[name]={'status':'unavailable','reason':str(e)[:500],'codec_benchmarked':False}
print(json.dumps(upstreams,indent=2))
(OUT/'official_source_provenance.json').write_text(json.dumps(upstreams,indent=2))


"""
## Reversible size audit

The independent Python Chimp-family implementation is **not the original author implementation** and is included as a screening baseline only. Pcodec's installed API version is detected by a minimal control roundtrip; if incompatible, the method is marked unavailable. A codec counts as `verified` only after exact reconstruction with `uint64`-bit comparison. Compressed payload bytes, a fixed 12-byte independent framing allowance, and wall-clock times are recorded. This same 12-byte overhead is charged to every codec; it is a reporting convention, not a serialized common file-format implementation.
"""

import traceback
from floatbungler import chimp,chimp128, gorilla, patas
import pcodec
print('pcodec module',getattr(pcodec,'__version__','unknown'))

def words(x):return np.ascontiguousarray(x,dtype='<f8').view('<u8')
def zstenc(x):return zstd.ZstdCompressor(level=3).compress(x.tobytes())
def zstdec(x):return zstd.ZstdDecompressor().decompress(x)

def pcodec_try(x):
    # Common public Python APIs vary across versions: test both without assuming success.
    candidates=[('standalone',lambda:pcodec.standalone.compress(x),lambda y:pcodec.standalone.decompress(y)),
                ('compress',lambda:pcodec.compress(x),lambda y:pcodec.decompress(y))]
    for name,enc,dec in candidates:
        try:
            payload=enc();out=dec(payload)
            if np.array_equal(words(np.asarray(out,dtype='<f8')),words(x)):
                return name,enc,dec
        except Exception:pass
    return None,None,None

rows=[]
for name,x in D.items():
    original=words(x)
    methods={'zstd':(lambda:zstenc(x),lambda y:np.frombuffer(zstdec(y),dtype='<f8'))}
    for label,module in [('chimp_independent',chimp),('chimp128_independent',chimp128),('gorilla_independent',gorilla),('patas_independent',patas)]:
        methods[label]=(lambda m=module:m.encode(x.tolist()),lambda payload,m=module:np.asarray(m.decode(payload,len(x)),dtype='<f8'))
    ver,enc,dec=pcodec_try(x)
    if enc is not None:methods['pcodec_'+ver]=(enc,dec)
    else:rows.append(dict(dataset=name,codec='pcodec',status='unavailable_api',reason='Python binding API not recognized or roundtrip failed'))
    for codec,(encoder,decoder) in methods.items():
        try:
            t=time.perf_counter();payload=encoder();enc_time=time.perf_counter()-t
            t=time.perf_counter();restored=decoder(payload);dec_time=time.perf_counter()-t
            if isinstance(restored,(bytes,bytearray)):
                restored=np.frombuffer(restored,dtype='<f8')
            restored=np.asarray(restored,dtype='<f8')
            if len(restored)!=len(x) or not np.array_equal(words(restored),original):
                rows.append(dict(dataset=name,codec=codec,status='bitwise_mismatch',reason='Original IEEE-754 words not preserved'))
                continue
            rows.append(dict(dataset=name,codec=codec,status='verified',input_words=len(x),raw_bytes=x.nbytes,payload_bytes=len(payload),framing_allowance_bytes=12,total_reported_bytes=len(payload)+12,encode_seconds=enc_time,decode_seconds=dec_time))
        except Exception as e:
            rows.append(dict(dataset=name,codec=codec,status='error',reason=str(e)[:400]))
res=pd.DataFrame(rows)
res.to_csv(OUT/'codec_results.csv',index=False)
display(res)


"""
## Decision gate

Only verified codecs appear in the numeric comparison. If specialized implementations are missing, the experiment is **incomplete**, not a compression win. The next step is to construct and validate an official ALP standalone framing wrapper on a small oracle corpus before comparing bytes across all methods.
"""

valid=res[res.status=='verified'].copy()
if len(valid):
    display(valid.pivot(index='dataset',columns='codec',values='total_reported_bytes'))
full=set(D).issubset(set(valid.dataset))
chimp_ok=bool((valid.codec=='chimp_independent').sum()==len(D))
pcodec_ok=bool(valid.codec.str.startswith('pcodec_').sum()==len(D))
status={'notebook':'NB127','all_datasets_have_verified_codec':full,'independent_chimp_bitexact_all':chimp_ok,'pcodec_bitexact_all':pcodec_ok,
        'official_ALP_benchmarked':False,'official_Chimp_benchmarked':False,
        'novelty_established':False,'recommendation':'Do not claim a specialized-codec ranking until official ALP stream codec is validated; use this audit as independent Chimp-family screening only.'}
(OUT/'decision.json').write_text(json.dumps(status,indent=2))
(OUT/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'datasets':URLS,'limit_values':MAX_VALUES},indent=2))
(OUT/'README.md').write_text('NB127: independent Chimp-family screening and official upstream source provenance. NOT an official ALP/Chimp standalone benchmark. Only verified roundtrips are usable.\n')
print(json.dumps(status,indent=2))
with zipfile.ZipFile('/content/SGX_NB127_results.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():z.write(p,p.name)
from google.colab import files
files.download('/content/SGX_NB127_results.zip')

