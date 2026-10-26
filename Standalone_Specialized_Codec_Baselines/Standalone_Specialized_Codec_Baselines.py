#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X — NB125: Fair Standalone Numerical-Codec Audit

**Purpose:** Replace NB124's incomparable DuckDB database-vs-stream sizes with **standalone self-contained streams** from real specialist numerical compressors. This notebook benchmarks **Pcodec (Pco)** and **FPZIP** (official published Python bindings), plus zstd and lzma controls. These are real implementations, not home-made approximations of ALP/Chimp. **ALP and Chimp are NOT benchmarked here**; explicitly record that missing comparison. ALP needs separate integration with its official C++ project before a definitive state-of-the-art ranking.

**Primary rule:** Same IEEE-754 `float64` input for all codecs, exact binary64 word checks, total standalone stream bytes including codec headers, no DuckDB database-size comparisons. Input source SHA256, tool versions, timing, failures and excluded rows recorded. This is a baseline study, **not a novel codec**.

**Sources:** [Pcodec Python API](https://github.com/pcodec/pcodec/blob/main/pco_python/README.md), [FPZIP Python API](https://pypi.org/project/fpzip/), [ALP official reference (not tested)](https://github.com/cwida/ALP).
"""

# %pip -q install "numpy>=1.24" pandas pyarrow zstandard pcodec fpzip


import os,io,time,json,zipfile,hashlib,platform,sys,urllib.request,lzma,importlib.metadata
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq,zstandard as zstd
from pcodec import standalone,ChunkConfig
import fpzip
OUT=Path('/content/SGX_NB125_results');OUT.mkdir(exist_ok=True)
SOURCES={
 'yellow_jan_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet',
 'yellow_feb_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet',
 'green_feb_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-02.parquet'}
MAX_ROWS=125000;REPEATS=3
versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','pyarrow','zstandard','pcodec','fpzip']}
print('versions:',versions)


"""
## Source preparation

Read the numeric `total_amount` column from the published Parquet files, with source-file SHA256. Drop missing values based on PyArrow null masks, rather than accidentally changing them into NaNs. Preserve remaining 64-bit floating point words exactly. The experiment benchmarks a **column stream**, not the full original Parquet file. Test each dataset independently; do not mix training or test data.
"""

def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def load_parquet(name,url):
 path=Path('/content')/(name+'.parquet')
 if not path.exists():
  print('Downloading',name,flush=True);urllib.request.urlretrieve(url,path)
 pf=pq.ParquetFile(path); out=[];n=0;nulls=0
 for batch in pf.iter_batches(batch_size=32768,columns=['total_amount']):
  col=batch.column(0); nulls+=col.null_count
  vals=col.to_numpy(zero_copy_only=False)
  # Only nonnull numbers enter this column-level benchmark.
  ok=~col.is_null().to_numpy(zero_copy_only=False)
  out.append(np.ascontiguousarray(vals[ok],dtype='<f8'));n+=len(vals)
  if n>=MAX_ROWS:break
 a=np.ascontiguousarray(np.concatenate(out)[:MAX_ROWS],dtype='<f8')
 return a,dict(source_url=url,parquet_sha256=sha256(path),sampled_rows=min(n,MAX_ROWS),valid_rows=len(a),null_rows=nulls,stream_sha256=hashlib.sha256(a.tobytes()).hexdigest())
DATA={};manifest={}
for name,url in SOURCES.items():DATA[name],manifest[name]=load_parquet(name,url)
print({k:len(v) for k,v in DATA.items()})


"""
## Standalone codecs and exact reconstruction

FPZIP uses `precision=0` for lossless coding. Pcodec uses the standalone self-contained format. For zstd/lzma, record the standard compressed stream. Decoder outputs must have exactly the same `uint64` bits; this detects `+0/-0`, NaN-payload, and other representation changes. Do not silently ignore a failed round-trip.
"""

def codecs():
 return {
  'zstd_3':(lambda a:zstd.ZstdCompressor(level=3).compress(a.tobytes()),lambda data:zstd.ZstdDecompressor().decompress(data)),
  'lzma_3':(lambda a:lzma.compress(a.tobytes(),preset=3),lambda data:lzma.decompress(data)),
  'pcodec_default':(lambda a:standalone.simple_compress(a,ChunkConfig()),lambda data:standalone.simple_decompress(data)),
  'fpzip_lossless':(lambda a:fpzip.compress(a,precision=0,order='C'),lambda data:fpzip.decompress(data,order='C'))}
def reconstruct(name,data,n):
 if name in ('zstd_3','lzma_3'):
  assert len(data)==8*n
  return np.frombuffer(data,dtype='<f8').copy()
 return np.ascontiguousarray(data,dtype='<f8')
def equal_words(a,b):
 return a.shape==b.shape and np.array_equal(a.view('<u8'),b.view('<u8'))
# Sanity gate on finite values, signed zeros, and subnormals.
control=np.array([0,0x8000000000000000,1,0x3ff0000000000000,0x7fefffffffffffff],dtype='<u8').view('<f8')
for name,(enc,dec) in codecs().items():
 try:
  packed=enc(control)
  assert equal_words(control,reconstruct(name,dec(packed),len(control))),f'{name} failed control'
  print(name,'control PASS',len(packed),'bytes')
 except Exception as exc:
  print(name,'control failure:',str(exc)[:240])


"""
## Fair measurements

Run each method three times on the *identical* stream, recording median encode/decode wall-clock time (not a claim of cycle-accurate benchmarking on shared Colab hardware). Compressed size must be identical across repeats. Failures are explicitly recorded and excluded from comparisons. No custom side metadata is required for the standalone codec streams; raw source length is independently known for the zstd/lzma decoder validation. Where decoder needs external length, record this limitation.
"""

rows=[]; failures=[]
for dataset,a in DATA.items():
 orig=a.tobytes();raw=len(orig)
 for name,(enc,dec) in codecs().items():
  try:
   e_times=[];d_times=[];sizes=[]
   for rep in range(REPEATS):
    t=time.perf_counter();blob=enc(a); e_times.append(time.perf_counter()-t)
    t=time.perf_counter();unpacked=reconstruct(name,dec(blob),len(a));d_times.append(time.perf_counter()-t)
    assert equal_words(a,unpacked),f'non-lossless decode on {dataset} / {name}'
    sizes.append(len(blob))
   assert len(set(sizes))==1,'nondeterministic stream size'
   rows.append(dict(dataset=dataset,codec=name,values=len(a),raw_bytes=raw,stream_bytes=sizes[0],bits_per_value=8*sizes[0]/len(a),ratio=raw/sizes[0],encode_seconds_median=float(np.median(e_times)),decode_seconds_median=float(np.median(d_times)),bitwise_pass=True))
  except Exception as exc:
   failures.append(dict(dataset=dataset,codec=name,error=repr(exc)))
res=pd.DataFrame(rows)
res.to_csv(OUT/'standalone_baselines.csv',index=False)
pd.DataFrame(failures,columns=['dataset','codec','error']).to_csv(OUT/'failures.csv',index=False)
display(res[['dataset','codec','stream_bytes','bits_per_value','ratio','bitwise_pass']])
if failures:print('Some codec-dataset tests failed; see failures.csv:',failures)


"""
## Gate / export

This notebook establishes **specialized-codec reference sizes**, not a novel algorithm or a complete SOTA comparison. If two specialist codecs cannot run with verified bitwise reconstruction, mark the gate incomplete. The future ALP official-C++/Chimp comparison remains explicitly outstanding. Any new candidate must beat a stronger, complete suite on independent datasets with identical streams and all overhead.
"""

required={'pcodec_default','fpzip_lossless','zstd_3','lzma_3'}
complete=all(required.issubset(set(res[res.dataset==d].codec)) for d in DATA)
decision={'notebook':'NB125','experiment':'standalone numerical codec baseline','all_required_codecs_verified':bool(complete),'novelty_established':False,'ALP_official_standalone_measured':False,'Chimp_official_standalone_measured':False,'comparable_DuckDB_database_sizes_used':False,'decision':('Specialist stream benchmark complete; ALP/Chimp and cross-domain tests still required' if complete else 'INCOMPLETE: repair codec failures before making compression comparisons')}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
(OUT/'environment.json').write_text(json.dumps(dict(python=sys.version,versions=versions,rows_cap=MAX_ROWS,repeats=REPEATS),indent=2))
(OUT/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
(OUT/'README.md').write_text('NB125 compares standalone official Python bindings of Pcodec and FPZIP against zstd/lzma, on identical float64 column streams. It does NOT measure ALP/Chimp official streams or whole Parquet files.\n')
archive='/content/SGX_NB125_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for f in OUT.iterdir():z.write(f,f.name)
print(json.dumps(decision,indent=2))
from google.colab import files
files.download(archive)

