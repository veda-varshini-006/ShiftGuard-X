#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB126 — FPZIP Exactness Diagnosis and Specialized Baseline Repair

This notebook follows NB125. It runs **official Python FPZIP binding in precision=0 (lossless)** and evaluates actual bitwise reconstructed words. The NB125 failure must not be assumed to be FPZIP's inherent inability to compress losslessly. Full 64-bit word equality, mismatch patterns, and escape/correction overhead are recorded.

**Scope:** standalone FPZIP alongside Pcodec and zstd on identical NYC TLC column streams. The official standalone ALP/Chimp implementations are **not included** (still outstanding). Run all cells in Google Colab; exported `SGX_NB126_results.zip` contains complete results, including failures, versions and input hashes. A claimed compression result is accepted only when exact bits match. No novelty claim.
"""

# %pip -q install "numpy>=1.24" pandas pyarrow zstandard pcodec fpzip


import io,os,sys,json,zipfile,time,hashlib,platform,urllib.request,importlib.metadata
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq,zstandard as zstd,fpzip
from pcodec import standalone,ChunkConfig
OUT=Path('/content/SGX_NB126_results');OUT.mkdir(exist_ok=True)
SOURCES={'yellow_jan_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet','yellow_feb_2024':'https://d37ci6vzurych.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet','green_feb_2024':'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-02.parquet'}
SOURCES['yellow_feb_2024']='https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet'
NROWS=125000
print({k:importlib.metadata.version(k) for k in ('numpy','pandas','pyarrow','pcodec','fpzip','zstandard')})


def sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for part in iter(lambda:f.read(2**20),b''):h.update(part)
 return h.hexdigest()
def dataset(name,url):
 p=Path('/content')/(name+'.parquet')
 if not p.exists():
  print('Downloading',name,flush=True);urllib.request.urlretrieve(url,p)
 pf=pq.ParquetFile(p); out=[];count=0;missing=0
 for batch in pf.iter_batches(columns=['total_amount'],batch_size=32768):
  col=batch.column(0);ok=~col.is_null().to_numpy(zero_copy_only=False)
  a=np.ascontiguousarray(col.to_numpy(zero_copy_only=False)[ok],dtype='<f8')
  out.append(a);count+=len(ok);missing+=int((~ok).sum())
  if count>=NROWS:break
 a=np.ascontiguousarray(np.concatenate(out)[:NROWS],dtype='<f8')
 return a,dict(parquet_sha256=sha_file(p),stream_sha256=hashlib.sha256(a.tobytes()).hexdigest(),source=url,non_null_rows=len(a),nulls_in_read_batches=missing)
DATA={};manifest={}
for name,url in SOURCES.items():DATA[name],manifest[name]=dataset(name,url)
print({k:len(a) for k,a in DATA.items()})


"""
## Controls: exact IEEE-754 words

FPZIP supports multidimensional floating arrays and documents `precision=0` as lossless. Test both 1D and 2D shapes and count **bitwise** differences. If numerical equality holds but raw words differ, signed-zero or NaN details might be responsible; no such changes are accepted as bit-exact. All runs are assessed independently.
"""

def words(x):return np.ascontiguousarray(x,dtype='<f8').view('<u8')
def diagnose(a,b):
 a=words(a);b=words(b)
 if a.shape!=b.shape:return dict(shape_match=False,bitwise_equal=False,mismatch_count=None)
 diff=a!=b
 ix=np.flatnonzero(diff)[:10]
 return dict(shape_match=True,bitwise_equal=bool(not diff.any()),mismatch_count=int(diff.sum()),first_mismatch_indices=ix.tolist(),original_words_hex=[hex(int(v)) for v in a[ix]],decoded_words_hex=[hex(int(v)) for v in b[ix]])
controls={
 'regular':np.array([0.25,1.,-1.,9.5,37.,1000.,-2.75],dtype='<f8'),
 'adversarial':np.array([0,0x8000000000000000,1,0x3ff0000000000000,0x7fefffffffffffff,0x7ff8000000000001],dtype='<u8').view('<f8'),
 'taxi_first1024':DATA['yellow_jan_2024'][:1024]}
control_results=[]
for name,a in controls.items():
 for shape in ('1D','2D'):
  inp=a if shape=='1D' else a.reshape(-1,1)
  try:
   packed=fpzip.compress(inp,precision=0,order='C')
   restored=np.asarray(fpzip.decompress(packed,order='C'),dtype='<f8').reshape(-1)
   result=diagnose(a,restored);result.update(control=name,shape=shape,payload_bytes=len(packed),error=None)
  except Exception as e:result=dict(control=name,shape=shape,error=repr(e),bitwise_equal=False)
  control_results.append(result)
control_df=pd.DataFrame(control_results);display(control_df)
control_df.to_json(OUT/'control_diagnostics.json',orient='records',indent=2)


"""
## Full data: record mismatches, correct sizes, timing and failures

We compute a complete independent stream for each codec. An FPZIP stream with any mismatch is **not** a valid bit-exact result. An optional diagnostic counts a theoretical worst-case repair stream but does **not** report that as a validated complete codec.
"""

rows=[];mismatches=[]
for name,a in DATA.items():
 for codec in ('zstd3','pcodec','fpzip_1D','fpzip_2D'):
  try:
   t=time.perf_counter()
   if codec=='zstd3':
    packed=zstd.ZstdCompressor(level=3).compress(a.tobytes());encode_t=time.perf_counter()-t
    t=time.perf_counter();rec=np.frombuffer(zstd.ZstdDecompressor().decompress(packed),dtype='<f8').copy();decode_t=time.perf_counter()-t
   elif codec=='pcodec':
    packed=standalone.simple_compress(a,ChunkConfig());encode_t=time.perf_counter()-t
    t=time.perf_counter();rec=np.asarray(standalone.simple_decompress(packed),dtype='<f8').reshape(-1);decode_t=time.perf_counter()-t
   else:
    inp=a if codec=='fpzip_1D' else a.reshape(-1,1)
    packed=fpzip.compress(inp,precision=0,order='C');encode_t=time.perf_counter()-t
    t=time.perf_counter();rec=np.asarray(fpzip.decompress(packed,order='C'),dtype='<f8').reshape(-1);decode_t=time.perf_counter()-t
   d=diagnose(a,rec)
   rows.append(dict(dataset=name,codec=codec,rows=len(a),compressed_bytes=len(packed),encode_seconds=encode_t,decode_seconds=decode_t,bitwise_equal=d['bitwise_equal'],mismatch_count=d['mismatch_count'],error=''))
   if not d['bitwise_equal']:mismatches.append(dict(dataset=name,codec=codec,**d))
  except Exception as e:rows.append(dict(dataset=name,codec=codec,rows=len(a),compressed_bytes=None,encode_seconds=None,decode_seconds=None,bitwise_equal=False,mismatch_count=None,error=repr(e)))
res=pd.DataFrame(rows);display(res)
res.to_csv(OUT/'benchmark.csv',index=False)
(OUT/'bitwise_failures.json').write_text(json.dumps(mismatches,indent=2))


passed_fpzip=bool(res[res.codec=='fpzip_1D'].bitwise_equal.all() or res[res.codec=='fpzip_2D'].bitwise_equal.all())
complete=bool(res[res.codec.isin(['pcodec','zstd3'])].bitwise_equal.all() and passed_fpzip)
decision=dict(notebook='NB126',fpzip_bitwise_passes_all_datasets_in_at_least_one_shape=passed_fpzip,complete_specialist_subset=complete,novelty_established=False,ALP_standalone_measured=False,Chimp_standalone_measured=False,recommendation=('Continue fair specialized baseline suite, including official ALP and Chimp standalone codecs' if complete else 'Investigate FPZIP library limitations and use verified Pcodec baseline only until repaired'))
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
(OUT/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
(OUT/'environment.json').write_text(json.dumps({'python':sys.version,'versions':{k:importlib.metadata.version(k) for k in ('numpy','pandas','pyarrow','pcodec','fpzip','zstandard')}},indent=2))
(OUT/'README.md').write_text('NB126: FPZIP exactness diagnostics. FPZIP streams failing 64-bit reconstruction must not be considered lossless. ALP and Chimp standalone benchmarking remains outstanding.\n')
archive='/content/SGX_NB126_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for f in OUT.iterdir():z.write(f,arcname=f.name)
print(json.dumps(decision,indent=2))
from google.colab import files
files.download(archive)

