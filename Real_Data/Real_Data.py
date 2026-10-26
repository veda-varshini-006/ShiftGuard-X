#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB12_Real_Data: Public numerical dataset benchmark

**Exploratory experiment — NOT a validated novel compression algorithm.**

Run all cells. A results ZIP is exported at the end. All measurements include a 14-byte stream header. Numerical inputs are interpreted as **raw IEEE-754 64-bit word patterns**; no floating-point calculations change their bits.

This notebook is standalone, CPU-only and makes no claims about journal acceptance.

Downloads **real** UCI Air Quality and household power consumption datasets from primary repository, with provenance and error logs. No synthetic fallback is mislabeled as real. Dataset preparation serializes the *exact IEEE-754 binary64* bits produced by the documented CSV-to-float conversion; this is lossless relative to that converted binary64 stream, **not** lossless reconstruction of the original CSV text.
"""

# %pip -q install zstandard



import os,sys,json,csv,time,io,hashlib,zipfile,platform,struct,zlib,lzma,bz2,urllib.request,subprocess,shutil,tempfile
from pathlib import Path
import numpy as np
OUT=Path('sgx_outputs');OUT.mkdir(exist_ok=True)
try:
 import zstandard as zstd
except ImportError:
 zstd=None
MAGIC=b'SGX3'; METHODS={'raw':0,'xor':1,'xor_plane':2,'delta':3,'delta_plane':4}; INV={v:k for k,v in METHODS.items()}
BACKENDS={'zlib':0,'raw':1,'zstd':2}; BIN={v:k for k,v in BACKENDS.items()}
def sha(b): return hashlib.sha256(b).hexdigest()
def words(data):
 assert len(data)%8==0
 return np.frombuffer(data,dtype='<u8').copy()
def residual(data,method):
 a=words(data)
 if method=='raw':return data
 if method in ('xor','xor_plane'):
  prev=np.empty_like(a);prev[0]=0;prev[1:]=a[:-1]
  r=np.bitwise_xor(a,prev)
 elif method in ('delta','delta_plane'):
  prev=np.empty_like(a);prev[0]=0;prev[1:]=a[:-1]
  r=np.subtract(a,prev,dtype=np.uint64)
 else:raise ValueError(method)
 b=r.astype('<u8',copy=False).tobytes()
 if method.endswith('plane'):
  v=np.frombuffer(b,dtype=np.uint8).reshape(-1,8)
  # Pack bitplanes across bytes; exact bijection including padding bits
  return np.packbits(np.unpackbits(v.reshape(-1),bitorder='little').reshape(-1,8).T,bitorder='little').tobytes()
 return b
def undo(data,method):
 if method=='raw':return data
 if method.endswith('plane'):
  # Bitplanes are packed in each of 8 planes, length bytes divisible by 8.
  arr=np.frombuffer(data,dtype=np.uint8)
  bits=np.unpackbits(arr,bitorder='little').reshape(8,-1).T.reshape(-1)
  data=np.packbits(bits,bitorder='little').tobytes()
 r=words(data)
 if method.startswith('xor'):
  # Sequential recurrence; use cumulative XOR ufunc for native execution
  a=np.bitwise_xor.accumulate(r)
 else:
  a=np.cumsum(r,dtype=np.uint64)
 return a.astype('<u8',copy=False).tobytes()
def compress_payload(b,backend):
 if backend=='raw': return b
 if backend=='zlib':return zlib.compress(b,level=3)
 if backend=='zstd':
  if zstd is None:raise RuntimeError('zstandard not installed')
  return zstd.ZstdCompressor(level=3).compress(b)
 raise ValueError(backend)
def decompress_payload(b,backend):
 if backend=='raw':return b
 if backend=='zlib':return zlib.decompress(b)
 if backend=='zstd':return zstd.ZstdDecompressor().decompress(b)
 raise ValueError(backend)
def encode(data,method,backend):
 assert isinstance(data,bytes) and len(data)%8==0
 payload=compress_payload(residual(data,method),backend)
 header=struct.pack('<4sBBQ',MAGIC,METHODS[method],BACKENDS[backend],len(data))
 return header+payload
def decode(blob):
 if len(blob)<14:raise ValueError('truncated header')
 magic,mi,bi,n=struct.unpack('<4sBBQ',blob[:14])
 if magic!=MAGIC or mi not in INV or bi not in BIN:raise ValueError('invalid header')
 transformed=decompress_payload(blob[14:],BIN[bi]);original=undo(transformed,INV[mi]);
 if len(original)!=n:raise ValueError('length mismatch')
 return original
METHOD_LIST=list(METHODS)
BACKEND_LIST=['zlib','raw']+(['zstd'] if zstd else [])
def measure(data,method,backend,repeats=3):
 times=[];dt=[];n=0
 for _ in range(repeats):
  t=time.perf_counter();b=encode(data,method,backend);times.append(time.perf_counter()-t)
  t=time.perf_counter();x=decode(b);dt.append(time.perf_counter()-t)
  assert x==data,(method,backend)
  n=len(b)
 return {'method':method,'backend':backend,'original_bytes':len(data),'compressed_bytes':n,'bits_per_value':8*n/(len(data)//8),'encode_seconds_med':float(np.median(times)),'decode_seconds_med':float(np.median(dt)),'encode_mib_s':len(data)/1048576/max(float(np.median(times)),1e-12),'decode_mib_s':len(data)/1048576/max(float(np.median(dt)),1e-12),'exact_ok':True}
def save_csv(path,rows):
 if not rows:return
 with open(OUT/path,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pack_results(name,meta):
 (OUT/'manifest.json').write_text(json.dumps({'notebook':name,'timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'python':sys.version,'numpy':np.__version__,'host':platform.platform(),'zstd_available':zstd is not None,'source_files':{p.name:sha(p.read_bytes()) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'},'notes':meta},indent=2,default=str))
 zp=Path(name+'_ShiftGuardX_results.zip')
 with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(OUT.iterdir()):
   if p.is_file():z.write(p,p.name)
 print('EXPORTED:',str(zp.resolve()),'bytes:',zp.stat().st_size)
 try:
  from google.colab import files
  files.download(str(zp))
 except ImportError:pass
 return zp


# Get real data. Network failure is a hard gate: we write a status record, and never fabricate real results.
import pandas as pd
urls={
 'UCI_AirQuality':'https://archive.ics.uci.edu/static/public/360/air+quality.zip',
 'UCI_HouseholdPower':'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip'
}
DATASET_LIMIT=100000 # sample 100k chronological numeric values per dataset/variable to fit free Colab
status=[];source_data={};
for label,url in urls.items():
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'ShiftGuardX-research/0.4'})
  with urllib.request.urlopen(req,timeout=45) as r:archive_bytes=r.read()
  z=zipfile.ZipFile(io.BytesIO(archive_bytes))
  names=z.namelist()
  fname=next((x for x in names if x.lower().endswith('airqualityuci.csv')),None) if label=='UCI_AirQuality' else next((x for x in names if x.lower().endswith('household_power_consumption.txt')),None)
  if fname is None:raise RuntimeError('expected source file not found: '+str(names[:10]))
  raw=z.read(fname)
  if label=='UCI_AirQuality':
   df=pd.read_csv(io.BytesIO(raw),sep=';',decimal=',',low_memory=False)
   columns=['CO(GT)','T','RH']
  else:
   df=pd.read_csv(io.BytesIO(raw),sep=';',na_values=['?'],low_memory=False)
   columns=['Global_active_power','Voltage','Global_intensity']
  count=0
  for column in columns:
   if column not in df.columns:continue
   vec=pd.to_numeric(df[column],errors='coerce').dropna().to_numpy(dtype='<f8')[:DATASET_LIMIT]
   if len(vec)<1024:continue
   # No sorting/shuffling; preserve time order. Missing values are dropped explicitly and documented.
   key=label+'__'+column.replace('/','_')
   source_data[key]=vec.tobytes();count+=1
  status.append({'dataset':label,'status':'REAL_DATA_LOADED' if count else 'FAILED','archive_sha256':sha(archive_bytes),'csv_sha256':sha(raw),'source_url':url,'archive_member':fname,'n_numeric_series':count,'note':'CSV numeric parse to little-endian float64; missing values omitted chronologically'})
 except Exception as ex:
  status.append({'dataset':label,'status':'FAILED','archive_sha256':'','csv_sha256':'','source_url':url,'archive_member':'','n_numeric_series':0,'note':str(ex)[:500]})
save_csv('data_sources.csv',status)
print('Data status:',status)
print('Usable real series:',list(source_data))


# Real-data, matched backend comparisons. No made-up rows on download failures.
results=[];splits=[]
for name,data in source_data.items():
 n=len(data)//8; k=int(n*.7)
 # 70% calibration segment, 30% held out; no fitting in present static codecs.
 # Both subsets are independently encoded, keeping all metadata in each.
 for split,segment in [('calibration',data[:k*8]),('heldout',data[k*8:])]:
  splits.append({'dataset':name,'split':split,'values':len(segment)//8,'sha256':sha(segment)})
  for method in METHOD_LIST:
   for backend in BACKEND_LIST:
    rec=measure(segment,method,backend,repeats=3)
    rec.update({'dataset':name,'split':split,'source_type':'REAL_UCI_CONVERTED_FLOAT64','sha256':sha(segment)})
    results.append(rec)
save_csv('real_data_benchmarks.csv',results)
save_csv('frozen_splits.csv',splits)
print('Real-data experiment rows:',len(results),'heldout datasets:',len([r for r in results if r['split']=='heldout']))
if not results:print('NO REAL RESULTS — dataset downloads failed. Re-run in Colab with network or upload original UCI files; do not claim journal benchmark completed.')


# Transparent summary: top candidate by size vs raw+same backend. Not a novel state-of-the-art comparison.
summary=[]
for name in source_data:
 for backend in BACKEND_LIST:
  vals=[r for r in results if r['dataset']==name and r['split']=='heldout' and r['backend']==backend]
  if not vals:continue
  base=next(r for r in vals if r['method']=='raw')
  best=min(vals,key=lambda r:r['compressed_bytes'])
  summary.append({'dataset':name,'backend':backend,'raw_compressed_bytes':base['compressed_bytes'],'best_candidate':best['method'],'best_candidate_bytes':best['compressed_bytes'],'best_vs_raw_pct':100*(best['compressed_bytes']/base['compressed_bytes']-1),'note':'best candidate chosen hindsight on heldout; optimistic oracle, NOT a deployable selector'})
save_csv('heldout_hindsight_analysis.csv',summary)
pack_results('NB12',{'sources':'Primary UCI archives, SHA-256 recorded','real_series':len(source_data),'evaluation_warning':'Best-on-heldout is hindsight oracle, not prospective selection; converted binary64 is the compressed source; missing values dropped','public_dataset_downloads_failed':sum(s['status']=='FAILED' for s in status)})

