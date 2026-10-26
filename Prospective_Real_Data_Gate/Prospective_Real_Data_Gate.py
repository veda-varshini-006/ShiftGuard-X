#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# NB15 — Prospective_Real_Data_Gate

Uses genuine UCI numeric datasets where available. Selects coding strategy **only on the chronological calibration segment** and measures its locked choice on held-out data. No hindsight selection leakage; failures produce FAILED status, not fake results. If real downloads fail, the notebook exports a diagnostic ZIP but the research gate is NOT PASSED.

**Run all cells on Google Colab CPU.** Export ZIP downloads automatically. No prior notebook runtime required. Results are research diagnostics, not publication claims.
"""

# %pip -q install zstandard pandas


NOTEBOOK_ID='NB15'

import os,sys,json,csv,time,io,hashlib,zipfile,platform,struct,zlib,lzma,bz2,urllib.request,subprocess,shutil,tempfile
from pathlib import Path
import numpy as np
OUT=Path('sgx_outputs')/NOTEBOOK_ID;OUT.mkdir(parents=True,exist_ok=True)
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



import pandas as pd
urls={'UCI_AirQuality':'https://archive.ics.uci.edu/static/public/360/air+quality.zip',
'UCI_HouseholdPower':'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip'}
source_rows=[];streams={};MAX_VALUES=50000
for source,url in urls.items():
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'ShiftGuardX-NB15/0.5'})
  with urllib.request.urlopen(req,timeout=90) as resp:blob=resp.read()
  z=zipfile.ZipFile(io.BytesIO(blob))
  name=next((i for i in z.namelist() if i.lower().endswith('airqualityuci.csv' if source=='UCI_AirQuality' else 'household_power_consumption.txt')),None)
  if name is None:raise RuntimeError('expected data member absent')
  content=z.read(name)
  if source=='UCI_AirQuality':
   df=pd.read_csv(io.BytesIO(content),sep=';',decimal=',',low_memory=False);columns=['CO(GT)','T','RH']
  else:
   df=pd.read_csv(io.BytesIO(content),sep=';',na_values=['?'],low_memory=False);columns=['Global_active_power','Voltage','Global_intensity']
  successes=0
  for column in columns:
   if column not in df:continue
   vec=pd.to_numeric(df[column],errors='coerce').dropna().to_numpy(dtype='<f8')[:MAX_VALUES]
   if len(vec)<1024:continue
   key=f'{source}_{column}';streams[key]=vec.tobytes();successes+=1
  source_rows.append({'source':source,'status':'REAL_DATA_LOADED' if successes else 'FAILED','url':url,'archive_sha256':sha(blob),'source_sha256':sha(content),'columns_loaded':successes,'detail':'Parsed numeric float64; missing rows omitted; not a lossless CSV compressor'})
 except Exception as exc:source_rows.append({'source':source,'status':'FAILED','url':url,'archive_sha256':'','source_sha256':'','columns_loaded':0,'detail':repr(exc)[:400]})
save_csv('real_data_sources.csv',source_rows)
print('Loaded real numeric series:',len(streams));print(source_rows)



# Calibration-only selection of candidate transform+codec, then locked heldout evaluation.
# Code must be independent of training artifacts from any other notebook.
rng=np.random.default_rng(515)
rows=[];choices=[];methods=['raw','xor','xor_plane','delta','delta_plane'];backs=['zlib']+(['zstd'] if zstd else [])
for name,raw in streams.items():
 n=len(raw)//8;cut=int(n*.7)*8;train,test=raw[:cut],raw[cut:]
 trials=[]
 for method in methods:
  for back in backs:
   r=measure(train,method,back,repeats=2)
   trials.append((r['compressed_bytes'],method,back))
 _,chosen_method,chosen_backend=min(trials)
 choices.append({'dataset':name,'calibration_sha256':sha(train),'heldout_sha256':sha(test),'chosen_method':chosen_method,'chosen_backend':chosen_backend,'calibration_values':len(train)//8,'heldout_values':len(test)//8,'selection':'calibration_only'})
 # Always include an independently encoded fixed baseline, and lock the selected method.
 for policy,m,b in [('locked_calibrated',chosen_method,chosen_backend),('fixed_raw_zlib','raw','zlib')]+([('fixed_raw_zstd','raw','zstd')] if zstd else []):
  r=measure(test,m,b,repeats=3)
  r.update({'dataset':name,'policy':policy,'chosen_on':'calibration' if policy=='locked_calibrated' else 'fixed_in_advance','data_origin':'REAL_UCI_PARSED_FLOAT64','heldout_sha256':sha(test)})
  rows.append(r)
save_csv('locked_choices.csv',choices);save_csv('prospective_heldout_results.csv',rows)
(OUT/'readiness.json').write_text(json.dumps({'real_series':len(streams),'real_data_gate':'PASS' if len(streams)>=4 else 'INSUFFICIENT_REAL_DATA','specialized_baselines':'NOT_RUN_IN_THIS_NOTEBOOK','note':'No oracle selection on heldout. Must compare against specialized numerical codecs before top-journal claims.'},indent=2))
print('Prospective heldout records:',len(rows))
pack_results('NB15',{'dataset_handling':'UCI downloaded or FAILED; missing numerical rows dropped; source file itself not byte-exact recovered','decision':'first real-data prospective result only; specialized baselines still needed'})

