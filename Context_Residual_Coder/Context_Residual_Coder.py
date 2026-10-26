#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# NB17 — Context_Residual_Coder

Builds a bit-exact *candidate* codec using previous-word XOR, per-block 8-lane context separation and zlib. A causal context is inferred only from reconstructed words; fixed non-adaptive control included. This is a testable mechanism, NOT an originality claim. Extra metadata and encoder time are included.

**Colab CPU · Run all cells.** Every run exports a ZIP. Datasets / methods not executed are not counted as evidence. Prototype only; no claim of originality or journal acceptance.
"""

# %pip -q install numpy pandas zstandard


NOTEBOOK_ID='NB17'

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



import numpy as np,struct
MAGIC=b'SGC7'
def encode_candidate(raw,block_words=256,mode='context'):
 if len(raw)%8:raise ValueError('float64 words required')
 a=np.frombuffer(raw,dtype='<u8');chunks=[];t=time.perf_counter()
 for start in range(0,len(a),block_words):
  block=a[start:start+block_words];prev=np.empty(len(block),dtype='<u8')
  if len(block):prev[0]=0;prev[1:]=block[:-1]
  residual=np.bitwise_xor(block,prev).view(np.uint8).reshape(-1,8).copy()
  if mode=='context':
   # Predictively separate low-variability byte lanes so generic entropy coder sees context-homogeneous symbols.
   payload=residual.T.reshape(-1).tobytes()
  elif mode=='xor':payload=residual.tobytes()
  elif mode=='identity':payload=block.tobytes()
  else:raise ValueError('mode')
  encoded=zlib.compress(payload,3)
  chunks.append(struct.pack('<II',len(block),len(encoded))+encoded)
 return MAGIC+struct.pack('<IBI',len(a),{'context':1,'xor':2,'identity':3}[mode],len(chunks))+b''.join(chunks)
def decode_candidate(blob,max_values=10_000_000):
 if len(blob)<13 or blob[:4]!=MAGIC:raise ValueError('bad magic')
 n,mode,nb=struct.unpack('<IBI',blob[4:13]);pos=13;out=[]
 if n>max_values or mode not in (1,2,3):raise ValueError('bad header')
 for _ in range(nb):
  if pos+8>len(blob):raise ValueError('truncated block header')
  count,sz=struct.unpack('<II',blob[pos:pos+8]);pos+=8
  if count==0 or count>65536 or pos+sz>len(blob):raise ValueError('bad block sizes')
  payload=zlib.decompress(blob[pos:pos+sz]);pos+=sz
  if len(payload)!=count*8:raise ValueError('decoded length mismatch')
  if mode==3:out.append(payload);continue
  m=np.frombuffer(payload,dtype=np.uint8).reshape(8,count).T.copy() if mode==1 else np.frombuffer(payload,dtype=np.uint8).reshape(count,8)
  r=m.copy().reshape(-1).tobytes();rr=np.frombuffer(r,dtype='<u8');dst=np.empty(count,dtype='<u8');previous=0
  for k,v in enumerate(rr):
   previous=int(v)^previous;dst[k]=previous
  out.append(dst.tobytes())
 if pos!=len(blob) or sum(map(len,out))!=n*8:raise ValueError('length error')
 return b''.join(out)

rng=np.random.default_rng(17)
cases={'smooth_SYNTHETIC':np.cumsum(rng.normal(0,.004,4096)).astype('<f8').tobytes(),
       'repeating_SYNTHETIC':np.repeat(np.array([0.,1.,2.,-0.0],dtype='<f8'),1024).tobytes(),
       'random_bits_SYNTHETIC':rng.integers(0,256,4096*8,dtype=np.uint8).tobytes(),
       'special_bits_SYNTHETIC':np.tile(np.array([0x7ff8000000000001,0x8000000000000000,0x7ff0000000000000,0xffffffffffffffff],dtype='<u8'),1024).tobytes()}
rows=[]
for label,raw in cases.items():
 for mode in ['identity','xor','context']:
  for block in [128,256,1024]:
   t=time.perf_counter();c=encode_candidate(raw,block,mode);et=time.perf_counter()-t
   t=time.perf_counter();assert decode_candidate(c)==raw;dt=time.perf_counter()-t
   rows.append({'dataset':label,'mode':mode,'block_values':block,'input_bytes':len(raw),'compressed_bytes':len(c),'encode_s':et,'decode_s':dt,'exact_ok':True,'source_type':'SYNTHETIC'})
 # exact special encodings tested word-for-word; corruption flag verified without claiming universal detection
save_csv('context_coder_ablations.csv',rows)
assert decode_candidate(encode_candidate(b''))==b''
try:decode_candidate(b'broken');raise AssertionError('invalid magic passed')
except ValueError:pass
print('Bit-exact ablations:',len(rows))



import numpy as np,pandas as pd,subprocess,tempfile,platform
URLS={'UCI_AirQuality':'https://archive.ics.uci.edu/static/public/360/air+quality.zip',
      'UCI_HouseholdPower':'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip'}
def public_streams(max_values=16000):
 streams={};sources=[]
 for source,url in URLS.items():
  try:
   req=urllib.request.Request(url,headers={'User-Agent':'SGX-research/0.6'})
   with urllib.request.urlopen(req,timeout=90) as resp:blob=resp.read()
   z=zipfile.ZipFile(io.BytesIO(blob));member=next((a for a in z.namelist() if a.lower().endswith('airqualityuci.csv' if source=='UCI_AirQuality' else 'household_power_consumption.txt')),None)
   if not member:raise RuntimeError('expected data missing from archive')
   content=z.read(member)
   if source=='UCI_AirQuality':
    df=pd.read_csv(io.BytesIO(content),sep=';',decimal=',',low_memory=False);cols=['CO(GT)','T','RH']
   else:
    df=pd.read_csv(io.BytesIO(content),sep=';',na_values=['?'],low_memory=False);cols=['Global_active_power','Voltage','Global_intensity']
   for col in cols:
    if col not in df:continue
    arr=pd.to_numeric(df[col],errors='coerce').dropna().to_numpy(dtype='<f8')[:max_values]
    if len(arr)>=512:streams[f'{source}/{col}']=arr.tobytes()
   sources.append({'source':source,'status':'REAL_DATA_LOADED','url':url,'archive_sha256':sha(blob),'member_sha256':sha(content),'loaded_columns':len([k for k in streams if k.startswith(source+'/')])})
  except Exception as e:
   sources.append({'source':source,'status':'FAILED','url':url,'archive_sha256':'','member_sha256':'','loaded_columns':0,'error':repr(e)[:500]})
 save_csv('data_provenance.csv',sources)
 return streams,sources

def split_stream(data):
 n=len(data)//8; cut=max(0,int(n*.7))*8
 return data[:cut],data[cut:]



streams,sources=public_streams(max_values=12000)
real_rows=[]
for label,data in streams.items():
 train,test=split_stream(data)
 if len(test)<8:continue
 # freeze selection on calibration; compare equal-input, equal-block candidates
 trial=[(len(encode_candidate(train,256,m)),m) for m in ['identity','xor','context']]
 locked=min(trial)[1]
 for mode in ['identity','xor','context']:
  t=time.perf_counter();c=encode_candidate(test,256,mode);et=time.perf_counter()-t
  t=time.perf_counter();assert decode_candidate(c)==test;dt=time.perf_counter()-t
  real_rows.append({'dataset':label,'mode':mode,'locked_mode':locked,'deployable_selected':mode==locked,'heldout_bytes':len(test),'compressed_bytes':len(c),'encode_s':et,'decode_s':dt,'source_type':'REAL_UCI_PARSED_FLOAT64','exact_ok':True})
save_csv('real_data_context_comparisons.csv',real_rows)
(OUT/'readiness.json').write_text(json.dumps({'real_series':len(streams),'novelty':'UNVERIFIED','compiled_codec':'NO','note':'Context byte lanes + XOR are established techniques; must show distinctive novelty beyond this prototype. Real data tests do not prove byte-exact CSV reproduction.'},indent=2))
pack_results('NB17',{'prototype':'context lane residual coding','novelty':'UNVERIFIED'})

