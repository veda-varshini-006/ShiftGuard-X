#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# NB14 — Context_Coding_Ablation

Investigates decoder-synchronous **byte-lane context decomposition** with a reversible XOR residual representation. The candidate is a research probe, not claimed novel. Real novelty requires prior-art comparison. Quantify whether context partitioning helps beyond predictor choice and zlib alone.

**Run all cells on Google Colab CPU.** Export ZIP downloads automatically. No prior notebook runtime required. Results are research diagnostics, not publication claims.
"""

# %pip -q install zstandard pandas


NOTEBOOK_ID='NB14'

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



import numpy as np
# One 8-byte word is split into 8 byte lanes. Every stream has identical length; concatenation is reversible.
def lane_transform(data):
 return np.frombuffer(data,dtype=np.uint8).reshape(-1,8).T.copy().reshape(-1).tobytes()
def lane_undo(data):
 return np.frombuffer(data,dtype=np.uint8).reshape(8,-1).T.copy().reshape(-1).tobytes()
def context_encode(data,transform='xor_lane',backend='zlib'):
 if transform not in ('identity','xor','lane','xor_lane'):raise ValueError(transform)
 residual_bytes=residual(data,'xor') if transform.startswith('xor') else data
 transformed=lane_transform(residual_bytes) if transform.endswith('lane') or transform=='lane' else residual_bytes
 payload=compress_payload(transformed,backend)
 hdr=struct.pack('<4sBBQ',b'SG14',('identity','xor','lane','xor_lane').index(transform),BACKENDS[backend],len(data))
 return hdr+payload

def context_decode(encoded):
 if len(encoded)<14:raise ValueError('truncated')
 magic,mid,bid,size=struct.unpack('<4sBBQ',encoded[:14])
 if magic!=b'SG14' or mid>3 or bid not in BIN:raise ValueError('invalid header')
 method=('identity','xor','lane','xor_lane')[mid]
 transformed=decompress_payload(encoded[14:],BIN[bid]);raw=lane_undo(transformed) if 'lane' in method else transformed
 raw=undo(raw,'xor') if method.startswith('xor') else raw
 if len(raw)!=size:raise ValueError('size mismatch')
 return raw
rng=np.random.default_rng(514)
streams={'smooth_SYNTHETIC':np.cumsum(rng.normal(0,.005,4096)).astype('<f8').tobytes(),
 'step_SYNTHETIC':np.repeat(np.array([1.0,1.5,-0.5,0.0],dtype='<f8'),1024).tobytes(),
 'noise_SYNTHETIC':rng.integers(0,256,8*4096,dtype=np.uint8).tobytes(),
 'int_UINT64_SYNTHETIC':np.cumsum(rng.integers(-2,3,4096),dtype=np.int64).astype('<u8').tobytes()}
methods=['identity','xor','lane','xor_lane'];backs=['zlib']+(['zstd'] if zstd else []);rows=[]
for data_id,data in streams.items():
 for back in backs:
  for m in methods:
   start=time.perf_counter();packed=context_encode(data,m,back);elapsed=time.perf_counter()-start
   start=time.perf_counter();restored=context_decode(packed);decelapsed=time.perf_counter()-start
   assert restored==data
   rows.append({'dataset':data_id,'backend':back,'transform':m,'input_bytes':len(data),'packed_bytes':len(packed),'encode_seconds':elapsed,'decode_seconds':decelapsed,'exact_ok':True,'data_origin':'SYNTHETIC','data_sha256':sha(data)})
save_csv('coding_ablation.csv',rows)
print('Verified:',len(rows))



# Pilot decision: compare each candidate to matched untransformed backend.
summary=[]
for dataset in streams:
 for back in backs:
  group=[r for r in rows if r['dataset']==dataset and r['backend']==back]
  base=next(r for r in group if r['transform']=='identity')
  for r in group:
   summary.append({'dataset':dataset,'backend':back,'transform':r['transform'],'raw_bytes':base['packed_bytes'],'candidate_bytes':r['packed_bytes'],'saving_percent':100*(base['packed_bytes']-r['packed_bytes'])/base['packed_bytes']})
save_csv('same_backend_ablation.csv',summary)
(OUT/'novelty_status.json').write_text(json.dumps({'coder':'reversible byte-lane grouped XOR residual + conventional entropy backend','novelty':'NOT_VERIFIED','risks':['bit-plane and byte-lane transforms are known','XOR predictors are known','synthetic datasets only','python overhead materially biases speeds']},indent=2))
pack_results('NB14',{'goal':'identify coding inefficiency; do not claim new entropy-coding theorem','datasets':'synthetic only'})

