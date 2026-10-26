#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB10_Residual_Bitplane_Coding: Residual-bitplane codec experiment

**Exploratory experiment — NOT a validated novel compression algorithm.**

Run all cells. A results ZIP is exported at the end. All measurements include a 14-byte stream header. Numerical inputs are interpreted as **raw IEEE-754 64-bit word patterns**; no floating-point calculations change their bits.

This notebook is standalone, CPU-only and makes no claims about journal acceptance.

Examines a **candidate** bit-plane rearrangement after XOR or modular delta prediction with matched entropy backends. These operations have extensive prior art; this notebook tests performance, not originality. Synthetic probes are labeled synthetic.
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


rng=np.random.default_rng(17)
n=8192
cases={
 'smooth_f64_SYNTHETIC':np.cumsum(rng.normal(0,.01,n)).astype('<f8').tobytes(),
 'integers_as_u64_SYNTHETIC':np.cumsum(rng.integers(-3,4,n),dtype=np.int64).astype('<u8').tobytes(),
 'alternating_f64_SYNTHETIC':np.tile(np.array([1.,-1.,0.,0.],dtype='<f8'),n//4).tobytes(),
 'random_bits_SYNTHETIC':rng.integers(0,256,n*8,dtype=np.uint8).tobytes(),
 'special_bits_SYNTHETIC':np.tile(np.array([0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0x7ff0000000000001],dtype='<u8'),n//6+1)[:n].tobytes()
}
# Exhaustive exact reconstruction for all candidate predictor/backend pairs
rows=[]
for dataset,data in cases.items():
 for meth in METHOD_LIST:
  for back in BACKEND_LIST:
   rec=measure(data,meth,back,repeats=2);rec['dataset']=dataset;rec['data_sha256']=sha(data);rec['data_origin']='SYNTHETIC';rows.append(rec)
save_csv('candidate_codecs.csv',rows)
assert all(r['exact_ok'] for r in rows)
print('Exact tests:',len(rows),'Datasets:',len(cases))
for k in cases:
 ranked=sorted([x for x in rows if x['dataset']==k],key=lambda r:r['compressed_bytes'])
 print(k,[(x['method'],x['backend'],x['compressed_bytes']) for x in ranked[:3]])


# Non-overclaiming ablative comparison of identical entropy backend
ab=[]
for d in cases:
 for b in BACKEND_LIST:
  block=[r for r in rows if r['dataset']==d and r['backend']==b]
  base=next(v for v in block if v['method']=='raw')
  for v in block:
   ab.append({'dataset':d,'backend':b,'method':v['method'],'raw_backend_bytes':base['compressed_bytes'],'candidate_bytes':v['compressed_bytes'],'change_pct':100*(v['compressed_bytes']/base['compressed_bytes']-1)})
save_csv('ablation_same_backend.csv',ab)
# corruption cases must fail; corrupted raw payload can evade detection: not claimed tamper proof
for method in ['xor','delta_plane','raw']:
 a=encode(cases['special_bits_SYNTHETIC'],method,'zlib');assert decode(a)==cases['special_bits_SYNTHETIC']
 for bad in (a[:3],a[:12],b'XXXX'+a[4:]):
  try:decode(bad);raise AssertionError('invalid stream accepted')
  except (ValueError,struct.error,zlib.error):pass
print('Header truncation and invalid-magic tests passed. Payload checksum not implemented.')


pack_results('NB10',{'data':'five labeled synthetic streams','scope':'candidate residual bit-plane representation; no novel claim','benchmark_fairness':'same block and backend; 14-byte header; timing includes transform'})

