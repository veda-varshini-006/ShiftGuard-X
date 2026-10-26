#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# NB13 — Specialized_Baseline_Gate

This notebook tests a **transparent reference Gorilla-style XOR code** with exact decoding and checks for independently supplied production-specialized codec executables. Reference Python Gorilla must NOT be represented as official Gorilla benchmark throughput. Unsupported competitors remain NOT_RUN. External executables are opt-in and must round-trip raw little-endian float64 bytes.

**Run all cells on Google Colab CPU.** Export ZIP downloads automatically. No prior notebook runtime required. Results are research diagnostics, not publication claims.
"""

# %pip -q install zstandard pandas


NOTEBOOK_ID='NB13'

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



import numpy as np, math, os, tempfile, subprocess
class BitWriter:
 def __init__(self):self.bits=[]
 def push(self,value,n):self.bits.extend((int(value)>>j)&1 for j in range(n-1,-1,-1))
 def bytes(self):
  n=len(self.bits);arr=np.array(self.bits+[0]*((-n)%8),dtype=np.uint8)
  return np.packbits(arr,bitorder='big').tobytes(),n
class BitReader:
 def __init__(self,b,n):self.a=np.unpackbits(np.frombuffer(b,dtype=np.uint8),bitorder='big')[:n];self.i=0
 def get(self,n):
  if self.i+n>len(self.a):raise ValueError('truncated bitstream')
  x=0
  for b in self.a[self.i:self.i+n]:x=(x<<1)|int(b)
  self.i+=n;return x

def gorilla_ref_encode(data):
 a=words(data);w=BitWriter();prev=0;old_l=0;old_t=0;old_sig=0
 for k,v in enumerate(a):
  v=int(v)
  if k==0:w.push(v,64);prev=v;continue
  x=v^prev;prev=v
  if x==0:w.push(0,1);continue
  w.push(1,1);l=64-x.bit_length();t=(x&-x).bit_length()-1;sig=64-l-t
  if old_sig and l>=old_l and t>=old_t:
   w.push(0,1);w.push(x>>old_t,old_sig)
  else:
   w.push(1,1);w.push(l,6);w.push(sig%64,6);w.push(x>>t,sig)
   old_l,old_t,old_sig=l,t,sig
 payload,bit_count=w.bytes()
 return b'GREF'+__import__('struct').pack('<QQ',len(a),bit_count)+payload

def gorilla_ref_decode(blob):
 if len(blob)<20 or blob[:4]!=b'GREF':raise ValueError('invalid header')
 n,bit_count=__import__('struct').unpack('<QQ',blob[4:20]);r=BitReader(blob[20:],bit_count)
 if n>20_000_000:raise ValueError('length limit')
 out=np.empty(n,dtype='<u8');old_l=0;old_t=0;old_sig=0;prev=0
 for k in range(n):
  if k==0:prev=r.get(64);out[k]=prev;continue
  flag=r.get(1)
  if not flag:out[k]=prev;continue
  reuse=r.get(1)
  if not reuse:
   if old_sig==0:raise ValueError('missing prior window')
   x=r.get(old_sig)<<old_t
  else:
   l=r.get(6);sig=r.get(6) or 64;t=64-l-sig
   if t<0:raise ValueError('invalid span')
   x=r.get(sig)<<t;old_l,old_t,old_sig=l,t,sig
  prev^=x;out[k]=prev
 return out.tobytes()

rng=np.random.default_rng(513)
assert gorilla_ref_decode(gorilla_ref_encode(b''))==b''
cases={'smooth_SYNTHETIC':np.cumsum(rng.normal(0,.005,2048)).astype('<f8').tobytes(),
       'repeats_SYNTHETIC':np.repeat(np.array([1.,2.,3.],dtype='<f8'),500).tobytes(),
       'random_bits_SYNTHETIC':rng.integers(0,256,8*2048,dtype=np.uint8).tobytes(),
       'IEEE_special_bits_SYNTHETIC':np.tile(np.array([0,0x8000000000000000,0x7ff8000000000001,0x7ff0000000000001],dtype='<u8'),500).tobytes()}
rows=[]
for label,data in cases.items():
 t=time.perf_counter();c=gorilla_ref_encode(data);te=time.perf_counter()-t
 t=time.perf_counter();assert gorilla_ref_decode(c)==data;td=time.perf_counter()-t
 rows.append({'dataset':label,'method':'GORILLA_STYLE_PYTHON_REFERENCE_NOT_OFFICIAL','compressed_bytes':len(c),'encode_s':te,'decode_s':td,'input_sha256':sha(data),'verified':True,'comparison_class':'unoptimized_reference'})
 for name,enc,dec in [('zlib3',lambda x:zlib.compress(x,3),zlib.decompress),('lzma3',lambda x:lzma.compress(x,preset=3),lzma.decompress)]:
  t=time.perf_counter();c=enc(data);te=time.perf_counter()-t
  t=time.perf_counter();assert dec(c)==data;td=time.perf_counter()-t
  rows.append({'dataset':label,'method':name,'compressed_bytes':len(c),'encode_s':te,'decode_s':td,'input_sha256':sha(data),'verified':True,'comparison_class':'compiled_generic'})
save_csv('reference_baselines.csv',rows)
print('Verified baseline records:',len(rows))



# Optional genuine specialized compressors; user supplies safe executable paths/args.
# Config JSON: {"ALP":{"encode":["/path/encode","{input}","{output}"],"decode":["/path/decode","{input}","{output}"]}}
# No automatic compilation/download from unfamiliar repositories.
config_path=Path('/content/sgx_specialized_commands.json')
config=json.loads(config_path.read_text()) if config_path.exists() else {}
status=[]
for codec in ['Gorilla','Chimp','ALP','Elf','Dolphin','EAS']:
 result={'codec':codec,'status':'NOT_RUN','reason':'No compatible executable configured','compressed_bytes':'','source_commit':'','exact_ok':False}
 if codec in config:
  try:
   spec=config[codec]
   assert 'encode' in spec and 'decode' in spec and spec.get('source_commit'), 'encode/decode/source_commit required'
   with tempfile.TemporaryDirectory() as td:
    ip=Path(td)/'input.f64';cp=Path(td)/'encoded.bin';op=Path(td)/'output.f64';ip.write_bytes(cases['smooth_SYNTHETIC'])
    for typ,frm,to in [('encode',ip,cp),('decode',cp,op)]:
     cmd=[str(v).replace('{input}',str(frm)).replace('{output}',str(to)) for v in spec[typ]]
     subprocess.run(cmd,check=True,timeout=180,capture_output=True)
    assert op.read_bytes()==ip.read_bytes(), 'exact roundtrip failed'
    result.update(status='VERIFIED_RUN',reason='',compressed_bytes=cp.stat().st_size,source_commit=spec['source_commit'],exact_ok=True)
  except Exception as exc:result.update(status='FAILED',reason=repr(exc)[:300])
 status.append(result)
save_csv('external_specialized_status.csv',status)
(OUT/'limitations.txt').write_text('Gorilla-style Python reference is an independent educational implementation, not official or optimized. Production specialized compressors are not tested unless VERIFIED_RUN. Synthetic input alone cannot establish publication competitiveness.\n')
pack_results('NB13',{'source_type':'SYNTHETIC','specialized_baseline_requirement':'at least 2 production specialized codecs VERIFIED_RUN on real data for stronger submission evidence'})

