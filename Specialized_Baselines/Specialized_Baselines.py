#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB11_Specialized_Baselines: Baseline audit and integration

**Exploratory experiment — NOT a validated novel compression algorithm.**

Run all cells. A results ZIP is exported at the end. All measurements include a 14-byte stream header. Numerical inputs are interpreted as **raw IEEE-754 64-bit word patterns**; no floating-point calculations change their bits.

This notebook is standalone, CPU-only and makes no claims about journal acceptance.

Records exact baseline status. Publicly available specialised codecs **must not** be represented as run unless a working binary/library is configured. Provide a command template through `SGX_SPECIALIZED_COMMANDS_JSON` if you have compatible binaries. Template uses `{input}` and `{output}` substitutions and must support exact binary64 byte arrays.
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


# Consistent input for available codecs, no claim that zlib/lzma/bz2 are specialized floating-point codecs
rng=np.random.default_rng(29); n=32768
DATA=np.cumsum(rng.normal(size=n)).astype('<f8').tobytes()
records=[]
for meth in ['raw','xor','xor_plane','delta','delta_plane']:
 for b in BACKEND_LIST:
  v=measure(DATA,meth,b,repeats=3);v.update({'dataset':'random_walk_f64_SYNTHETIC','baseline_class':'candidate_or_generic','data_sha256':sha(DATA)});records.append(v)
for name,enc,dec in [
 ('lzma',lambda b:lzma.compress(b,preset=3),lzma.decompress),
 ('bz2',lambda b:bz2.compress(b,compresslevel=3),bz2.decompress),
 ('zlib',lambda b:zlib.compress(b,level=3),zlib.decompress)]:
 et=[];dt=[]
 for _ in range(3):
  t=time.perf_counter();c=enc(DATA);et.append(time.perf_counter()-t)
  t=time.perf_counter();raw=dec(c);dt.append(time.perf_counter()-t);assert raw==DATA
 records.append({'dataset':'random_walk_f64_SYNTHETIC','method':'none','backend':name,'original_bytes':len(DATA),'compressed_bytes':len(c),'bits_per_value':8*len(c)/n,'encode_seconds_med':float(np.median(et)),'decode_seconds_med':float(np.median(dt)),'encode_mib_s':len(DATA)/1048576/np.median(et),'decode_mib_s':len(DATA)/1048576/np.median(dt),'exact_ok':True,'baseline_class':'generic_production','data_sha256':sha(DATA)})
save_csv('measured_generic_baselines.csv',records)
print('Measured configurations:',len(records))


# SPECIALIZED EXTERNAL BINARIES. Set json with e.g.
# {"Gorilla":{"encode":["/content/gorilla_enc","{input}","{output}"],"decode":["/content/gorilla_dec","{input}","{output}"]}}
# Commands must use a matching format that really handles input as raw little-endian binary64.
# Caution: outputs that require a separate shared library or metadata must be self-contained and counted.
SPECIAL=['Gorilla','Chimp','ALP','Elf','Dolphin','EAS']
config_path=os.environ.get('SGX_SPECIALIZED_COMMANDS_JSON','/content/sgx_specialized_commands.json')
config={}
if Path(config_path).exists():config=json.loads(Path(config_path).read_text())
statuses=[]
for name in SPECIAL:
 status={'codec':name,'status':'NOT_RUN','reason':'No compatible encode/decode commands configured','compressed_bytes':'','verified_roundtrip':False}
 if name in config:
  try:
   spec=config[name];assert 'encode' in spec and 'decode' in spec
   with tempfile.TemporaryDirectory() as td:
    inp=Path(td)/'in.f64';comp=Path(td)/'compressed';decoded=Path(td)/'out.f64';inp.write_bytes(DATA)
    for mode,source,dest in [('encode',inp,comp),('decode',comp,decoded)]:
     cmd=[str(x).replace('{input}',str(source)).replace('{output}',str(dest)) for x in spec[mode]]
     subprocess.run(cmd,check=True,timeout=120,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert decoded.read_bytes()==DATA,'exact bytes mismatch'
    status.update(status='VERIFIED_RUN',reason='',compressed_bytes=comp.stat().st_size,verified_roundtrip=True)
  except Exception as ex:status.update(status='FAILED',reason=str(ex)[:400])
 statuses.append(status)
save_csv('specialized_baseline_status.csv',statuses)
print('Specialized baselines:',statuses)
assert all(x['status'] in ['NOT_RUN','FAILED','VERIFIED_RUN'] for x in statuses)


(OUT/'baseline_sources_and_limitations.json').write_text(json.dumps({'Gorilla':'https://www.vldb.org/pvldb/vol8/p1816-teller.pdf','Chimp':'https://github.com/panagiotisl/chimp','ALP':'https://github.com/cwida/ALP','status_policy':'NOT_RUN when absent; FAILED when invocation or byte-exact reconstruction fails; VERIFIED_RUN only when successful','fairness_limitations':'External compressor CLI overhead may differ. Source-format assumptions must be documented. This notebook is an availability audit, not a fully matched codec benchmark.'},indent=2))
pack_results('NB11',{'data':'single synthetic float64 pilot','critical_requirement':'real specialty codecs are NOT_RUN unless supplied and verified','results_scope':'generic baseline audit; no paper-ready specialized comparison'})

