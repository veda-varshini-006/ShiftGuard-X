#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# NB16 — Verified_Specialized_Baselines

Runs actual generic baselines and optional *external executable* ALP, Chimp or Gorilla codecs, only if reproducible encoder+decoder commands are provided. Clearly marks missing production implementations as NOT_RUN. Data are genuine UCI float64 streams only when downloads succeed.

**Colab CPU · Run all cells.** Every run exports a ZIP. Datasets / methods not executed are not counted as evidence. Prototype only; no claim of originality or journal acceptance.
"""

# %pip -q install numpy pandas zstandard


NOTEBOOK_ID='NB16'

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



streams,sources=public_streams()
if not streams: print('REAL DATA UNAVAILABLE: no real-data baseline results; check data_provenance.csv')
# A user-supplied JSON at /content/sgx_specialized_commands.json may configure genuine binaries:
# {"ALP":{"encode":["/content/ALP/encode","{input}","{output}"],"decode":["/content/ALP/decode","{input}","{output}"],"source_url":"https://github.com/cwida/ALP","source_commit":"full git SHA","input_format":"raw_le_f64"}}
# Never label a reimplementation 'official'. Verify exact bytes after decompression.
config_file=Path('/content/sgx_specialized_commands.json')
config=json.loads(config_file.read_text()) if config_file.exists() else {}
rows=[];status=[]
for codec in ['ALP','Chimp','Gorilla','Elf','Dolphin','EAS']:
 conf=config.get(codec)
 st={'codec':codec,'status':'NOT_RUN','verified_streams':0,'reason':'No compatible executable configured','source_url':'','source_commit':''}
 if conf:
  st.update(source_url=conf.get('source_url',''),source_commit=conf.get('source_commit',''))
  if conf.get('input_format')!='raw_le_f64' or not conf.get('source_commit') or not conf.get('source_url'):
   st.update(status='INVALID_CONFIG',reason='Require raw_le_f64 input_format and source_url/source_commit')
  else:
   errors=[]
   for name,data in streams.items():
    train,test=split_stream(data)
    if not test:continue
    try:
     with tempfile.TemporaryDirectory() as folder:
      ip=Path(folder)/'in.f64';cp=Path(folder)/'out.bin';op=Path(folder)/'decoded.f64';ip.write_bytes(test)
      def call(key,fi,fo):
       cmd=[str(x).replace('{input}',str(fi)).replace('{output}',str(fo)) for x in conf[key]]
       t=time.perf_counter();subprocess.run(cmd,check=True,capture_output=True,timeout=180);return time.perf_counter()-t
      et=call('encode',ip,cp);dt=call('decode',cp,op)
      assert op.read_bytes()==test,'bitwise reconstruction failure'
      rows.append({'dataset':name,'codec':codec,'input_bytes':len(test),'compressed_bytes':cp.stat().st_size,'encode_seconds':et,'decode_seconds':dt,'exact_ok':True,'source_commit':conf['source_commit'],'source_type':'REAL_UCI_PARSED_FLOAT64'})
      st['verified_streams']+=1
    except Exception as e:errors.append(f'{name}: {type(e).__name__} {str(e)[:100]}')
   st.update(status='VERIFIED_RUN' if st['verified_streams'] else 'FAILED',reason='; '.join(errors)[:500])
 status.append(st)
for name,data in streams.items():
 _,test=split_stream(data)
 if not test:continue
 for method in ['raw']:
  for back in ['zlib']+(['zstd'] if zstd else []):
   rec=measure(test,method,back,repeats=5)
   rows.append({'dataset':name,'codec':f'{method}+{back}','input_bytes':len(test),'compressed_bytes':rec['compressed_bytes'],'encode_seconds':rec.get('encode_s',rec.get('encode_seconds','')),'decode_seconds':rec.get('decode_s',rec.get('decode_seconds','')),'exact_ok':rec.get('exact_ok',True),'source_commit':'python package environment','source_type':'REAL_UCI_PARSED_FLOAT64'})
save_csv('baseline_measurements.csv',rows);save_csv('specialized_baseline_status.csv',status)
gate=sum(s['status']=='VERIFIED_RUN' for s in status)
(OUT/'readiness.json').write_text(json.dumps({'real_series':len(streams),'production_specialized_codecs_verified':gate,'baseline_gate':'PASS' if gate>=2 and len(streams)>=4 else 'INCOMPLETE','note':'Requires actual verified specialized implementations, not Python reference. Timings include subprocess startup for configured executables.'},indent=2))
pack_results('NB16',{'baseline_gate':'PASS' if gate>=2 and len(streams)>=4 else 'INCOMPLETE','note':'Missing specialized baselines block state-of-art claims'})
print('Real series:',len(streams),'genuine external specialized codecs verified:',gate)

