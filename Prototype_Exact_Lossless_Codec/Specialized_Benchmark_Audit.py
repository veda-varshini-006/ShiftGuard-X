#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# NB09 — Specialized lossless benchmark and evidence audit

Provides FAIR measured general-purpose reference codecs and clearly documents missing specialized Gorilla/Chimp/ALP/Elf*/Dolphin comparisons. Does not fabricate comparisons.

**Research constraints:** exploratory prototype only; no claim of a novel or state-of-the-art codec. All synthetic datasets are labeled. Put independently obtained IEEE-754 little-endian binary64 `.bin` files in `sgx_public_data/` to add real datasets. A 64-bit payload is preserved bitwise, including NaNs and signed zero. Each notebook is independent and exports its own ZIP. Run all cells in order.
"""


import os, sys, math, time, json, csv, hashlib, random, struct, zlib, lzma, bz2, platform, zipfile, statistics
from pathlib import Path
from collections import Counter
try:
 import zstandard as zstd
except ImportError: zstd=None
try:
 import lz4.frame as lz4f
except ImportError: lz4f=None
OUT=Path('sgx_results_current'); OUT.mkdir(exist_ok=True)
B=struct.Struct('<Q')

def words(data):
 assert len(data)%8==0
 return [x[0] for x in struct.iter_unpack('<Q',data)]
def tobytes(ws): return b''.join(B.pack(v) for v in ws)
def datasets(seed=1287):
 rng=random.Random(seed)
 n=4096
 # fixed bit representations, not float numeric conversions during encoding
 smooth=[struct.unpack('<Q',struct.pack('<d',100.0+math.sin(i/17)*0.12+i*0.0001))[0] for i in range(n)]
 constant=[struct.unpack('<Q',struct.pack('<d',21.25))[0]]*n
 stepped=[struct.unpack('<Q',struct.pack('<d',(i//128)*0.125))[0] for i in range(n)]
 noise=[rng.getrandbits(64) for _ in range(n)]
 mixed=smooth[:1024]+constant[:1024]+noise[:1024]+stepped[:1024]
 exceptions=[0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0x7ff0000000000001,0x0000000000000001,0x7fefffffffffffff]*32
 data={'smooth_f64':tobytes(smooth),'constant_f64':tobytes(constant),'stepped_f64':tobytes(stepped),'random_bits':tobytes(noise),'mixed_regimes':tobytes(mixed),'ieee_special_bits':tobytes(exceptions)}
 folder=Path('sgx_public_data')
 if folder.exists():
  for p in sorted(folder.glob('*.bin')):
   if 8<=p.stat().st_size<=8_000_000 and p.stat().st_size%8==0:
    data['external_'+p.name]=p.read_bytes()
 return data
class Writer:
 def __init__(self): self.buf=bytearray();self.pending=0;self.n=0;self.bits=0
 def put(self,v,n):
  assert 0<=n<=64 and 0<=v<(1<<n) if n else v==0
  for k in range(n-1,-1,-1):
   self.pending=(self.pending<<1)|((v>>k)&1);self.n+=1;self.bits+=1
   if self.n==8:self.buf.append(self.pending);self.n=0;self.pending=0
 def finish(self):
  if self.n:self.buf.append(self.pending<<(8-self.n))
  return bytes(self.buf)
class Reader:
 def __init__(self,buf): self.buf=buf;self.pos=0
 def get(self,n):
  if self.pos+n>8*len(self.buf):raise ValueError('truncated bitstream')
  a=0
  for _ in range(n):
   a=(a<<1)|((self.buf[self.pos//8]>>(7-self.pos%8))&1);self.pos+=1
  return a

def gamma(w,k):
 assert k>=1
 length=k.bit_length();w.put(0,length-1);w.put(k,length)
def ungamma(r):
 zeros=0
 while r.get(1)==0:
  zeros+=1
  if zeros>63:raise ValueError('invalid gamma')
 return (1<<zeros)|r.get(zeros)
def span_write(w,x):
 assert x>0
 leading=64-x.bit_length();trailing=(x&-x).bit_length()-1
 sig=x.bit_length()-trailing
 w.put(leading,6);w.put(sig-1,6);w.put(x>>trailing,sig)
def span_read(r):
 leading=r.get(6);sig=r.get(6)+1
 if leading+sig>64:raise ValueError('invalid span')
 payload=r.get(sig)
 if payload==0 or payload&1==0 or payload>>(sig-1)!=1:raise ValueError('invalid span payload')
 return payload<<(64-leading-sig)
# 20-byte exact framing, decoder does not rely on encoder context
HEADER=struct.Struct('<4sBBHIII') # magic,version,mode,reserved,word count, payload bytes, CRC32

def compress_xor(data,mode='span'):
 if len(data)%8:raise ValueError('Input must be a sequence of 64-bit words')
 ws=words(data);w=Writer();prev=0
 if mode=='span':
  for x in ws:
   d=x^prev;prev=x
   if not d:w.put(0,1)
   else:w.put(1,1);span_write(w,d)
 elif mode=='run':
  i=0
  while i<len(ws):
   j=i
   while j<len(ws) and ws[j]==prev:j+=1
   # gamma run-length + 1 represents zero or more repeated values
   gamma(w,j-i+1)
   if j==len(ws):break
   d=ws[j]^prev;span_write(w,d);prev=ws[j];i=j+1
 else:raise ValueError(mode)
 payload=w.finish();tag=0 if mode=='span' else 1
 return HEADER.pack(b'SGXC',1,tag,0,len(ws),len(payload),zlib.crc32(data))+payload

def decompress_xor(blob):
 if len(blob)<HEADER.size:raise ValueError('short header')
 magic,ver,mode,reserved,n,size,crc=HEADER.unpack_from(blob)
 if magic!=b'SGXC' or ver!=1 or mode not in (0,1) or reserved or len(blob)!=HEADER.size+size:raise ValueError('invalid header')
 r=Reader(blob[HEADER.size:]);ws=[];prev=0
 if mode==0:
  for _ in range(n):
   flag=r.get(1)
   d=span_read(r) if flag else 0
   prev^=d;ws.append(prev)
 else:
  while len(ws)<n:
   run=ungamma(r)-1
   if run>n-len(ws):raise ValueError('invalid zero run')
   ws.extend([prev]*run)
   if len(ws)==n:break
   prev^=span_read(r);ws.append(prev)
 result=tobytes(ws)
 if zlib.crc32(result)!=crc:raise ValueError('CRC error')
 return result

def codec(data,name):
 if name=='xor_span':return compress_xor(data,'span')
 if name=='xor_run':return compress_xor(data,'run')
 if name=='zlib6':return zlib.compress(data,6)
 if name=='lzma1':return lzma.compress(data,preset=1)
 if name=='bz2':return bz2.compress(data,compresslevel=5)
 if name=='zstd1' and zstd:return zstd.ZstdCompressor(level=1).compress(data)
 if name=='zstd3' and zstd:return zstd.ZstdCompressor(level=3).compress(data)
 if name=='lz4' and lz4f:return lz4f.compress(data)
 if name=='raw':return data
 raise ValueError('codec unavailable: '+name)
def restore(blob,name):
 if name.startswith('xor_'):return decompress_xor(blob)
 if name=='zlib6':return zlib.decompress(blob)
 if name=='lzma1':return lzma.decompress(blob)
 if name=='bz2':return bz2.decompress(blob)
 if name.startswith('zstd') and zstd:return zstd.ZstdDecompressor().decompress(blob)
 if name=='lz4' and lz4f:return lz4f.decompress(blob)
 if name=='raw':return blob
 raise ValueError(name)
def save_csv(file,rows):
 if not rows:return
 with (OUT/file).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save_json(file,obj):(OUT/file).write_text(json.dumps(obj,indent=2,default=str))
def finish(name):
 manifest={'experiment':name,'python':sys.version,'system':platform.platform(),'utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'files':[]}
 for p in sorted(OUT.iterdir()):
  if p.is_file():manifest['files'].append({'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 save_json('manifest.json',manifest)
 path=Path(name+'_ShiftGuardX_results.zip')
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
  for p in OUT.iterdir():
   if p.is_file():z.write(p,p.name)
 print('Created',path.resolve(),path.stat().st_size,'bytes')
 try:
  from google.colab import files
  files.download(str(path))
 except ImportError:pass
 return path


"""
## Versions and availability audit
"""


import importlib.metadata
versions={}
for package in ['zstandard','lz4','numpy']:
 try:versions[package]=importlib.metadata.version(package)
 except importlib.metadata.PackageNotFoundError:versions[package]='not installed'
status=[{'algorithm':x,'status':'NOT RUN — verified implementation and protocol needed','reason':'No validated compatible implementation provided; never treat as benchmark result'} for x in ['Gorilla','Chimp','ALP','Elf*','Dolphin','EAS']]
save_csv('specialized_baseline_status.csv',status)
save_json('environment.json',{'python':platform.python_version(),'packages':versions,'external_dataset_count':sum(k.startswith('external_') for k in datasets())})
print('Available Zstd',zstd is not None,'LZ4',lz4f is not None)


"""
## Matched coding benchmarks, end-to-end encode/decode
"""


methods=['raw','zlib6','bz2','lzma1','xor_span','xor_run']
if zstd:methods+=['zstd1','zstd3']
if lz4f:methods+=['lz4']
rows=[]
for name,b in datasets().items():
 for method in methods:
  for trial in range(5):
   st=time.perf_counter();packed=codec(b,method);et=time.perf_counter()-st
   st=time.perf_counter();unpacked=restore(packed,method);dt=time.perf_counter()-st
   assert unpacked==b
   rows.append({'dataset':name,'source':'external' if name.startswith('external_') else 'synthetic','algorithm':method,'trial':trial,'original_bytes':len(b),'compressed_bytes':len(packed),'bits_per_value':len(packed)*8/(len(b)//8),'encode_seconds':et,'decode_seconds':dt,'encode_MiB_s':len(b)/(et*1024**2) if et else 0,'decode_MiB_s':len(b)/(dt*1024**2) if dt else 0,'passed_roundtrip':True})
save_csv('matched_benchmarks.csv',rows)
summary=[]
for (name,method) in sorted({(r['dataset'],r['algorithm']) for r in rows}):
 a=[r for r in rows if r['dataset']==name and r['algorithm']==method]
 summary.append({'dataset':name,'algorithm':method,'source':a[0]['source'],'compressed_bytes':a[0]['compressed_bytes'],'mean_encode_MiB_s':statistics.mean(r['encode_MiB_s'] for r in a),'mean_decode_MiB_s':statistics.mean(r['decode_MiB_s'] for r in a),'all_verified':all(r['passed_roundtrip'] for r in a)})
save_csv('benchmark_summary.csv',summary)
save_json('journal_readiness.json',{'prototype_correctness':'tested','public_independent_data':'available only if external_ dataset present','specialized_baselines':'not run; see status CSV','novelty':'unproven','top_journal_ready':False,'next_steps':['Collect independent real binary64 streams','Integrate verified specialized benchmark implementations','Optimize C++ encoder/decoder for throughput','Perform literature prior-art and ablations']})
finish('NB09')

