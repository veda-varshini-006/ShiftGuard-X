#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB50: Lossless Decimal Lattice Ablation

**Purpose:** Tests exact IEEE-754 decimal lattice representations with explicit exception streams and per-stream size ablations on real UCI Wine Quality data. This is an established-family technique, not a claimed novel compressor.

**Instructions:** Run **Runtime → Run all** in a fresh Google Colab CPU runtime. The notebook writes `SGX_NB50_results.zip` and attempts to download it. Each notebook is independent. If network access or ALP fails, status and logs are still exported rather than inventing numbers.

**Research integrity:** Bitwise round-trip success is necessary but not sufficient for novelty, competitive compression, or a Q1 publication.
"""

import os,sys,json,csv,io,hashlib,time,zipfile,statistics,zlib,lzma,platform,traceback,subprocess,shutil,struct,math
from pathlib import Path
import numpy as np
ROOT=Path('/content') if Path('/content').exists() else Path('/mnt/data')
OUT=ROOT/'SGX_NB50_outputs';OUT.mkdir(parents=True,exist_ok=True)
def write_csv(name,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def write_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=str))
def export():
 out=ROOT/'SGX_NB50_results.zip'
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(OUT.rglob('*')):
   if f.is_file():z.write(f,f.relative_to(OUT))
 print('RESULTS_ZIP:',out,'BYTES:',out.stat().st_size)
 try:
  from google.colab import files
  files.download(str(out))
 except ImportError:pass
 except Exception as e:print('Automatic download unavailable; download manually:',out,repr(e))
 return out
def run(cmd,timeout=600,cwd=None,env=None):
 t=time.monotonic()
 try:
  p=subprocess.run(cmd,cwd=cwd,env=env,capture_output=True,text=True,timeout=timeout)
  return {'command':cmd,'returncode':p.returncode,'duration_s':round(time.monotonic()-t,3),'stdout_tail':p.stdout[-24000:],'stderr_tail':p.stderr[-12000:]}
 except Exception as e:return {'command':cmd,'returncode':-999,'duration_s':round(time.monotonic()-t,3),'error':repr(e)}
def raw(v):return np.ascontiguousarray(v,dtype='<f8').tobytes()
def sha(b):return hashlib.sha256(b).hexdigest()
write_json('environment.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'smoke':os.environ.get('SGX_LOCAL_SMOKE_ONLY')=='1'})

def wine_streams():
 import urllib.request
 sources={'red':'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
          'white':'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'}
 for name,url in sources.items():
  text=urllib.request.urlopen(url,timeout=40).read().decode('utf-8')
  reader=csv.DictReader(io.StringIO(text),delimiter=';');rows=list(reader)
  usable=(len(rows)//1024)*1024
  if usable<1024:raise ValueError('Too few real samples '+name)
  for j,key in enumerate(reader.fieldnames):
   arr=np.array([float(r[key]) for r in rows[:usable]],dtype='<f8')
   yield {'stream':f'{name}_wine__c{j:02d}','dataset':name,'column':key,'values':arr,'n':len(arr),'url':url,'sha256':sha(raw(arr))}

# Exploratory exact decimal lattice with raw-bit exceptions and compact integer bitpacking.
# Decimal scaling, exception streams and bitpacking all have prior art (including ALP-family ideas).
# No novelty or performance claims are made by this notebook.
SCALES=[1,10,100,1000,10000]
MAG=b'QPK1'
def bitpack(values,width):
  acc=0;pending=0;out=bytearray()
  for v in values:
   v=int(v)
   if v<0 or (width and v >= (1<<width)) or (not width and v):raise ValueError('Out-of-range packed value')
   acc|=v<<pending;pending+=width
   while pending>=8:
    out.append(acc&255);acc>>=8;pending-=8
  if pending:out.append(acc&255)
  return bytes(out)
def bitunpack(buf,width,count):
  acc=0;pending=0;it=iter(buf);out=[]
  for _ in range(count):
   while pending<width:
    try:acc|=next(it)<<pending
    except StopIteration:raise ValueError('Truncated packed values')
    pending+=8
   out.append(acc&((1<<width)-1));acc>>=width;pending-=width
  if any(it):raise ValueError('Trailing nonzero packed bytes')
  return out
def lattice_pack(r,scale):
 if len(r)%8:raise ValueError('misaligned input')
 a=np.frombuffer(r,dtype='<f8'); n=len(a)
 # Word equality rather than numeric equality handles -0.0 and NaN payloads.
 original=np.frombuffer(r,dtype='<u8')
 present=[];integers=[];exceptions=[]
 for i,v in enumerate(a):
  v=float(v);accept=False;k=0
  if math.isfinite(v) and -2147483648 <= v*scale <= 2147483647:
   candidate=int(round(v*scale))
   if -2147483648<=candidate<=2147483647:
    restored=float(candidate)/scale
    if struct.pack('<d',restored)==r[i*8:i*8+8]:accept=True;k=candidate
  if accept:present.append(1);integers.append(k)
  else:present.append(0);exceptions.append(r[i*8:i*8+8])
 mn=min(integers) if integers else 0
 shifted=[v-mn for v in integers]
 width=max(shifted,default=0).bit_length()
 packed=bitpack(shifted,width)
 marker=bitpack(present,1)
 sections=[zlib.compress(marker,6),zlib.compress(packed,6),zlib.compress(b''.join(exceptions),6)]
 # 4 magic, 1 scale index, 4 n, 4 accepted, 8 min, 1 width, 12 compressed section lengths
 header=MAG+struct.pack('<BIIqBIII',SCALES.index(scale),n,len(integers),mn,width,*[len(x) for x in sections])
 return header+b''.join(sections)
def lattice_unpack(blob):
 if blob[:4]!=MAG or len(blob)<34:raise ValueError('Bad header')
 k,n,naccept,mn,width,a,b,c=struct.unpack('<BIIqBIII',blob[4:34]);scale=SCALES[k]
 if n>100_000_000 or naccept>n or width>64:raise ValueError('Invalid parameters')
 if len(blob)!=34+a+b+c:raise ValueError('Length mismatch')
 p=34;marker=zlib.decompress(blob[p:p+a]);p+=a
 packed=zlib.decompress(blob[p:p+b]);p+=b
 rawexceptions=zlib.decompress(blob[p:p+c]);p+=c
 mask=bitunpack(marker,1,n)
 if sum(mask)!=naccept:raise ValueError('Mask count mismatch')
 numbers=bitunpack(packed,width,naccept)
 if len(rawexceptions)!=(n-naccept)*8:raise ValueError('Exception size mismatch')
 it=iter(numbers);e=0;out=bytearray()
 for tag in mask:
  if tag:
   val=(next(it)+mn)/scale
   out.extend(struct.pack('<d',val))
  else:out.extend(rawexceptions[e:e+8]);e+=8
 return bytes(out)
def generic_pack(mode,r):
 if mode=='zlib6':return zlib.compress(r,6)
 if mode=='lzma1':return lzma.compress(r,preset=1)
 if mode=='zstd3':
  import zstandard as zstd
  return zstd.ZstdCompressor(level=3).compress(r)
 raise ValueError(mode)
def generic_unpack(mode,b):
 if mode=='zlib6':return zlib.decompress(b)
 if mode=='lzma1':return lzma.decompress(b)
 if mode=='zstd3':
  import zstandard as zstd
  return zstd.ZstdDecompressor().decompress(b)
 raise ValueError(mode)
def trial(mode,r,reps=3):
 enc=lattice_pack if mode.startswith('lattice') else generic_pack
 sizes=[];et=[];dt=[]
 for _ in range(reps):
  t=time.perf_counter_ns()
  if mode.startswith('lattice'):b=lattice_pack(r,SCALES[int(mode.split('_')[1])])
  else:b=generic_pack(mode,r)
  et.append((time.perf_counter_ns()-t)/1e6)
  t=time.perf_counter_ns()
  decoded=lattice_unpack(b) if mode.startswith('lattice') else generic_unpack(mode,b)
  dt.append((time.perf_counter_ns()-t)/1e6)
  assert decoded==r,('lossless failure',mode)
  sizes.append(len(b))
 return {'bytes':sizes[0],'encode_ms_median':float(statistics.median(et)),'decode_ms_median':float(statistics.median(dt))}
def tests():
 edge_bits=np.array([0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0x7ff8000000001234,0x0000000000000001,0xffffffffffffffff],dtype='<u8')
 examples=[b'',edge_bits.tobytes(),edge_bits.tobytes()*8,raw(np.arange(512,dtype='<f8')/10),raw(np.zeros(200))]
 checks=[]
 for a in examples:
  for scale in SCALES:
   c=lattice_pack(a,scale);d=lattice_unpack(c)
   checks.append({'n':len(a)//8,'scale':scale,'success':d==a})
 assert all(x['success'] for x in checks)
 return checks

rows=[];errors=[];status={'purpose':'hypothesis screening; no novelty claim','dataset':'UCI Wine Quality (real data)','data_download_success':False}
try:
 write_csv('edge_case_tests.csv',tests())
 streams=list(wine_streams());status['data_download_success']=True
 modes=['zlib6','lzma1']+['lattice_'+str(i) for i in range(len(SCALES))]
 try:import zstandard;modes.insert(2,'zstd3')
 except ImportError:pass
 for item in streams:
  b=raw(item['values'])
  for mode in modes:
   try:
    result=trial(mode,b,3)
    rows.append({'stream':item['stream'],'dataset':item['dataset'],'column':item['column'],'n':item['n'],'source_url':item['url'],'sha256':sha(b),'mode':mode,'bpv':8*result['bytes']/item['n'],'exact':True,**result})
   except Exception as e:errors.append({'stream':item['stream'],'mode':mode,'error':repr(e)})
 write_csv('ablation.csv',rows);write_csv('errors.csv',errors)
 if rows:
  best={}
  for r in rows:
   k=r['stream'];best.setdefault(k,[]).append(r)
  summ=[]
  for stream,rs in best.items():
   g=min((r for r in rs if r['mode'] in ('zlib6','lzma1','zstd3')),key=lambda r:r['bytes'])
   l=min((r for r in rs if r['mode'].startswith('lattice')),key=lambda r:r['bytes'])
   summ.append({'stream':stream,'best_generic':g['mode'],'best_lattice':l['mode'],'generic_bytes':g['bytes'],'lattice_bytes':l['bytes'],'oracle_savings_percent':100*(g['bytes']-l['bytes'])/g['bytes']})
  write_csv('oracle_stream_summary_diagnostic_only.csv',summ)
  status.update({'streams':len(summ),'lattice_oracle_wins':sum(r['oracle_savings_percent']>0 for r in summ),'lattice_oracle_losses':sum(r['oracle_savings_percent']<0 for r in summ)})
 status['stage']='COMPLETE'
except Exception as e:status.update({'stage':'ERROR','error':repr(e),'traceback':traceback.format_exc()[-5000:]})
finally:
 write_json('status.json',status);print('STATUS',status);export()

