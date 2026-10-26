#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB88F — Repaired Beijing native-codec benchmark

This fresh copy is a **Beijing-focused repair** of the missing nested ZIP and Pcodec import. It validates native decoders, runs chronological 25% calibration and 75% held-out PM2.5, PM10, SO2, NO2 measurements at eight stations, and exports `SGX_NB88F_results.zip`. Earlier US EPA results remain in the original NB88 archive; **this notebook does not repeat the EPA experiment**. No novel compression algorithm is claimed.
"""


import sys,subprocess,importlib,io,zipfile,urllib.request,hashlib,struct,zlib,lzma,json,csv,time,traceback
from pathlib import Path
import numpy as np,pandas as pd
for package in ('zstandard==0.25.0','floatbungler==0.1.2','pcodec==1.0.3'):
 try:
  p=subprocess.run([sys.executable,'-m','pip','-q','install',package],capture_output=True,text=True,timeout=180)
  print(package,'installed' if p.returncode==0 else 'not installed',p.stderr[-180:])
 except Exception as e:print(package,'install issue',e)
try:import zstandard as zstd
except Exception:zstd=None
try:from floatbungler import gorilla,chimp,chimp128,patas; fb={'gorilla':gorilla,'chimp':chimp,'chimp128':chimp128,'patas':patas}
except Exception as e:fb={};print('native codec error',e)
try:from pcodec import standalone as pc_standalone;import pcodec
except Exception as e:pc_standalone=None;print('Pcodec import issue:',e)
ROOT=Path('/content') if Path('/content').exists() else Path('/mnt/data');O=ROOT/'SGX_NB88F_outputs';O.mkdir(exist_ok=True)
methods=['zlib','lzma','xor_zstd','gorilla','chimp','chimp128','patas','pcodec']
def raw(a):return np.asarray(a,dtype='<f8').tobytes()
def xor(b):
 a=np.frombuffer(b,dtype='<u8');q=a.copy()
 if len(a)>1:q[1:]=a[1:]^a[:-1]
 return q.tobytes()
def unxor(b):return np.bitwise_xor.accumulate(np.frombuffer(b,dtype='<u8')).tobytes()
def encode(m,b):
 if m=='zlib':p=zlib.compress(b,6)
 elif m=='lzma':p=lzma.compress(b,preset=1)
 elif m=='xor_zstd':p=zstd.ZstdCompressor(level=3).compress(xor(b))
 elif m=='pcodec':p=bytes(pc_standalone.simple_compress(np.frombuffer(b,dtype='<f8').copy(),pcodec.ChunkConfig()))
 else:p=bytes(fb[m].encode(np.frombuffer(b,dtype='<f8').tolist()))
 return b'SGX88'+struct.pack('<BQQ',methods.index(m),len(b)//8,len(p))+p
def decode(frame):
 if frame[:5]!=b'SGX88' or len(frame)<22:raise ValueError('bad header')
 i,n,k=struct.unpack('<BQQ',frame[5:22]);p=frame[22:]
 if len(p)!=k:raise ValueError('bad length')
 m=methods[i]
 if m=='zlib':b=zlib.decompress(p)
 elif m=='lzma':b=lzma.decompress(p)
 elif m=='xor_zstd':b=unxor(zstd.ZstdDecompressor().decompress(p))
 elif m=='pcodec':b=raw(pc_standalone.simple_decompress(p))
 else:b=raw(np.asarray(fb[m].decode(p,n),dtype='<f8'))
 if len(b)!=n*8:raise ValueError('raw size')
 return b
def valid(m):return (m not in fb and m not in ('gorilla','chimp','chimp128','patas') or m in fb) and (m!='pcodec' or pc_standalone is not None) and (m!='xor_zstd' or zstd is not None)
def save_csv(name,rows):
 if not rows:rows=[{'status':'NO_ROWS'}]
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
edge=[];test=np.array([0,0x8000000000000000,0x7ff8000000000001,0x7ff0000000000000,0x0000000000000001],dtype='<u8').tobytes()
for m in methods:
 try:
  if not valid(m):raise ValueError('unavailable')
  assert decode(encode(m,test))==test
  edge.append({'method':m,'status':'BIT_EXACT'})
 except Exception as e:edge.append({'method':m,'status':'NOT_VERIFIED','error':repr(e)})
save_csv('edge_checks.csv',edge)
# Nested Beijing UCI archive is verified by descending into the inner PRSA2017 zip.
url='https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip'
source=[];results=[];errors=[];families=[]
try:
 request=urllib.request.Request(url,headers={'User-Agent':'SGX88F-academic-research'})
 with urllib.request.urlopen(request,timeout=120) as resp:blob=resp.read()
 source.append({'name':'Beijing','sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob),'url':url})
 with zipfile.ZipFile(io.BytesIO(blob)) as outer:
  nested=[k for k in outer.namelist() if k.lower().endswith('.zip')]
  if nested:
   chosen=next((k for k in nested if 'PRSA2017' in k),nested[0]);inner=zipfile.ZipFile(io.BytesIO(outer.read(chosen)))
  else:inner=outer
  for station in ['Aotizhongxin','Changping','Dongsi','Guanyuan','Gucheng','Shunyi','Wanliu','Wanshouxigong']:
   matches=[k for k in inner.namelist() if ('PRSA_Data_'+station+'_') in k and k.lower().endswith('.csv')]
   if len(matches)!=1:
    errors.append({'station':station,'error':'Could not locate exactly one station CSV'});continue
   data=pd.read_csv(io.BytesIO(inner.read(matches[0]))).sort_values(['year','month','day','hour'],kind='stable')
   for col in ['PM2.5','PM10','SO2','NO2']:
    if col not in data:continue
    a=pd.to_numeric(data[col],errors='coerce').to_numpy(dtype='<f8')[:32000];cut=len(a)//4
    cal=raw(a[:cut]);held=raw(a[cut:]);candidates={}
    for m in methods:
     if not valid(m):continue
     try:
      frame=encode(m,cal)
      if decode(frame)!=cal:raise AssertionError('not bit exact on calibration')
      candidates[m]=len(frame)
     except Exception as e:errors.append({'station':station,'column':col,'method':m,'stage':'cal','error':repr(e)})
    for group,options in [('generic',['zlib','lzma']),('specialized',['pcodec','gorilla','chimp','chimp128','patas']),('sgx',['xor_zstd'])]:
     selected=min([m for m in options if m in candidates],key=lambda x:candidates[x],default=None)
     if selected is None:continue
     try:
      t=time.perf_counter_ns();packed=encode(selected,held);em=(time.perf_counter_ns()-t)/1e6
      t=time.perf_counter_ns();d=decode(packed);dm=(time.perf_counter_ns()-t)/1e6
      if d!=held:raise AssertionError('not bitwise exact')
      results.append({'family':'beijing','station':station,'column':col,'group':group,'method':selected,'n_hold':len(a)-cut,'complete_bytes':len(packed),'encode_ms':em,'decode_ms':dm,'sha256_heldout':hashlib.sha256(held).hexdigest(),'exact':True})
     except Exception as e:errors.append({'station':station,'column':col,'method':selected,'stage':'hold','error':repr(e)})
except Exception as e:errors.append({'family':'Beijing','stage':'source_load','error':repr(e)})
save_csv('dataset_provenance.csv',source);save_csv('beijing_heldout_results.csv',results);save_csv('errors.csv',errors)
state={'state':'BEIJING_MEASURED' if results and not any(r.get('stage')=='source_load' for r in errors) else 'INCOMPLETE','streams':len(set((r['station'],r['column']) for r in results)),'native_methods_verified':[r['method'] for r in edge if r['status']=='BIT_EXACT'],'native_specialized_portability_not_authors_official':True,'ALP_not_included':True,'Q1_originality_established':False}
(O/'decision.json').write_text(json.dumps(state,indent=2))
p=ROOT/'SGX_NB88F_results.zip'
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
 for f in O.iterdir():z.write(f,f.name)
print('RESULTS:',state,'ZIP:',p)
try:
 from google.colab import files;files.download(str(p))
except ImportError:pass

