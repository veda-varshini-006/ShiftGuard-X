#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB41 — Identical_Input_Generic_Codecs

Measures self-describing end-to-end sizes, exact decoding, and real encode/decode timings for generic codecs. Official ALP comparison is explicitly not implemented, rather than invented.
"""

import os,sys,json,csv,zipfile,subprocess,time,hashlib,io,statistics,zlib,lzma,struct,platform
from pathlib import Path
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/'SGX_NB41_outputs';OUT.mkdir(parents=True,exist_ok=True)
def save(name):
 target=ROOT/name
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for f in OUT.rglob('*'):
   if f.is_file():z.write(f,f.relative_to(OUT))
 print('OUTPUT:',target)
 try:
  from google.colab import files;files.download(str(target))
 except ImportError:pass
 return target
def run(cmd,timeout=600,env=None):
 t=time.monotonic()
 try:
  p=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout,env=env)
  return {'cmd':cmd,'returncode':p.returncode,'elapsed_sec':round(time.monotonic()-t,3),'stdout_tail':p.stdout[-10000:],'stderr_tail':p.stderr[-10000:]}
 except Exception as e:return {'cmd':cmd,'error':repr(e),'elapsed_sec':round(time.monotonic()-t,3)}
def writecsv(name,rows):
 if rows:
  with (OUT/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


# Reproduce real numerical streams and benchmark *generic* codecs with complete self-describing frame accounting.
# ALP is not silently treated as a matched codec: this notebook reports ALP interface as UNVERIFIED.
import numpy as np
from sklearn.datasets import load_diabetes,load_wine,load_breast_cancer,load_digits
src=[('diabetes',load_diabetes().data),('wine',load_wine().data),('breast_cancer',load_breast_cancer().data),('digits',load_digits().data)]
try:
 import zstandard as zstd
except ImportError:zstd=None
modes=['raw','zlib6','lzma1']+(['zstd3'] if zstd else [])
def codec(m,raw):
 if m=='raw':return raw
 if m=='zlib6':return zlib.compress(raw,6)
 if m=='lzma1':return lzma.compress(raw,preset=1)
 if m=='zstd3':return zstd.ZstdCompressor(level=3).compress(raw)
def uncodec(m,p):
 if m=='raw':return p
 if m=='zlib6':return zlib.decompress(p)
 if m=='lzma1':return lzma.decompress(p)
 if m=='zstd3':return zstd.ZstdDecompressor().decompress(p)
def framed(m,raw):
 payload=codec(m,raw);name=m.encode()
 return b'SGX4'+struct.pack('<QBI',len(raw)//8,len(name),len(payload))+name+payload
def decoded(blob):
 assert blob[:4]==b'SGX4'
 n,ml,sz=struct.unpack('<QBI',blob[4:17]);m=blob[17:17+ml].decode();p=blob[17+ml:]
 assert len(p)==sz
 raw=uncodec(m,p);assert len(raw)==n*8
 return raw
rows=[];errs=[]
for family,data in src:
 for col in range(data.shape[1]):
  raw=np.ascontiguousarray(data[:,col],dtype='<f8').tobytes();digest=hashlib.sha256(raw).hexdigest()
  for m in modes:
   try:
    vals=[]
    for repeat in range(5):
     t=time.perf_counter_ns();blob=framed(m,raw);enc=(time.perf_counter_ns()-t)/1e6
     t=time.perf_counter_ns();got=decoded(blob);dec=(time.perf_counter_ns()-t)/1e6
     assert got==raw;vals.append((len(blob),enc,dec))
    rows.append({'family':family,'column':col,'n':len(raw)//8,'mode':m,'bytes':vals[0][0],'bpv':8*vals[0][0]/(len(raw)//8),'encode_ms_median':statistics.median(v[1] for v in vals),'decode_ms_median':statistics.median(v[2] for v in vals),'raw_sha256':digest,'bit_exact':True})
   except Exception as e:errs.append({'family':family,'column':col,'mode':m,'error':repr(e)})
writecsv('matched_generic.csv',rows);writecsv('errors.csv',errs)
(OUT/'alp_match_status.json').write_text(json.dumps({'status':'NOT_RUN','reason':'A working official ALP arbitrary-input encode/decode interface with comparable container metadata has not been validated. NB40 only builds ALP. Do not infer an ALP comparison.'},indent=2))
print('rows',len(rows),'errors',len(errs));save('SGX_NB41_results.zip')

