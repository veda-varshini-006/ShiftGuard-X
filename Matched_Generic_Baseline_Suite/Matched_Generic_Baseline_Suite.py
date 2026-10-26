#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB38 — Matched_Generic_Baseline_Suite

Benchmarks identical float64 input bytes across general-purpose codecs and reversible byte/XOR transforms, with five timing repetitions and bitwise round-trip verification.

Run in a fresh Google Colab CPU runtime. Every result is saved in a ZIP. No result is called a fair ALP comparison unless the same exact input values and complete encoded sizes are available for both methods.
"""

import os,sys,time,json,csv,hashlib,platform,subprocess,zipfile,statistics,struct,zlib,lzma,shutil
from pathlib import Path
import numpy as np
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/'SGX_NB38_OUTPUTS'; OUT.mkdir(parents=True,exist_ok=True)
def csvout(name,rows):
 if rows:
  with (OUT/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def zipout(name):
 p=ROOT/name
 with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
  for f in OUT.rglob('*'):
   if f.is_file():z.write(f,f.relative_to(OUT))
 print('RESULT ZIP:',p,'bytes=',p.stat().st_size)
 try:
  from google.colab import files; files.download(str(p))
 except Exception: pass
 return p

from sklearn.datasets import load_diabetes,load_wine,load_breast_cancer,load_digits
sources=[('diabetes',load_diabetes().data),('wine',load_wine().data),('breast_cancer',load_breast_cancer().data),('digits',load_digits().data)]
def rawbytes(v): return np.ascontiguousarray(np.asarray(v,dtype='<f8')).tobytes()
def shuf(b):return np.frombuffer(b,dtype=np.uint8).reshape(-1,8).T.copy().tobytes()
def xor(b):
 a=np.frombuffer(b,dtype='<u8');q=a.copy()
 if len(q)>1:q[1:]=a[1:]^a[:-1]
 return q.astype('<u8').tobytes()
def enc(m,b):
 if m=='raw':return b
 if m=='zlib6':return zlib.compress(b,6)
 if m=='lzma1':return lzma.compress(b,preset=1)
 if m=='shuffle_zlib6':return zlib.compress(shuf(b),6)
 if m=='xor_zlib6':return zlib.compress(xor(b),6)
 if m=='xor_shuffle_zlib6':return zlib.compress(shuf(xor(b)),6)
 if m=='zstd3':
  import zstandard as zstd;return zstd.ZstdCompressor(level=3).compress(b)
 raise ValueError(m)
def dec(m,b,n):
 if m=='raw':return b
 if m=='zlib6':return zlib.decompress(b)
 if m=='lzma1':return lzma.decompress(b)
 if m=='shuffle_zlib6':return np.frombuffer(zlib.decompress(b),dtype=np.uint8).reshape(8,-1).T.copy().tobytes()
 if m=='xor_zlib6':
  a=np.frombuffer(zlib.decompress(b),dtype='<u8').copy()
  for i in range(1,len(a)):a[i]^=a[i-1]
  return a.astype('<u8').tobytes()
 if m=='xor_shuffle_zlib6':
  q=np.frombuffer(zlib.decompress(b),dtype=np.uint8).reshape(8,-1).T.copy().tobytes();a=np.frombuffer(q,dtype='<u8').copy()
  for i in range(1,len(a)):a[i]^=a[i-1]
  return a.astype('<u8').tobytes()
 if m=='zstd3':
  import zstandard as zstd;return zstd.ZstdDecompressor().decompress(b)
rows=[]; errors=[]; modes=['raw','zlib6','lzma1','shuffle_zlib6','xor_zlib6','xor_shuffle_zlib6']
try:
 import zstandard; modes.append('zstd3')
except ImportError: pass
for ds,a in sources:
 for j in range(a.shape[1]):
  b=rawbytes(a[:,j]); n=len(b)//8; digest=hashlib.sha256(b).hexdigest()
  for m in modes:
   try:
    samples=[]
    for _ in range(5):
     t=time.perf_counter_ns();c=enc(m,b);et=(time.perf_counter_ns()-t)/1e6
     t=time.perf_counter_ns();d=dec(m,c,n);dt=(time.perf_counter_ns()-t)/1e6
     assert d==b
     samples.append((len(c),et,dt))
    rows.append({'dataset':ds,'column':j,'n_values':n,'mode':m,'payload_bytes':samples[0][0],'container_bytes':samples[0][0],'bpv':8*samples[0][0]/n,'encode_ms_median':statistics.median(x[1] for x in samples),'decode_ms_median':statistics.median(x[2] for x in samples),'sha256_raw':digest,'bit_exact':True})
   except Exception as e: errors.append({'dataset':ds,'column':j,'mode':m,'error':repr(e)})
csvout('generic_baselines.csv',rows);csvout('errors.csv',errors)
(OUT/'accounting_note.txt').write_text('All methods receive identical raw little-endian float64 bytes. These measurements include codec payload bytes but do not add an application container header; they are internally comparable. No ALP/Chimp/Gorilla result is generated in this notebook.\n')
print('rows',len(rows),'errors',len(errors),'modes',modes)
zipout('SGX_NB38_results.zip')

