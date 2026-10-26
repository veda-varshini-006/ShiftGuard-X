#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB44: ALP Same Input Audit

Tests identical raw numerical bytes for generic codecs and explicitly checks whether a matched official ALP result is available.

**Important:** This notebook does not assert novelty or Q1 readiness. Run all cells in Google Colab CPU.
"""

import os,sys,json,csv,time,zipfile,subprocess,hashlib,traceback,shutil,io,statistics
from pathlib import Path
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/'SGX_NB44_outputs';OUT.mkdir(parents=True,exist_ok=True)
def savezip(n):
 p=ROOT/f'SGX_NB{n}_results.zip'
 with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(OUT.rglob('*')):
   if f.is_file():z.write(f,f.relative_to(OUT))
 print('RESULTS ZIP',p)
 try:
  from google.colab import files
  files.download(str(p))
 except ImportError:pass
 return p
def cmd(args,timeout=300,env=None):
 t=time.time()
 try:
  p=subprocess.run(args,text=True,capture_output=True,timeout=timeout,env=env)
  return {'cmd':args,'returncode':p.returncode,'elapsed_sec':round(time.time()-t,2),'stdout_tail':p.stdout[-12000:],'stderr_tail':p.stderr[-12000:]}
 except Exception as e:return {'cmd':args,'returncode':-1,'error':repr(e),'elapsed_sec':round(time.time()-t,2)}

# Same-input identity gate. Optional: upload prior NB32 and NB41 ZIPs via Colab files picker.
import numpy as np,zlib,lzma
try:
 from sklearn.datasets import load_diabetes,load_wine,load_breast_cancer,load_digits
 datasets=[('diabetes',load_diabetes().data),('wine',load_wine().data),('breast_cancer',load_breast_cancer().data),('digits',load_digits().data)]
except Exception as e:raise RuntimeError('scikit-learn is required') from e
rows=[]
for name,data in datasets:
 for j in range(data.shape[1]):
  raw=np.asarray(data[:,j],dtype='<f8').tobytes();sha=hashlib.sha256(raw).hexdigest();n=len(raw)//8
  for codec in ('zlib6','lzma1'):
   t=time.perf_counter_ns();c=zlib.compress(raw,6) if codec=='zlib6' else lzma.compress(raw,preset=1);enc=(time.perf_counter_ns()-t)/1e6
   t=time.perf_counter_ns();back=zlib.decompress(c) if codec=='zlib6' else lzma.decompress(c);dec=(time.perf_counter_ns()-t)/1e6
   assert back==raw
   rows.append({'dataset':name,'column':j,'n':n,'raw_sha256':sha,'codec':codec,'payload_bytes':len(c),'encode_ms':enc,'decode_ms':dec,'exact':True})
with (OUT/'generic_same_input.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
# ALP's official benchmark must expose decoded bytes and matching raw SHA-256 for a fair claim.
found=[]
for name in ('SGX_NB43_results.zip','SGX_NB32_Official_ALP_results.zip','SGX_NB41_results.zip'):
 p=ROOT/name
 if p.exists():
  with zipfile.ZipFile(p) as z:found.append({'file':name,'members':z.namelist()[:100]})
status={'generic_streams':len(rows)//2,'generic_rows':len(rows),'alp_same_input_proven':False,'alp_compressed_size_matched':False,'fair_specialized_comparison':'NOT_ESTABLISHED','available_archives':found,'explanation':'ALP build or native BPV alone is insufficient: same raw SHA256, exact roundtrip, and matching size accounting are required. No ALP result is invented.'}
(OUT/'match_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2)[:5000]);savezip(44)

