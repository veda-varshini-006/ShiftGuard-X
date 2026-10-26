#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB39 — Final_Evidence_Gate

Combines the available NB32/NB19F ALP result archives with NB37 build logs and NB38 generic measurements. Requires explicit input identity before permitting a matched-comparison claim.

Run in a fresh Google Colab CPU runtime. Every result is saved in a ZIP. No result is called a fair ALP comparison unless the same exact input values and complete encoded sizes are available for both methods.
"""

import os,sys,time,json,csv,hashlib,platform,subprocess,zipfile,statistics,struct,zlib,lzma,shutil
from pathlib import Path
import numpy as np
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/'SGX_NB39_OUTPUTS'; OUT.mkdir(parents=True,exist_ok=True)
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

import glob
INPUTS={ 'NB37':ROOT/'SGX_NB37_results.zip','NB38':ROOT/'SGX_NB38_results.zip','NB32':ROOT/'SGX_NB32_Official_ALP_results.zip','NB19F':ROOT/'SGX_NB19F_Official_ALP_results.zip'}
summary={'purpose':'final evidence gate','inputs':{},'same_input_specialized_comparison':'NOT_ESTABLISHED','decision':'INCOMPLETE'}
for label,p in INPUTS.items():
 if p.exists():
  try:
   with zipfile.ZipFile(p) as z:
    files=z.namelist();summary['inputs'][label]={'zip':p.name,'files':files}
    for fn in files:
     if fn.endswith('.csv') and any(k in fn.lower() for k in ('alp','benchmark','result','size')):
      dest=OUT/f'{label}_{Path(fn).name}';dest.write_bytes(z.read(fn))
  except Exception as e:summary['inputs'][label]={'error':repr(e)}
 else:summary['inputs'][label]={'status':'not uploaded/not found'}
# Only mark a matched comparison if both sides explicitly share a raw input hash and have codec-specific size values.
alp_files=list(OUT.glob('NB*_*.csv'));summary['candidate_csvs']=[p.name for p in alp_files]
summary['limitations']=['Build logs are not benchmark measurements.','NB32/NB19F ALP outputs must expose input identity and complete encoded size before comparing to NB38.','Do not compare bits/value from one tool against bytes excluding metadata from another.','No publication readiness or novelty claim follows from this audit.']
(OUT/'final_gate.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2)[:12000])
zipout('SGX_NB39_results.zip')

