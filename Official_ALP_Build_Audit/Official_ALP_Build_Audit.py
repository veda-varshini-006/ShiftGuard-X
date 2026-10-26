#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB37 — Official_ALP_Build_Audit

Reproducibly clones the official ALP repository, records the commit, attempts its documented CMake build and archives logs. It deliberately does not fabricate same-input results if the upstream benchmark interface is not available.

Run in a fresh Google Colab CPU runtime. Every result is saved in a ZIP. No result is called a fair ALP comparison unless the same exact input values and complete encoded sizes are available for both methods.
"""

import os,sys,time,json,csv,hashlib,platform,subprocess,zipfile,statistics,struct,zlib,lzma,shutil
from pathlib import Path
import numpy as np
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/'SGX_NB37_OUTPUTS'; OUT.mkdir(parents=True,exist_ok=True)
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

import urllib.request
repo=ROOT/'SGX_official_ALP'
status={'repo':'https://github.com/cwida/ALP','clone':'NOT_RUN','cmake':'NOT_RUN','executables':[],'note':'Building repository is not itself a same-input codec comparison.'}
try:
 if not repo.exists():
  r=subprocess.run(['git','clone','--depth','1','https://github.com/cwida/ALP',str(repo)],capture_output=True,text=True,timeout=240)
  status['clone']='PASS' if r.returncode==0 else 'FAIL';status['clone_log']=(r.stdout+'\n'+r.stderr)[-5000:]
 else: status['clone']='EXISTING_CHECKOUT'
 if repo.exists():
  (OUT/'alp_commit.txt').write_text(subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip())
  status['top_level']=sorted(p.name for p in repo.iterdir())
  status['build_docs']=[str(p.relative_to(repo)) for p in repo.rglob('*') if p.is_file() and p.name.lower() in ('readme.md','benchmarking.md','cmakelists.txt')][:80]
  build=repo/'build';build.mkdir(exist_ok=True)
  cm=subprocess.run(['cmake','-S',str(repo),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release'],capture_output=True,text=True,timeout=300)
  status['cmake']='PASS' if cm.returncode==0 else 'FAIL';status['cmake_log']=(cm.stdout+'\n'+cm.stderr)[-8000:]
  if cm.returncode==0:
   b=subprocess.run(['cmake','--build',str(build),'--config','Release','-j2'],capture_output=True,text=True,timeout=900)
   status['build']='PASS' if b.returncode==0 else 'FAIL';status['build_log']=(b.stdout+'\n'+b.stderr)[-8000:]
   status['executables']=[str(p) for p in build.rglob('*') if p.is_file() and os.access(p,os.X_OK) and not p.name.endswith(('.so','.a'))][:100]
except Exception as e: status['error']=repr(e)
(OUT/'official_alp_build_audit.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status,indent=2)[:12000])
zipout('SGX_NB37_results.zip')

