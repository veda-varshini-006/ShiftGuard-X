#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB40 — ALP_Clang_Build_Repair

Repairs NB37 by forcing Clang and archiving actual build logs. Building is not a codec comparison.
"""

import os,sys,json,csv,zipfile,subprocess,time,hashlib,io,statistics,zlib,lzma,struct,platform
from pathlib import Path
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/'SGX_NB40_outputs';OUT.mkdir(parents=True,exist_ok=True)
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


# Official ALP Clang repair. No fabricated codec output.
import shutil
repo=ROOT/'SGX_official_ALP';build=repo/'build_clang'
logs=[]
if not shutil.which('clang++') or not shutil.which('cmake'):
 logs.append(run(['bash','-lc','apt-get update -qq && apt-get install -y -qq clang cmake ninja-build'],timeout=600))
if not (repo/'.git').exists():logs.append(run(['git','clone','--depth','1','https://github.com/cwida/ALP',str(repo)],timeout=240))
status={'clang':shutil.which('clang'),'clangxx':shutil.which('clang++'),'cmake':shutil.which('cmake'),'repo_present':(repo/'.git').exists()}
if status['repo_present'] and status['clang'] and status['clangxx']:
 logs.append(run(['git','-C',str(repo),'rev-parse','HEAD']))
 logs.append(run(['cmake','-S',str(repo),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_C_COMPILER={status["clang"]}',f'-DCMAKE_CXX_COMPILER={status["clangxx"]}'],timeout=360))
 if logs[-1].get('returncode')==0:
  logs.append(run(['cmake','--build',str(build),'--parallel','2'],timeout=1200))
  status['build_ok']=logs[-1].get('returncode')==0
 else:status['build_ok']=False
 status['executables']=[str(p.relative_to(build)) for p in build.rglob('*') if p.is_file() and os.access(p,os.X_OK) and not p.name.endswith(('.so','.a'))][:100]
else:status['build_ok']=False
(OUT/'build_status.json').write_text(json.dumps(status,indent=2))
(OUT/'build_logs.json').write_text(json.dumps(logs,indent=2))
print('Build success:',status['build_ok'],'executables:',status.get('executables',[])[:15])
save('SGX_NB40_results.zip')

