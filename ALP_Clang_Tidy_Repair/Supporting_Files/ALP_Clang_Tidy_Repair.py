#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB43: ALP Clang Tidy Repair

Installs clang-tidy, configures official ALP using Clang, builds and exports full logs.

**Important:** This notebook does not assert novelty or Q1 readiness. Run all cells in Google Colab CPU.
"""

import os,sys,json,csv,time,zipfile,subprocess,hashlib,traceback,shutil,io,statistics
from pathlib import Path
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/'SGX_NB43_outputs';OUT.mkdir(parents=True,exist_ok=True)
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

# NB43 requires internet, sudo/root, and a Linux Colab runtime.
logs=[];status={'clone_ok':False,'cmake_ok':False,'build_ok':False,'official_benchmark_executed':False}
try:
 if shutil.which('apt-get'):
  logs.append(cmd(['bash','-lc','apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq clang clang-tidy cmake ninja-build git'],timeout=900))
 for tool in ('clang','clang++','clang-tidy','cmake','git'):
  status[tool]=shutil.which(tool)
 repo=ROOT/'SGX_ALP_official'
 if not (repo/'.git').exists():
  logs.append(cmd(['git','clone','--depth','1','https://github.com/cwida/ALP.git',str(repo)],timeout=300))
 status['clone_ok']=(repo/'.git').exists()
 if status['clone_ok']:
  revision=cmd(['git','-C',str(repo),'rev-parse','HEAD']);logs.append(revision);status['commit']=revision.get('stdout_tail','').strip()
  build=repo/'sgx_build'
  logs.append(cmd(['cmake','-S',str(repo),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_COMPILER='+str(status['clang']),'-DCMAKE_CXX_COMPILER='+str(status['clang++'])],timeout=300))
  status['cmake_ok']=logs[-1]['returncode']==0
  if status['cmake_ok']:
   logs.append(cmd(['cmake','--build',str(build),'--parallel','2'],timeout=1200))
   status['build_ok']=logs[-1]['returncode']==0
   status['executables']=[str(p.relative_to(build)) for p in build.rglob('*') if p.is_file() and os.access(p,os.X_OK) and not any(x in p.parts for x in ('CMakeFiles','_deps'))][:100]
except Exception as e:status['unexpected_error']=repr(e)
(OUT/'build_status.json').write_text(json.dumps(status,indent=2))
(OUT/'build_logs.json').write_text(json.dumps(logs,indent=2))
print(json.dumps(status,indent=2));savezip(43)

