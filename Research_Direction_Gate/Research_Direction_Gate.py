#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB45: Research Direction Gate

Loads NB43 and NB44 archives and produces a conservative go/no-go report.

**Important:** This notebook does not assert novelty or Q1 readiness. Run all cells in Google Colab CPU.
"""

import os,sys,json,csv,time,zipfile,subprocess,hashlib,traceback,shutil,io,statistics
from pathlib import Path
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/'SGX_NB45_outputs';OUT.mkdir(parents=True,exist_ok=True)
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

# Upload NB43 and NB44 result ZIPs if not already in this runtime.
need=['SGX_NB43_results.zip','SGX_NB44_results.zip']
missing=[n for n in need if not (ROOT/n).exists()]
if missing:
 try:
  from google.colab import files
  print('Upload:',missing)
  files.upload()
 except ImportError:print('Missing archives:',missing)
statuses={}
for n in (43,44):
 p=ROOT/f'SGX_NB{n}_results.zip'
 if not p.exists():statuses[str(n)]={'status':'MISSING'};continue
 try:
  with zipfile.ZipFile(p) as z:
   member='build_status.json' if n==43 else 'match_status.json'
   statuses[str(n)]=json.loads(z.read(member))
 except Exception as e:statuses[str(n)]={'status':'UNREADABLE','error':repr(e)}
build=statuses.get('43',{}).get('build_ok',False)
matched=statuses.get('44',{}).get('alp_same_input_proven',False) and statuses.get('44',{}).get('alp_compressed_size_matched',False)
result={'alp_build_ok':build,'fair_alp_comparison_established':matched,'new_codec_novelty_established':False,'Q1_ready':False,'decision':'RESEARCH_REDESIGN_REQUIRED' if build and not matched else 'BENCHMARK_INFRASTRUCTURE_INCOMPLETE','notes':['A successful ALP build does not prove compression advantage.','NB44 deliberately does not claim an ALP matchup without a same-stream API.','A novel method and specialized baselines are still required.'],'inputs':statuses}
(OUT/'final_decision.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2)[:6500]);savezip(45)

