#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB42 — Conservative_Research_Gate

Combines result ZIPs and issues a conservative research decision; upload inputs in Colab when prompted.
"""

import os,sys,json,csv,zipfile,subprocess,time,hashlib,io,statistics,zlib,lzma,struct,platform
from pathlib import Path
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/'SGX_NB42_outputs';OUT.mkdir(parents=True,exist_ok=True)
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


# Upload NB40/NB41 and, optionally, previous results ZIPs when prompted.
# Missing files produce INCOMPLETE, never a fabricated comparison.
try:
 from google.colab import files
 print('Upload NB40_results.zip and NB41_results.zip (and any prior ZIPs you want archived).')
 uploaded=files.upload()
 for filename,content in uploaded.items():(ROOT/Path(filename).name).write_bytes(content)
except ImportError:pass
inputs={};data=[]
for number in [40,41,37,38,39]:
 p=ROOT/f'SGX_NB{number}_results.zip'
 if not p.exists():inputs[str(number)]={'status':'MISSING'};continue
 try:
  with zipfile.ZipFile(p) as z:
   inputs[str(number)]={'status':'AVAILABLE','files':z.namelist()}
   if number==40 and 'build_status.json' in z.namelist():inputs[str(number)]['build']=json.loads(z.read('build_status.json'))
   if number==41 and 'matched_generic.csv' in z.namelist():data=list(csv.DictReader(io.StringIO(z.read('matched_generic.csv').decode())))
 except Exception as e:inputs[str(number)]={'status':'ERROR','error':repr(e)}
summary={'inputs':inputs,'generic_records':len(data),'generic_streams':len(set((r['family'],r['column']) for r in data)),'matched_ALP':False,'novelty_established':False,'Q1_ready':False,'decision':'INCOMPLETE_OR_REDESIGN','reason':'A new coding contribution and a fair official specialized-codec benchmark remain unestablished.'}
if data:
 valid=[r for r in data if r['mode']!='raw'];groups={}
 for r in valid:groups.setdefault((r['family'],r['column']),[]).append(r)
 summary['generic_best_modes']={m:sum(min(v,key=lambda x:float(x['bytes']))['mode']==m for v in groups.values()) for m in sorted(set(r['mode'] for r in valid))}
(OUT/'final_gate.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='inputs'},indent=2))
save('SGX_NB42_results.zip')

