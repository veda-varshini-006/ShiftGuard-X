#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB59: ALP_Source_Forensics_and_Benchmark

Pins the official ALP commit, locates the exact parser exception in source, tests documented binary/CSV configurations, and records official native results if the benchmark actually succeeds. ALP native bits/value are never mislabeled as same-accounting container sizes.

Run **Runtime → Run all** in a fresh Google Colab CPU session. The final cell writes a results ZIP and attempts a browser download. This work is exploratory and does not establish novelty, journal acceptance, or ALP superiority.
"""


import os,sys,subprocess,time,json,traceback,hashlib,csv,io,zipfile,platform,re,shutil
from pathlib import Path
import numpy as np
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/'SGX_NB59_outputs';OUT.mkdir(parents=True,exist_ok=True)
ALP_COMMIT='31ca0ed11c93c99d3f5b5c30e01a3e1c3832d3ce'
REPO=ROOT/'SGX_NB59_ALP';BUILD=REPO/'sgx_nb59_build';DATA=ROOT/'SGX_NB59_INPUT';DATA.mkdir(parents=True,exist_ok=True)
LOGS=[]
def run(args,cwd=None,timeout=450,env=None):
 try:
  r=subprocess.run(args,cwd=cwd,capture_output=True,text=True,timeout=timeout,env=env)
  rec={'cmd':args,'returncode':r.returncode,'stdout':r.stdout[-18000:],'stderr':r.stderr[-18000:]}
 except Exception as e:rec={'cmd':args,'returncode':-999,'error':repr(e)}
 LOGS.append(rec);print('CALL', ' '.join(args[:7]),'RC',rec['returncode'])
 return rec

def outzip():
 (OUT/'command_logs.json').write_text(json.dumps(LOGS,indent=2))
 p=ROOT/'SGX_NB59_results.zip'
 with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(OUT.rglob('*')):
   if f.is_file():z.write(f,f.relative_to(OUT))
 print('RESULTS:',p,p.stat().st_size)
 try:
  from google.colab import files;files.download(str(p))
 except Exception:pass

def need_tool(binary,apt=None):
 if shutil.which(binary):return
 name=apt or binary
 r=run(['bash','-lc','DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq '+name],timeout=400)
 if r['returncode']!=0 or not shutil.which(binary):raise RuntimeError('Could not install '+binary)

def sha(b):return hashlib.sha256(b).hexdigest()
status={'stage':'START','source':'official cwida/ALP','commit_requested':ALP_COMMIT,
        'official_native_size_accounting':'bits per value; not complete SGX5 file bytes',
        'native_alp_decode_verified':False,'same_input_verified':False,
        'benchmark_data_substitution':'NEVER','probe_log':[]}
SMOKE=os.environ.get('SGX_LOCAL_SMOKE')=='1'
try:
 if SMOKE:
  status['stage']='SMOKE_ONLY';status['note']='Does not download or build ALP; no benchmark asserted.'
 else:
  for binary,apt in [('git','git'),('cmake','cmake'),('clang','clang'),('clang++','clang'),('clang-tidy','clang-tidy')]:need_tool(binary,apt)
  if not (REPO/'.git').exists():
   c=run(['git','clone','https://github.com/cwida/ALP',str(REPO)],timeout=250)
   if c['returncode']!=0:raise RuntimeError('clone failed')
  c=run(['git','fetch','origin',ALP_COMMIT],cwd=REPO,timeout=120)
  # A cloned repository may already have the commit even if fetch is rate-limited.
  c=run(['git','checkout','--detach',ALP_COMMIT],cwd=REPO)
  if c['returncode']!=0:raise RuntimeError('Cannot pin ALP commit')
  status['commit_observed']=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True).stdout.strip()
  if status['commit_observed']!=ALP_COMMIT:raise RuntimeError('Wrong checkout')
  # Source-level traceback: find the literal exception, archive actual containing code.
  locations=[]
  for src in REPO.rglob('*'):
   if src.is_file() and src.suffix in ('.hpp','.h','.cc','.cpp','.cxx'):
    try:t=src.read_text(encoding='utf-8',errors='replace')
    except Exception:continue
    for needle in ['No bin or csv file specified','bench_alp_on_your_dataset','your_own_dataset.csv']:
     off=t.find(needle)
     if off!=-1:
      lineno=t[:off].count('\n')+1;lines=t.splitlines()
      locations.append({'file':str(src.relative_to(REPO)),'needle':needle,'line':lineno,
                        'context':'\n'.join(f'{j+1}: {lines[j]}' for j in range(max(0,lineno-30),min(len(lines),lineno+32)))})
  (OUT/'source_parser_forensics.json').write_text(json.dumps(locations,indent=2))
  status['parser_source_matches']=len(locations)
  cfg=REPO/'benchmarks'/'your_own_dataset.csv'
  status['config_exists']=cfg.parent.exists()
  if not cfg.parent.exists():raise RuntimeError('Missing expected ALP benchmark configuration location')
  cmd=['cmake','-S',str(REPO),'-B',str(BUILD),'-DCMAKE_BUILD_TYPE=Release',
       '-DALP_BUILD_BENCHMARKING=ON','-DALP_ENABLE_VERBOSE_OUTPUT=ON',
       '-DALP_ENABLE_CLANG_TIDY=OFF','-DCMAKE_C_COMPILER='+shutil.which('clang'),
       '-DCMAKE_CXX_COMPILER='+shutil.which('clang++')]
  if run(cmd,timeout=350)['returncode']!=0:raise RuntimeError('CMake configuration failed')
  if run(['cmake','--build',str(BUILD),'--target','bench_your_dataset','--parallel','2'],timeout=1500)['returncode']!=0:raise RuntimeError('ALP benchmark build failed')
  exe=BUILD/'benchmarks'/'bench_your_dataset'
  if not exe.exists():raise RuntimeError('Expected binary not produced')
  # Exactly 1024 real UCI Wine Quality values. Preserves IEEE bit pattern for both codecs.
  from urllib.request import urlopen
  url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
  local=ROOT/'winequality-red.csv'
  if local.exists():payload=local.read_bytes();status['data_source']=str(local)
  else:
   with urlopen(url,timeout=75) as resp:payload=resp.read()
   status['data_source']=url
  rr=list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig')),delimiter=';'))
  col='fixed acidity'; a=np.array([float(row[col]) for row in rr[:1024]],dtype='<f8')
  if len(a)!=1024:raise RuntimeError('Not exactly 1024 wine samples')
  buf=a.tobytes();status['input_sha256']=sha(buf);status['n_values']=1024;status['input_source_sha256']=sha(payload)
  # Native baseline header/payload sizes for matching input ID, but not equivalent ALP native BPV.
  import zlib,lzma
  (OUT/'generic_size_reference.csv').write_text('stream,n_values,sha256,mode,payload_bytes\nred_wine__fixed_acidity,1024,%s,zlib6,%s\nred_wine__fixed_acidity,1024,%s,lzma1,%s\n'%(sha(buf),len(zlib.compress(buf,6)),sha(buf),len(lzma.compress(buf,preset=1))))
  fbin=DATA/'red_wine__fixed_acidity.bin';fbin.write_bytes(buf)
  fcsv=DATA/'red_wine__fixed_acidity.csv';fcsv.write_text('\n'.join(format(float(v),'.17g') for v in a)+'\n')
  # Also provide ALP's documented .bin.f64 style because some loaders infer from suffix.
  fbinf64=DATA/'red_wine__fixed_acidity.bin.f64';fbinf64.write_bytes(buf)
  probe_specs=[('documented_binary',fbinf64,'binary'),('binary_suffix',fbin,'binary'),('csv_text',fcsv,'csv')]
  for label,path,ft in probe_specs:
   with cfg.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['id','column_name','data_type','path','file_type'])
    w.writeheader();w.writerow({'id':0,'column_name':label,'data_type':'double','path':str(path),'file_type':ft})
   (OUT/('config_'+label+'.csv')).write_bytes(cfg.read_bytes())
   # Prevent appending previous failed test output being misread as current success.
   t=time.perf_counter_ns();p=run([str(exe),'--gtest_filter=ALPBench.bench_alp_on_your_dataset'],cwd=BUILD,timeout=150)
   success=(p['returncode']==0 and '[  PASSED  ] 1 test' in p.get('stdout','')+'\n'+p.get('stderr',''))
   status['probe_log'].append({'probe':label,'returncode':p['returncode'],'gtest_success':success,'input':str(path),'format':ft,'ms':(time.perf_counter_ns()-t)/1e6})
   if success:
    status['stage']='ALP_NATIVE_BENCHMARK_PASSED';status['same_input_verified']=True
    break
  if status['stage']!='ALP_NATIVE_BENCHMARK_PASSED':status['stage']='OFFICIAL_ALP_DATA_PARSER_UNRESOLVED'
  # Search newest output CSV if any, retain native report and identifiable logs; no invented size equivalence.
  files=[]
  for f in BUILD.rglob('*.csv'):
   try:
    if f.is_file() and f.stat().st_size<3000000 and f.stat().st_mtime>time.time()-1800:
     files.append(f)
   except Exception:continue
  for j,f in enumerate(files[:25]):
   (OUT/('official_output_'+str(j)+'_'+f.name)).write_bytes(f.read_bytes())
  status['benchmark_result_files']=[str(f.relative_to(BUILD)) for f in files[:25]]
except Exception as exc:
 status['stage']='ERROR';status['error']=repr(exc);status['traceback']=traceback.format_exc()[-4000:]
finally:
 (OUT/'status.json').write_text(json.dumps(status,indent=2))
 print('NB59 stage:',status['stage'],'ALP native decoded bit-exact?',status['native_alp_decode_verified'])
 outzip()

