#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB49: Official ALP Input Probe

**Purpose:** Tests official ALP binary `.bin` and decimal `.csv` formats after the earlier `.f64` error; only counts official native results when an actual benchmark CSV is generated. No ALP full-container or bitwise decoder claim.

**Instructions:** Run **Runtime → Run all** in a fresh Google Colab CPU runtime. The notebook writes `SGX_NB49_results.zip` and attempts to download it. Each notebook is independent. If network access or ALP fails, status and logs are still exported rather than inventing numbers.

**Research integrity:** Bitwise round-trip success is necessary but not sufficient for novelty, competitive compression, or a Q1 publication.
"""

import os,sys,json,csv,io,hashlib,time,zipfile,statistics,zlib,lzma,platform,traceback,subprocess,shutil,struct,math
from pathlib import Path
import numpy as np
ROOT=Path('/content') if Path('/content').exists() else Path('/mnt/data')
OUT=ROOT/'SGX_NB49_outputs';OUT.mkdir(parents=True,exist_ok=True)
def write_csv(name,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def write_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=str))
def export():
 out=ROOT/'SGX_NB49_results.zip'
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(OUT.rglob('*')):
   if f.is_file():z.write(f,f.relative_to(OUT))
 print('RESULTS_ZIP:',out,'BYTES:',out.stat().st_size)
 try:
  from google.colab import files
  files.download(str(out))
 except ImportError:pass
 except Exception as e:print('Automatic download unavailable; download manually:',out,repr(e))
 return out
def run(cmd,timeout=600,cwd=None,env=None):
 t=time.monotonic()
 try:
  p=subprocess.run(cmd,cwd=cwd,env=env,capture_output=True,text=True,timeout=timeout)
  return {'command':cmd,'returncode':p.returncode,'duration_s':round(time.monotonic()-t,3),'stdout_tail':p.stdout[-24000:],'stderr_tail':p.stderr[-12000:]}
 except Exception as e:return {'command':cmd,'returncode':-999,'duration_s':round(time.monotonic()-t,3),'error':repr(e)}
def raw(v):return np.ascontiguousarray(v,dtype='<f8').tobytes()
def sha(b):return hashlib.sha256(b).hexdigest()
write_json('environment.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'smoke':os.environ.get('SGX_LOCAL_SMOKE_ONLY')=='1'})

def wine_streams():
 import urllib.request
 sources={'red':'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
          'white':'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'}
 for name,url in sources.items():
  text=urllib.request.urlopen(url,timeout=40).read().decode('utf-8')
  reader=csv.DictReader(io.StringIO(text),delimiter=';');rows=list(reader)
  usable=(len(rows)//1024)*1024
  if usable<1024:raise ValueError('Too few real samples '+name)
  for j,key in enumerate(reader.fieldnames):
   arr=np.array([float(r[key]) for r in rows[:usable]],dtype='<f8')
   yield {'stream':f'{name}_wine__c{j:02d}','dataset':name,'column':key,'values':arr,'n':len(arr),'url':url,'sha256':sha(raw(arr))}

PIN='31ca0ed11c93c99d3f5b5c30e01a3e1c3832d3ce'
REPO=ROOT/'SGX_ALP_NB49'
LOGS=[];status={'stage':'INIT','pinned_commit':PIN,'alp_native_bpv_only':True,'alp_bit_exact_decoding_verified':False,'alp_full_container_size_available':False,'file_probes':[],'full_benchmark_success':False}
try:
 if os.environ.get('SGX_LOCAL_SMOKE_ONLY')=='1':
  status['stage']='LOCAL_SMOKE_SKIP_EXTERNAL'
 else:
  data=list(wine_streams());status['real_streams']=len(data)
  input_dir=ROOT/'SGX_NB49_INPUT';input_dir.mkdir(parents=True,exist_ok=True)
  manifest=[]
  for item in data:
   b=raw(item['values']);p=input_dir/(item['stream']+'.bin');p.write_bytes(b)
   manifest.append({k:v for k,v in item.items() if k!='values'}|{'path':str(p),'bytes':len(b)})
  write_json('input_manifest.json',manifest)
  # Do not change input bits between ALP and generic baselines.
  generic=[]
  for item in data:
   b=raw(item['values'])
   for method,fun in [('zlib6',lambda q:zlib.compress(q,6)),('lzma1',lambda q:lzma.compress(q,preset=1))]:
    c=fun(b);assert (zlib.decompress(c) if method=='zlib6' else lzma.decompress(c))==b
    generic.append({'stream':item['stream'],'sha256':item['sha256'],'n':len(item['values']),'codec':method,'native_bytes':len(c),'native_bpv':len(c)*8/len(item['values']),'bit_exact':True})
  write_csv('generic_native_size_reference.csv',generic)
  logs_cmd=['bash','-lc','apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq clang clang-tidy cmake git ninja-build']
  LOGS.append(run(logs_cmd,timeout=900))
  if not (REPO/'.git').exists():LOGS.append(run(['git','clone','https://github.com/cwida/ALP.git',str(REPO)],timeout=240))
  if not (REPO/'.git').exists():raise RuntimeError('Clone failed; see logs')
  LOGS.append(run(['git','-C',str(REPO),'checkout','--detach',PIN],timeout=120))
  head=run(['git','-C',str(REPO),'rev-parse','HEAD']);LOGS.append(head)
  if head.get('stdout_tail','').strip()!=PIN:raise RuntimeError('Commit pin mismatch')
  config=REPO/'benchmarks'/'your_own_dataset.csv'
  if not config.parent.is_dir():raise RuntimeError('Dataset configuration path missing')
  cc=shutil.which('clang');cxx=shutil.which('clang++')
  if not cc or not cxx:raise RuntimeError('Clang not found')
  build=REPO/'sgx_nb49_build'
  LOGS.append(run(['cmake','-S',str(REPO),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DALP_BUILD_BENCHMARKING=ON','-DALP_ENABLE_CLANG_TIDY=OFF','-DCMAKE_C_COMPILER='+cc,'-DCMAKE_CXX_COMPILER='+cxx],timeout=450))
  if LOGS[-1]['returncode']!=0:raise RuntimeError('CMake configure failed')
  LOGS.append(run(['cmake','--build',str(build),'--target','bench_your_dataset','--parallel','2'],timeout=1200))
  if LOGS[-1]['returncode']!=0:raise RuntimeError('ALP target build failed')
  exe=build/'benchmarks'/'bench_your_dataset'
  if not exe.exists():raise RuntimeError('Official executable missing')
  # The earlier .f64/binary configuration failed. Test .bin/binary, then .csv/csv using one stream.
  probe=data[0]
  csv_path=input_dir/(probe['stream']+'.csv')
  with csv_path.open('w') as f:
   for x in probe['values']:f.write(repr(float(x))+'\n')
  def write_conf(rows):
   with config.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['id','column_name','data_type','path','file_type']);writer.writeheader();writer.writerows(rows)
   (OUT/'submitted_config_latest.csv').write_bytes(config.read_bytes())
  result_file=REPO/'benchmarks'/'your_own_dataset_result.csv'
  for fmt,path in [('binary_bin',Path(manifest[0]['path'])),('csv_csv',csv_path)]:
   write_conf([{'id':0,'column_name':probe['stream'],'data_type':'double','path':str(path),'file_type':'binary' if fmt=='binary_bin' else 'csv'}])
   if result_file.exists():result_file.unlink()
   r=run([str(exe)],timeout=420,cwd=build);LOGS.append(r)
   existed=result_file.exists() and result_file.stat().st_size>0
   status['file_probes'].append({'format':fmt,'path':str(path),'returncode':r['returncode'],'official_output_nonempty':existed,'log_tail':r.get('stdout_tail','')[-5000:]})
   if r['returncode']==0 and existed:status['selected_input_format']=fmt;break
  if status.get('selected_input_format'):
   fmt=status['selected_input_format'];allrows=[]
   for j,item in enumerate(data):
    if fmt=='csv_csv':
     p=input_dir/(item['stream']+'.csv')
     with p.open('w') as f:
      for x in item['values']:f.write(repr(float(x))+'\n')
    else:p=Path(manifest[j]['path'])
    allrows.append({'id':j,'column_name':item['stream'],'data_type':'double','path':str(p),'file_type':'csv' if fmt=='csv_csv' else 'binary'})
   write_conf(allrows)
   (OUT/'submitted_full_config.csv').write_bytes(config.read_bytes())
   if result_file.exists():result_file.unlink()
   r=run([str(exe)],timeout=1000,cwd=build);LOGS.append(r)
   status['full_benchmark_returncode']=r['returncode']
   status['full_benchmark_success']=r['returncode']==0 and result_file.exists() and result_file.stat().st_size>0
   if result_file.exists():(OUT/'official_alp_native_result.csv').write_bytes(result_file.read_bytes())
   status['stage']='ALP_NATIVE_RESULTS_AVAILABLE' if status['full_benchmark_success'] else 'FULL_BENCHMARK_FAILED'
  else:status['stage']='BOTH_INPUT_PROBES_FAILED'
except Exception as e:
 status['stage']='ERROR';status['error']=repr(e);status['traceback']=traceback.format_exc()[-6000:]
finally:
 write_json('build_and_benchmark_logs.json',LOGS);write_json('status.json',status)
 print('ALP STATUS',json.dumps(status,indent=2)[:4000]);export()

