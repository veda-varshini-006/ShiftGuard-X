#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX–NB19C — Official ALP Build Recovery and Benchmark

**Goal:** Recover the official [cwida/ALP](https://github.com/cwida/ALP) benchmark after NB19B timed out installing Clang. This notebook **does not rerun the earlier generic-codec comparisons**.

- Runs in Google Colab CPU. Run all cells in order (installation may require several minutes).
- Checks tools before attempting installation. Installs only what is missing, without a separate `apt-get update` step.
- Builds **only** the official `bench_your_dataset` target with 2 parallel jobs, avoiding a full build.
- Uses a real UCI numerical CSV when it can be downloaded; otherwise it records `REAL_DATA_UNAVAILABLE` and uses a clearly marked *synthetic smoke* dataset to check the executable. A synthetic smoke test **does not count** as a real-data benchmark.
- Records the exact git commit, build logs, executable status, dataset config, benchmark output, and detailed machine-readable status even on failure.
- Never treats compressed bits/value or cycles/value as directly comparable with compressed file bytes and end-to-end MiB/s.

Benchmark data specification follows the [official custom-dataset guide](https://github.com/cwida/ALP/blob/main/how_to_benchmark_your_dataset.md).
"""

# 00 — Controls. Use Colab CPU and a fresh runtime.
import pathlib
ROOT=pathlib.Path('/content') if pathlib.Path('/content').exists() else pathlib.Path('/mnt/data')
OUT=ROOT/'SGX_NB19C_outputs'; OUT.mkdir(parents=True, exist_ok=True)
ALP_REPO=ROOT/'SGX_NB19C_ALP'
INSTALL_MISSING_TOOLS=True
INSTALL_TIMEOUT_SECONDS=900
CLONE_TIMEOUT_SECONDS=240
CONFIGURE_TIMEOUT_SECONDS=180
BUILD_TIMEOUT_SECONDS=900
BENCHMARK_TIMEOUT_SECONDS=240
MAX_VALUES=32768
print('Output folder:',OUT)


# 01 — Status and robust command runner. Saves all logs as soon as commands finish or time out.
import os, sys, platform, shutil, subprocess, json, csv, time, hashlib, zipfile, urllib.request, io, re, traceback
import numpy as np
START=time.time(); STATUS=[]
def log(stage, state, detail=''):
    row={'stage':stage,'state':state,'detail':str(detail)[:4000],'utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    STATUS.append(row)
    (OUT/'stage_status.json').write_text(json.dumps(STATUS,indent=2))
    print(stage, state, str(detail)[:250])
def run(args, name, timeout=120, cwd=None, env=None):
    dest=OUT/(name+'.log'); begin=time.monotonic()
    try:
        with dest.open('w',encoding='utf-8',errors='replace') as fh:
            proc=subprocess.run(args, cwd=str(cwd) if cwd else None, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout, check=False, text=True)
        result={'ok':proc.returncode==0,'returncode':proc.returncode,'seconds':round(time.monotonic()-begin,2),'log':str(dest)}
    except subprocess.TimeoutExpired:
        result={'ok':False,'returncode':None,'seconds':round(time.monotonic()-begin,2),'error':'TIMEOUT','log':str(dest)}
        with dest.open('a') as fh:fh.write('\nERROR: COMMAND TIMED OUT\n')
    except Exception as e:
        result={'ok':False,'returncode':None,'seconds':round(time.monotonic()-begin,2),'error':repr(e),'log':str(dest)}
        with dest.open('a') as fh:fh.write('\nERROR: '+repr(e)+'\n')
    return result
ALP={'source':'https://github.com/cwida/ALP','commit':'','install':'NOT_RUN','clone':'NOT_RUN','configure':'NOT_RUN','build':'NOT_RUN','benchmark':'NOT_RUN','parsed_rows':0,'data_origin':'UNAVAILABLE'}
print(platform.platform(),platform.python_version())


# 02 — Tool discovery and minimal install. No apt-get update unless package metadata is unavailable.
needed=['git','cmake','clang','clang++']
missing=[x for x in needed if shutil.which(x) is None]
print('Available:',{x:shutil.which(x) for x in needed});print('Missing:',missing)
if not missing:
    ALP['install']='SKIPPED_TOOLS_PRESENT';log('install','SKIPPED','Existing tools used')
elif INSTALL_MISSING_TOOLS and shutil.which('apt-get'):
    packages=list(dict.fromkeys({'git':'git','cmake':'cmake','clang':'clang','clang++':'clang'}[x] for x in missing))
    env=os.environ.copy();env['DEBIAN_FRONTEND']='noninteractive'
    cmd=['apt-get','install','-y','-qq','--no-install-recommends']+packages
    result=run(cmd,'apt_install',INSTALL_TIMEOUT_SECONDS,env=env)
    if not result['ok']:
        # In some runtimes apt cache has not been initialized; retry with refresh only if needed.
        recent=(OUT/'apt_install.log').read_text(errors='replace')[-10000:]
        if ('Unable to locate package' in recent or '404 Not Found' in recent) and shutil.which('apt-get'):
            run(['apt-get','update','-qq'],'apt_update_fallback',INSTALL_TIMEOUT_SECONDS,env=env)
            result=run(cmd,'apt_install_retry',INSTALL_TIMEOUT_SECONDS,env=env)
    ALP['install']='COMPLETE' if all(shutil.which(x) for x in needed) else 'FAILED'
    log('install',ALP['install'],result)
else:
    ALP['install']='MISSING_TOOLS';log('install','MISSING_TOOLS',missing)
for x in needed:
    p=shutil.which(x)
    if p:
        run([p,'--version'],'tool_'+x.replace('+','p'),30)
print('Detected clang:',shutil.which('clang'),shutil.which('clang++'))


# 03 — Clone official source only if toolchain exists; retain commit ID.
if all(shutil.which(x) for x in needed):
    if (ALP_REPO/'.git').exists():
        log('clone','REUSED',str(ALP_REPO));ALP['clone']='REUSED'
    else:
        result=run(['git','clone','--depth','1','https://github.com/cwida/ALP.git',str(ALP_REPO)],'git_clone',CLONE_TIMEOUT_SECONDS)
        ALP['clone']='COMPLETE' if result['ok'] else 'FAILED';log('clone',ALP['clone'],result)
    if (ALP_REPO/'.git').exists():
        r=run(['git','rev-parse','HEAD'],'git_commit',30,cwd=ALP_REPO)
        ALP['commit']=(OUT/'git_commit.log').read_text().strip() if r['ok'] else ''
else:
    ALP['clone']='BLOCKED_TOOLCHAIN';log('clone','BLOCKED_TOOLCHAIN','Required tools not installed')
print('Repository commit:',ALP['commit'])


# 04 — Configure and compile just bench_your_dataset using Clang.
BUILD=ALP_REPO/'sgx_nb19c_build'
if ALP['commit']:
    BUILD.mkdir(exist_ok=True)
    args=['cmake','-S',str(ALP_REPO),'-B',str(BUILD),'-DCMAKE_BUILD_TYPE=Release','-DALP_BUILD_BENCHMARKING=ON','-DALP_ENABLE_VERBOSE_OUTPUT=ON','-DCMAKE_C_COMPILER='+shutil.which('clang'),'-DCMAKE_CXX_COMPILER='+shutil.which('clang++')]
    config=run(args,'alp_cmake_configure',CONFIGURE_TIMEOUT_SECONDS)
    ALP['configure']='COMPLETE' if config['ok'] else 'FAILED';log('configure',ALP['configure'],config)
    if config['ok']:
        build=run(['cmake','--build',str(BUILD),'--target','bench_your_dataset','--parallel','2'],'alp_cmake_build',BUILD_TIMEOUT_SECONDS)
        ALP['build']='COMPLETE' if build['ok'] else 'FAILED';log('build',ALP['build'],build)
else:
    log('configure','SKIPPED','No checked-out repo')
BIN=BUILD/'benchmarks'/'bench_your_dataset'
if ALP['build']=='COMPLETE' and not BIN.exists():
    found=[p for p in BUILD.rglob('bench_your_dataset') if p.is_file() and os.access(p,os.X_OK)]
    if found: BIN=found[0]
print('Official binary:', BIN, 'exists:', BIN.exists())


# 05 — Obtain a real UCI numerical series and create EXACT float64 benchmark bytes.
# Limited download, no dataframe dependencies. If UCI fails, synthetic fallback remains clearly labeled.
import zipfile as zf, math
UCI='https://archive.ics.uci.edu/static/public/360/air+quality.zip'
DATA=ROOT/'SGX_NB19C_dataset';DATA.mkdir(exist_ok=True)
raw_data=None;data_origin='REAL_UCI';error=''
try:
    req=urllib.request.Request(UCI,headers={'User-Agent':'SGX-Benchmark-Audit/1.0'})
    with urllib.request.urlopen(req,timeout=55) as response:
        raw_data=response.read(18_000_001)
    if len(raw_data)>18_000_000:raise RuntimeError('download exceeds safe cap')
    (OUT/'uci_download_sha256.txt').write_text(hashlib.sha256(raw_data).hexdigest())
    with zf.ZipFile(io.BytesIO(raw_data)) as archive:
        fname=next((p for p in archive.namelist() if p.lower().endswith('.csv')),None)
        if not fname:raise RuntimeError('No CSV found in archive')
        txt=archive.read(fname).decode('utf-8-sig',errors='replace')
    lines=txt.splitlines();reader=csv.DictReader(lines,delimiter=';')
    vals=[]
    for row in reader:
        k=next((key for key in row if key and key.strip()=='T'),None)
        if k is None:continue
        value=(row.get(k) or '').strip().replace(',','.')
        if not value:continue
        try:v=float(value)
        except ValueError:continue
        if not math.isfinite(v) or v==-200:continue
        vals.append(v)
        if len(vals)>=MAX_VALUES:break
    if len(vals)<1024:raise RuntimeError('Too few valid temperature observations: '+str(len(vals)))
    values=np.array(vals,dtype='<f8')
except Exception as e:
    data_origin='SYNTHETIC_SMOKE_ONLY';error=repr(e)
    rng=np.random.default_rng(19);values=(20+np.cumsum(rng.normal(0,0.05,8192))).astype('<f8')
    log('real_data','FAILED_USING_SYNTHETIC_SMOKE',error)
else:log('real_data','COMPLETE','UCI Air Quality T: '+str(len(values))+' values')
ALP['data_origin']=data_origin
BIN_DATA=DATA/'sgx_temperature_f64.bin';BIN_DATA.write_bytes(values.tobytes())
(DATA/'data_manifest.json').write_text(json.dumps({'origin':data_origin,'url':UCI if data_origin=='REAL_UCI' else None,'series':'T' if data_origin=='REAL_UCI' else 'random_walk','dtype':'little-endian float64','num_values':int(values.size),'sha256':hashlib.sha256(BIN_DATA.read_bytes()).hexdigest(),'source_error':error},indent=2))
shutil.copy2(DATA/'data_manifest.json', OUT/'benchmark_dataset_manifest.json')
print('Origin:',data_origin,'values:',len(values),'bytes:',BIN_DATA.stat().st_size)


# 06 — Install dataset configuration in repository via grounded path discovery.
# Upstream guide requires: id,column_name,data_type,path,file_type
# Find the config path from the checked-out source. If unknown, do not guess or claim benchmark completeness.
config_candidates=[]
if ALP['build']=='COMPLETE':
    docs=list(ALP_REPO.glob('*benchmark*dataset*.md'))
    for p in docs: shutil.copy2(p,OUT/('guide_'+p.name))
    for p in (ALP_REPO/'benchmarks').rglob('*.cpp'):
        if 'your_dataset' not in p.name:continue
        text=p.read_text(errors='replace');(OUT/'inspected_bench_source.txt').write_text(text[:150000])
        for s in re.findall(r'[\"\']([^\"\']+\.csv)[\"\']',text):
            if any(w in s.lower() for w in ['your_own_dataset','your_dataset']):config_candidates.append(s)
    if not config_candidates:
        # Search the entire matching benchmark source for the full CSV filename.
        config_candidates=['your_own_dataset.csv']
    (OUT/'config_path_candidates.json').write_text(json.dumps(config_candidates,indent=2))
    # Set only where source explicitly references relative path OR a matching existing config exists.
    candidates=[]
    for path in ALP_REPO.rglob('your_own_dataset.csv'): candidates.append(path)
    for token in config_candidates:
        p=pathlib.Path(token)
        if p.is_absolute():
            if p.exists() and str(p).startswith(str(ALP_REPO)):candidates.append(p)
        else:
            for basedir in (ALP_REPO,ALP_REPO/'benchmarks',BUILD,BUILD/'benchmarks'):
                q=(basedir/p).resolve()
                if q.exists() and str(q).startswith(str(ALP_REPO)):candidates.append(q)
    candidates=list(dict.fromkeys(candidates))
    print('Found candidate configs:',candidates)
    if len(candidates)==1:
        TARGET_CONFIG=candidates[0]
        # Copy original configuration to results before overwriting.
        shutil.copy2(TARGET_CONFIG,OUT/'upstream_dataset_config_before.csv')
        with TARGET_CONFIG.open('w',newline='') as fh:
            w=csv.writer(fh);w.writerow(['id','column_name','data_type','path','file_type']);w.writerow([0,'SGX_UCI_T_f64' if data_origin=='REAL_UCI' else 'SGX_SYNTHETIC_SMOKE_f64','double',str(BIN_DATA.resolve()),'binary'])
        shutil.copy2(TARGET_CONFIG,OUT/'benchmark_dataset_config.csv')
        ALP['dataset_config']=str(TARGET_CONFIG);log('config','COMPLETE',TARGET_CONFIG)
    else:
        ALP['dataset_config']='';log('config','NEEDS_PATH_REVIEW','Matching config candidates '+str(candidates))
else:log('config','SKIPPED','ALP binary was not built')


# 07 — Execute ONLY the official binary. Capture its stdout and resulting CSV artifacts.
if ALP['build']=='COMPLETE' and ALP.get('dataset_config') and BIN.exists():
    before={str(p):p.stat().st_mtime_ns for p in ALP_REPO.rglob('*.csv')}
    result=run([str(BIN)],'alp_official_benchmark',BENCHMARK_TIMEOUT_SECONDS,cwd=BUILD)
    ALP['benchmark']='EXECUTED_SUCCESS' if result['ok'] else 'EXECUTION_FAILED'
    log('benchmark',ALP['benchmark'],result)
    changed=[]
    for p in ALP_REPO.rglob('*.csv'):
        try:
            if str(p) not in before or p.stat().st_mtime_ns>before[str(p)]:
                if p.resolve()!=pathlib.Path(ALP['dataset_config']).resolve():changed.append(p)
        except Exception:pass
    for i,p in enumerate(changed[:25]):
        dest=OUT/('upstream_result_%02d.csv'%i);shutil.copy2(p,dest)
    ALP['native_csv_files']=[p.name for p in OUT.glob('upstream_result_*.csv')]
else:
    ALP['benchmark']='NOT_RUN';log('benchmark','NOT_RUN','Requires successful build and unique grounded dataset config')
print('Benchmark status:',ALP['benchmark'])


# 08 — Honest completeness audit; parse available official output (including stdout).
results=[]
for p in OUT.glob('upstream_result_*.csv'):
    try:
        with p.open(newline='',errors='replace') as fh:
            reader=csv.DictReader(fh)
            for r in reader:results.append({'file':p.name,**r})
    except Exception as e:log('parse_csv','FAILED',repr(e))
if results:
    with (OUT/'official_parsed_rows.csv').open('w',newline='') as fh:
        keys=sorted(set().union(*(r.keys() for r in results)));w=csv.DictWriter(fh,fieldnames=keys);w.writeheader();w.writerows(results)
ALP['parsed_rows']=len(results)
ALP['real_data_benchmark_complete']=(ALP['benchmark']=='EXECUTED_SUCCESS' and ALP['data_origin']=='REAL_UCI' and len(results)>0)
ALP['interpretation']='Official native ALP size may be bits/value, and throughput may be cycles/value. Do NOT compare to payload bytes without matched conventions.'
(OUT/'official_alp_status.json').write_text(json.dumps(ALP,indent=2))
(OUT/'environment.json').write_text(json.dumps({'platform':platform.platform(),'python':platform.python_version(),'start_timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'elapsed_seconds':round(time.time()-START,2)},indent=2))
log('research_gate','COMPLETE' if ALP['real_data_benchmark_complete'] else 'INCOMPLETE',ALP)
print(json.dumps(ALP,indent=2))


# 09 — Always export ZIP, even if tool installation / build / benchmark fails.
manifest=[]
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name!='sha256_manifest.json':
        manifest.append({'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2))
ZIP=ROOT/'SGX_NB19C_Official_ALP_results.zip'
with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():
        if p.is_file():z.write(p,arcname=p.name)
print('RESULTS ZIP:',ZIP,'size:',ZIP.stat().st_size,'bytes')
if ROOT==pathlib.Path('/content'):
    from google.colab import files
    files.download(str(ZIP))

