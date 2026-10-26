#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX–NB19E — Official ALP Dataset-Format Repair

This standalone **Google Colab CPU** notebook recovers the official ALP `bench_your_dataset` benchmark, retaining the working NB19D Clang build and improving dataset/config diagnostics.

**Evidence from NB19D:** the official executable compiled successfully but rejected the dataset with `No bin or csv file specified`. The code now inspects the checked-out ALP source, validates the actual binary payload, and tries limited documented file-type tokens only after inspecting the source. Official benchmark results are marked complete **only** if the executable succeeds and produces interpretable numeric rows. Synthetic fallback is labeled smoke-only.

Run **Runtime → Run all**; final cell downloads `SGX_NB19E_Official_ALP_results.zip`. No fabricated ALP results.
"""

# 00 — Paths and conservative resource limits
import pathlib
ROOT=pathlib.Path('/content') if pathlib.Path('/content').exists() else pathlib.Path('/mnt/data')
OUT=ROOT/'SGX_NB19E_outputs';OUT.mkdir(parents=True,exist_ok=True)
ALP_REPO=ROOT/'SGX_NB19E_ALP'
INSTALL_MISSING_TOOLS=True
INSTALL_TIMEOUT_SECONDS=900
CLONE_TIMEOUT_SECONDS=240
CONFIGURE_TIMEOUT_SECONDS=240
BUILD_TIMEOUT_SECONDS=1200
BENCHMARK_TIMEOUT_SECONDS=180
MAX_VALUES=32768
print('Artifacts:',OUT)


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



# 04 — Build the unmodified ALP benchmark with Clang. Prior successful approach.
BUILD=ALP_REPO/'sgx_nb19e_build'
ALP['build']='NOT_RUN'
if ALP['commit']:
    BUILD.mkdir(exist_ok=True)
    args=['cmake','-S',str(ALP_REPO),'-B',str(BUILD),'-DCMAKE_BUILD_TYPE=Release',
          '-DALP_BUILD_BENCHMARKING=ON','-DALP_ENABLE_CLANG_TIDY=OFF',
          '-DALP_ENABLE_VERBOSE_OUTPUT=ON',
          '-DCMAKE_C_COMPILER='+shutil.which('clang'),
          '-DCMAKE_CXX_COMPILER='+shutil.which('clang++')]
    cfg=run(args,'alp_cmake_configure',CONFIGURE_TIMEOUT_SECONDS)
    ALP['configure']='COMPLETE' if cfg['ok'] else 'FAILED';log('configure',ALP['configure'],cfg)
    if cfg['ok']:
        built=run(['cmake','--build',str(BUILD),'--target','bench_your_dataset','--parallel','2'],
                  'alp_cmake_build',BUILD_TIMEOUT_SECONDS)
        ALP['build']='COMPLETE' if built['ok'] else 'FAILED';log('build',ALP['build'],built)
else:log('configure','SKIPPED','No cloned repository')
BIN=BUILD/'benchmarks'/'bench_your_dataset'
if ALP['build']=='COMPLETE' and not BIN.is_file():
    matches=[p for p in BUILD.rglob('bench_your_dataset') if p.is_file() and os.access(p,os.X_OK)]
    if matches:BIN=matches[0]
print('Binary:',BIN,'exists:',BIN.is_file())


# 05 — Obtain real UCI temperature values and construct canonical IEEE-754 float64 binary
import zipfile as zf, math
UCI='https://archive.ics.uci.edu/static/public/360/air+quality.zip'
DATA=ROOT/'SGX_NB19E_dataset';DATA.mkdir(parents=True,exist_ok=True)
data_origin='REAL_UCI';source_error=''
try:
    req=urllib.request.Request(UCI,headers={'User-Agent':'SGX-ALP-Evidence/1.0'})
    with urllib.request.urlopen(req,timeout=75) as response:raw=response.read(18_000_001)
    if len(raw)>18_000_000:raise RuntimeError('Source download exceeds cap')
    (OUT/'source_download_sha256.txt').write_text(hashlib.sha256(raw).hexdigest())
    with zf.ZipFile(io.BytesIO(raw)) as archive:
        fname=next((p for p in archive.namelist() if p.lower().endswith('.csv')),None)
        if not fname:raise RuntimeError('Dataset zip contains no CSV')
        text=archive.read(fname).decode('utf-8-sig',errors='replace')
    vals=[]
    for row in csv.DictReader(text.splitlines(),delimiter=';'):
        k=next((key for key in row if key and key.strip()=='T'),None)
        if k is None:continue
        s=(row.get(k) or '').strip().replace(',','.')
        if not s:continue
        try:v=float(s)
        except ValueError:continue
        if math.isfinite(v) and v!=-200: vals.append(v)
        if len(vals)>=MAX_VALUES:break
    if len(vals)<1024:raise RuntimeError('Too few valid UCI temperature values')
    values=np.array(vals,dtype='<f8')
except Exception as exc:
    source_error=repr(exc);data_origin='SYNTHETIC_SMOKE_ONLY'
    rng=np.random.default_rng(1905)
    values=(20+np.cumsum(rng.normal(0,.05,8192))).astype('<f8')
    log('data','SYNTHETIC_FALLBACK',source_error)
else:log('data','REAL_UCI',f'{values.size} temperature values')
ALP['data_origin']=data_origin
DATAFILE=DATA/'sgx_temperature_f64.bin'
DATAFILE.write_bytes(values.tobytes(order='C'))
assert DATAFILE.is_file() and DATAFILE.stat().st_size==values.size*8
assert np.array_equal(np.frombuffer(DATAFILE.read_bytes(),dtype='<f8').view('<u8'),values.view('<u8'))
manifest={'source_origin':data_origin,'uci_url':UCI if data_origin=='REAL_UCI' else None,
 'source_error':source_error,'dtype':'little-endian IEEE-754 float64','n_values':int(values.size),
 'byte_count':DATAFILE.stat().st_size,'path':str(DATAFILE.resolve()),
 'sha256':hashlib.sha256(DATAFILE.read_bytes()).hexdigest()}
(OUT/'benchmark_dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))


# 06 — Inspect ALP's exact file type handling and validate the existing configuration location.
# NB19D reached ALP's parser, but threw "No bin or csv file specified".
# Never patch the official ALP C++ source. Diagnose and modify ONLY the dataset CSV.
ALP['dataset_config']=''
file_type_candidates=['binary','bin','csv']
if ALP['build']=='COMPLETE':
    guide=ALP_REPO/'how_to_benchmark_your_dataset.md'
    if guide.is_file():shutil.copy2(guide,OUT/'guide_how_to_benchmark_your_dataset.md')
    configs=list(ALP_REPO.rglob('your_own_dataset.csv'))
    (OUT/'config_path_candidates.json').write_text(json.dumps([str(p) for p in configs],indent=2))
    # Choose the unique official config. Read-only source inspection is archived.
    if len(configs)==1:
        CONFIG=configs[0]
        shutil.copy2(CONFIG,OUT/'upstream_dataset_config_before.csv')
        ALP['dataset_config']=str(CONFIG)
        # Locate the error string in actual C++ source, not in compiled dependency trees.
        hits=[]
        for source in ALP_REPO.rglob('*'):
            if not source.is_file() or source.suffix not in ('.cpp','.cc','.hpp','.h'):continue
            if any(part in {'sgx_nb19e_build','.git'} for part in source.parts):continue
            try:txt=source.read_text(errors='replace')
            except Exception:continue
            for term in ['No bin or csv file specified','file_type ==','file_type==']:
                at=txt.find(term)
                if at>=0:
                    hits.append({'path':str(source.relative_to(ALP_REPO)),'term':term,
                      'excerpt':txt[max(0,at-750):at+750]})
        (OUT/'source_format_inspection.json').write_text(json.dumps(hits[:30],indent=2))
        # Most implementations check for 'bin' (not 'binary') or suffix '.bin'.
        # Candidate attempts are explicitly reported and only observed successes count.
        log('config','READY','Unique official dataset config: '+str(CONFIG))
    else:log('config','BLOCKED',f'Expected unique config; found {len(configs)}')
else:log('config','BLOCKED','Official binary unavailable')


# 07 — Controlled official ALP format probes (bin / binary / csv), same original numeric values.
# Each attempt writes a valid dataset file and official CSV; never changes ALP source.
# Full console output and each official result CSV are preserved.
ALP['benchmark']='NOT_RUN'
ALP['attempts']=[]
if ALP['build']=='COMPLETE' and ALP['dataset_config'] and BIN.is_file():
    # Rebuild this text CSV only as a compatible fallback format (lossless decimal repr),
    # and still prefer binary input for byte-exact raw representation.
    CSV_DATA=DATA/'sgx_temperature_f64.csv'
    CSV_DATA.write_text('value\n'+''.join(format(float(v),'.17g')+'\n' for v in values))
    candidates=[('bin',DATAFILE),('binary',DATAFILE),('csv',CSV_DATA)]
    for i,(ftype,source_path) in enumerate(candidates):
        # Validate preconditions and actual path before ALP invocation.
        assert source_path.resolve().is_file() and source_path.stat().st_size>0
        with pathlib.Path(ALP['dataset_config']).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['id','column_name','data_type','path','file_type'])
            writer.writerow([0,'SGX_UCI_T_f64' if data_origin=='REAL_UCI' else 'SGX_SYNTHETIC_f64',
                             'double',str(source_path.resolve()),ftype])
        shutil.copy2(ALP['dataset_config'],OUT/f'benchmark_config_attempt_{i}_{ftype}.csv')
        # Remove old native result to prevent accidentally accepting stale output.
        official_result=ALP_REPO/'benchmarks'/'your_own_dataset_result.csv'
        if official_result.is_file():official_result.unlink()
        before=time.time()
        r=run([str(BIN)],f'official_alp_attempt_{i}_{ftype}',BENCHMARK_TIMEOUT_SECONDS,cwd=BUILD)
        record={'candidate':ftype,'data_path':str(source_path.resolve()),'data_bytes':source_path.stat().st_size,
                'returncode':r['returncode'],'ok':r['ok'],'seconds':r['seconds'],
                'official_result_exists':official_result.is_file(),'rows':0}
        if official_result.is_file():
            dest=OUT/f'official_result_attempt_{i}_{ftype}.csv';shutil.copy2(official_result,dest)
            try:
                with dest.open(newline='') as f:
                    recs=list(csv.DictReader(f))
                record['rows']=len(recs)
                record['columns']=list(recs[0]) if recs else []
            except Exception as exc:record['parse_error']=repr(exc)
        ALP['attempts'].append(record)
        (OUT/'benchmark_attempts.json').write_text(json.dumps(ALP['attempts'],indent=2))
        log('benchmark_attempt', 'SUCCESS' if record['ok'] and record['rows'] else 'INCOMPLETE',record)
        # Stop only after a confirmed native row and a successful exit.
        if record['ok'] and record['rows']:
            ALP['benchmark']='EXECUTED_SUCCESS';ALP['successful_file_type']=ftype
            break
    if ALP['benchmark']!='EXECUTED_SUCCESS': ALP['benchmark']='EXECUTION_FAILED'
else:log('benchmark','NOT_RUN','Requires successful build and unique dataset config')
print('ALP benchmark status:',ALP['benchmark'])


# 08 — Evidence audit. Never claim official completion from an empty / stale CSV.
all_rows=[]
for p in sorted(OUT.glob('official_result_attempt_*.csv')):
    try:
        with p.open(newline='',errors='replace') as f:
            for r in csv.DictReader(f):all_rows.append({'source_file':p.name,**r})
    except Exception as exc:log('parse','FAILED',repr(exc))
if all_rows:
    keys=sorted(set().union(*(r.keys() for r in all_rows)))
    with (OUT/'official_alp_parsed_rows.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(all_rows)
ALP['parsed_rows']=len(all_rows)
ALP['real_data_benchmark_complete']=bool(ALP['benchmark']=='EXECUTED_SUCCESS' and
                                      ALP['data_origin']=='REAL_UCI' and len(all_rows)>0)
ALP['caution']='Native ALP size is bits/value and timing can be cycles/value; do not compare these directly with bytes or MiB/s.'
(OUT/'official_alp_status.json').write_text(json.dumps(ALP,indent=2))
(OUT/'environment.json').write_text(json.dumps({'python':platform.python_version(),
   'platform':platform.platform(),'elapsed_seconds':round(time.time()-START,2)},indent=2))
log('research_gate','COMPLETE' if ALP['real_data_benchmark_complete'] else 'INCOMPLETE',
    {'benchmark':ALP['benchmark'],'rows':ALP['parsed_rows'],'real_data':ALP['data_origin']})
print(json.dumps(ALP,indent=2))


# 09 — Always create a portable diagnostic ZIP.
manifest=[]
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name!='sha256_manifest.json':
        manifest.append({'file':p.name,'bytes':p.stat().st_size,
                        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2))
ZIP=ROOT/'SGX_NB19E_Official_ALP_results.zip'
with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file():z.write(p,p.name)
print('Created',ZIP,'bytes',ZIP.stat().st_size)
try:
    from google.colab import files
    files.download(str(ZIP))
except ImportError:
    print('Outside Colab. ZIP is available at',ZIP)

