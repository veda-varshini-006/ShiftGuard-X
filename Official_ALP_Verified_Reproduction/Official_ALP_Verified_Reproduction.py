#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX-NB32 — Official ALP Verified Reproduction and Generic-Codec Accounting

Replays the successful official NB19F ALP run on UCI temperature and records independently measured generic codec sizes on the same IEEE-754 binary data. Native ALP BPV and generic file BPV are **not** asserted to have identical accounting. It is a first step, not a complete matched specialized evaluation. Requires Colab CPU with outbound GitHub/UCI access.
"""

# 00 — Paths and conservative resource limits
import pathlib
ROOT=pathlib.Path('/content') if pathlib.Path('/content').exists() else pathlib.Path('/mnt/data')
OUT=ROOT/'SGX_NB32_outputs';OUT.mkdir(parents=True,exist_ok=True)
ALP_REPO=ROOT/'SGX_NB32_ALP'
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
BUILD=ALP_REPO/'sgx_nb32_build'
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
DATA=ROOT/'SGX_NB32_dataset';DATA.mkdir(parents=True,exist_ok=True)
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


# 06 — Verify official CSV schema, parser behaviour, and configuration location
ALP['dataset_config']=''
if ALP['build']=='COMPLETE':
    guide=ALP_REPO/'how_to_benchmark_your_dataset.md'
    if guide.is_file(): shutil.copy2(guide,OUT/'guide_how_to_benchmark_your_dataset.md')
    configs=[p for p in ALP_REPO.rglob('your_own_dataset.csv') if 'build' not in str(p.relative_to(ALP_REPO)).lower()]
    (OUT/'config_path_candidates.json').write_text(json.dumps([str(p) for p in configs],indent=2))
    if len(configs)==1:
        CONFIG=configs[0]
        shutil.copy2(CONFIG,OUT/'upstream_dataset_config_before.csv')
        ALP['dataset_config']=str(CONFIG)
        hits=[]
        for source in ALP_REPO.rglob('*'):
            if not source.is_file() or source.suffix not in ('.cpp','.cc','.hpp','.h'): continue
            if '.git' in source.parts or 'sgx_nb32_build' in source.parts: continue
            try: txt=source.read_text(errors='replace')
            except Exception: continue
            for term in ['No bin or csv file specified','FileType::BINARY','FileType::CSV','file_type']:
                at=txt.find(term)
                if at>=0: hits.append({'path':str(source.relative_to(ALP_REPO)),'term':term,'excerpt':txt[max(0,at-650):at+650]})
        (OUT/'source_format_inspection.json').write_text(json.dumps(hits[:35],indent=2))
        log('config','READY',str(CONFIG))
    else: log('config','BLOCKED',f'Expected one official config; got {len(configs)}')
else: log('config','BLOCKED','Official binary unavailable')


# 07 — Write Unix-LF configs, validate actual bytes, then benchmark official ALP
ALP['benchmark']='NOT_RUN'
ALP['attempts']=[]
if ALP['build']=='COMPLETE' and ALP['dataset_config'] and BIN.is_file():
    # The data CSV is ONE numeric value per line, without a header (ALP source contract).
    CSV_DATA=DATA/'sgx_temperature_f64.csv'
    CSV_DATA.write_bytes((''.join(format(float(v),'.17g')+'\n' for v in values)).encode('utf-8'))
    assert b'\r' not in CSV_DATA.read_bytes()
    candidates=[('binary',DATAFILE),('csv',CSV_DATA)]
    official_result=ALP_REPO/'benchmarks'/'your_own_dataset_result.csv'
    for i,(ftype,source_path) in enumerate(candidates):
        assert source_path.is_file() and source_path.stat().st_size>0
        column='SGX_UCI_T_f64' if data_origin=='REAL_UCI' else 'SGX_SYNTHETIC_f64'
        # Deliberately avoid Python csv.writer default CRLF: this caused NB19E failures.
        fields=['0',column,'double',str(source_path.resolve()),ftype]
        if any(',' in x or '\n' in x or '\r' in x for x in fields): raise ValueError('Unsafe CSV field')
        config_bytes=('id,column_name,data_type,path,file_type\n'+','.join(fields)+'\n').encode('utf-8')
        CONFIG.write_bytes(config_bytes)
        assert b'\r' not in CONFIG.read_bytes(), 'CR character in config!'
        with CONFIG.open('r',newline='') as f: parsed=list(csv.DictReader(f))
        assert len(parsed)==1 and parsed[0]['file_type']==ftype
        assert pathlib.Path(parsed[0]['path']).is_file()
        shutil.copy2(CONFIG,OUT/f'benchmark_config_attempt_{i}_{ftype}.csv')
        if official_result.exists(): official_result.unlink()
        r=run([str(BIN)],f'official_alp_attempt_{i}_{ftype}',BENCHMARK_TIMEOUT_SECONDS,cwd=BUILD)
        record={'candidate':ftype,'data_path':str(source_path.resolve()),'data_bytes':source_path.stat().st_size,'config_has_cr':b'\r' in config_bytes,
                'returncode':r['returncode'],'ok':r['ok'],'seconds':r['seconds'],
                'official_result_exists':official_result.is_file(),'rows':0}
        if official_result.is_file():
            dest=OUT/f'official_result_attempt_{i}_{ftype}.csv';shutil.copy2(official_result,dest)
            try:
                with dest.open(newline='') as f: recs=list(csv.DictReader(f))
                record['rows']=len(recs); record['columns']=list(recs[0]) if recs else []
                record['sample_row']=recs[0] if recs else None
            except Exception as exc: record['parse_error']=repr(exc)
        ALP['attempts'].append(record)
        (OUT/'benchmark_attempts.json').write_text(json.dumps(ALP['attempts'],indent=2))
        log('benchmark_attempt','SUCCESS' if record['ok'] and record['rows'] else 'INCOMPLETE',record)
        if record['ok'] and record['rows']:
            ALP['benchmark']='EXECUTED_SUCCESS';ALP['successful_file_type']=ftype
            break
    if ALP['benchmark']!='EXECUTED_SUCCESS': ALP['benchmark']='EXECUTION_FAILED'
else: log('benchmark','NOT_RUN','Requires successful build and unique dataset config')
print('ALP official benchmark status:',ALP['benchmark'])


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


# Supplementary native generic formats on the SAME raw binary file used by ALP.
# File name autodiscovery avoids silently processing an unrelated dataset.
report={'generic':'NOT_RUN','fairness_warning':'ALP native bench BPV may exclude metadata accounted by generic library native streams. Do not infer a fair win from comparing these directly.'}
try:
 import lzma,zlib
 candidates=list((ROOT/'SGX_NB32_dataset').glob('*.bin'))
 if len(candidates)!=1:raise RuntimeError(f'Expected one binary input, found: {candidates}')
 raw=candidates[0].read_bytes()
 assert len(raw)%8==0
 rr=[]
 for method in ['zlib','lzma']:
  packed=zlib.compress(raw,6) if method=='zlib' else lzma.compress(raw,preset=1)
  recovered=zlib.decompress(packed) if method=='zlib' else lzma.decompress(packed)
  assert recovered==raw
  rr.append({'method':method,'n_values':len(raw)//8,'bytes':len(packed),'bpv_native':8*len(packed)/(len(raw)//8),'bit_exact':True})
 with (OUT/'generic_codec_accounting.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
 report['generic']='COMPLETE';report['records']=rr
except Exception as e:
 report['error']=repr(e)
(OUT/'matched_accounting_note.json').write_text(json.dumps(report,indent=2))
print(report)


# 09 — Always create a portable diagnostic ZIP.
manifest=[]
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name!='sha256_manifest.json':
        manifest.append({'file':p.name,'bytes':p.stat().st_size,
                        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2))
ZIP=ROOT/'SGX_NB32_Official_ALP_results.zip'
with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file():z.write(p,p.name)
print('Created',ZIP,'bytes',ZIP.stat().st_size)
try:
    from google.colab import files
    files.download(str(ZIP))
except ImportError:
    print('Outside Colab. ZIP is available at',ZIP)

