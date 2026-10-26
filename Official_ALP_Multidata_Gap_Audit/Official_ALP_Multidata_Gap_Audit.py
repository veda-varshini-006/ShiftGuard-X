#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX-NB20 — Official ALP Multi-Dataset Compression Gap Audit

**Purpose:** Extend the successful NB19F official ALP benchmark to six real numerical columns (UCI Air Quality and UCI Household Power), compare common compressors on the **same serialized IEEE-754 float64 values**, and produce a diagnostic compression-gap report. **This is a benchmark, not a claimed new codec.**

**Expected ZIP:** `SGX_NB20_Official_ALP_Multidata_results.zip`. Dataset downloads and the ALP build require internet access in Colab. Missing datasets or unsuccessful official ALP runs are explicitly marked; synthetic controls never count as real evidence. Run **Runtime → Run all** on CPU.
"""

# 00 — Standalone setup, bounded resource settings
import os, sys, subprocess, pathlib, json, csv, io, zipfile, time, hashlib, shutil, platform, urllib.request, re, traceback, math
import numpy as np
ROOT=pathlib.Path('/content') if pathlib.Path('/content').is_dir() else pathlib.Path('/mnt/data')
OUT=ROOT/'SGX_NB20_outputs'; OUT.mkdir(parents=True,exist_ok=True)
REPO=ROOT/'SGX_NB20_ALP'; BUILD=REPO/'sgx_build'; DATADIR=ROOT/'SGX_NB20_datasets'; DATADIR.mkdir(exist_ok=True)
ALP_COMMIT='31ca0ed11c93c99d3f5b5c30e01a3e1c3832d3ce'
MAX_VALUES=32768
INSTALL_TIMEOUT=900; BUILD_TIMEOUT=1200; BENCH_TIMEOUT=300
START=time.time(); EVENTS=[]
def savejson(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=str))
def savecsv(name,rows):
    if not rows: (OUT/name).write_text('');return
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def stage(name,status,detail=''):
    EVENTS.append({'stage':name,'status':status,'detail':str(detail)[:2000]});savejson('stage_status.json',EVENTS)
    print(name,status,str(detail)[:150])
def run(cmd,label,timeout=120,cwd=None,env=None):
    f=OUT/(label+'.log');t=time.monotonic()
    try:
        with f.open('w') as h: p=subprocess.run(cmd,cwd=str(cwd) if cwd else None,env=env,stdout=h,stderr=subprocess.STDOUT,timeout=timeout,check=False)
        d={'ok':p.returncode==0,'returncode':p.returncode,'seconds':round(time.monotonic()-t,2),'log':f.name}
    except Exception as e:
        d={'ok':False,'error':repr(e),'seconds':round(time.monotonic()-t,2),'log':f.name}
        with f.open('a') as h:h.write('\nERROR: '+repr(e))
    stage(label,'PASS' if d['ok'] else 'FAIL',d);return d
print('Results folder:',OUT)


# 01 — Recover official ALP benchmark environment (known good NB19F flags)
TOOLS=['git','cmake','clang','clang++']
missing=[t for t in TOOLS if not shutil.which(t)]
if missing:
    run(['apt-get','update','-qq'],'apt_update',INSTALL_TIMEOUT)
    run(['apt-get','install','-y','-qq','clang','cmake','ninja-build'],'apt_install',INSTALL_TIMEOUT)
tools_ready=all(shutil.which(t) for t in TOOLS)
if not tools_ready:
    stage('tools','BLOCKED',f'Missing { [t for t in TOOLS if not shutil.which(t)] }')
else:stage('tools','PASS','Clang and CMake available')
if tools_ready:
    if not (REPO/'.git').is_dir():run(['git','clone','--depth','1','https://github.com/cwida/ALP.git',str(REPO)],'clone',240)
    if (REPO/'.git').is_dir():
        # NB19F exact known-good commit for reproducibility; fetch if shallow clone differs
        actual=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True).stdout.strip()
        if actual!=ALP_COMMIT:
            run(['git','fetch','--depth','1','origin',ALP_COMMIT],'pin_fetch',240,cwd=REPO)
            run(['git','checkout','--detach',ALP_COMMIT],'pin_checkout',120,cwd=REPO)
commit=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True).stdout.strip() if (REPO/'.git').is_dir() else ''
repo_ok=bool(commit==ALP_COMMIT)
if repo_ok:
    cfg=run(['cmake','-S',str(REPO),'-B',str(BUILD),'-DCMAKE_BUILD_TYPE=Release',
             '-DALP_BUILD_BENCHMARKING=ON','-DALP_ENABLE_CLANG_TIDY=OFF',
             '-DALP_ENABLE_VERBOSE_OUTPUT=ON','-DCMAKE_C_COMPILER='+shutil.which('clang'),
             '-DCMAKE_CXX_COMPILER='+shutil.which('clang++')], 'alp_configure',240)
    built=run(['cmake','--build',str(BUILD),'--target','bench_your_dataset','--parallel','2'],
              'alp_build',BUILD_TIMEOUT) if cfg['ok'] else {'ok':False}
else:
    stage('alp_source','BLOCKED','Pinned commit unavailable');cfg={'ok':False};built={'ok':False}
BIN=BUILD/'benchmarks'/'bench_your_dataset'
if built['ok'] and not BIN.is_file():
    matches=[x for x in BUILD.rglob('bench_your_dataset') if x.is_file() and os.access(x,os.X_OK)]
    if matches:BIN=matches[0]
binary_ok=bool(built['ok'] and BIN.is_file())
savejson('official_build_status.json',{'commit':commit,'expected_commit':ALP_COMMIT,'repo_ok':repo_ok,'configured':cfg['ok'],'built':built['ok'],'benchmark_executable':str(BIN),'executable_exists':BIN.is_file()})
print('ALP executable ready:',binary_ok)


# 02 — Download UCI datasets; keep six real series, without synthetic substitution
SOURCES=[('air','https://archive.ics.uci.edu/static/public/360/air+quality.zip',['T','RH','CO(GT)']),
         ('house','https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip',['Global_active_power','Voltage','Global_intensity'])]
def download_capped(url,limit=35_000_000):
    req=urllib.request.Request(url,headers={'User-Agent':'ShiftGuardX-NB20-Research/1.0'})
    b=io.BytesIO()
    with urllib.request.urlopen(req,timeout=90) as res:
        while True:
            x=res.read(1024*1024)
            if not x:break
            if b.tell()+len(x)>limit:raise RuntimeError('Download exceeds safety limit')
            b.write(x)
    return b.getvalue()
def parse_air(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name=next((x for x in z.namelist() if x.lower().endswith('.csv')),None)
        if not name:raise ValueError('Air Quality CSV absent')
        stream=io.TextIOWrapper(z.open(name),encoding='utf-8-sig',errors='replace')
        rd=csv.DictReader(stream,delimiter=';'); out={k:[] for k in SOURCES[0][2]}
        for row in rd:
            for col in out:
                if len(out[col])>=MAX_VALUES:continue
                try:
                    rawv=(row.get(col) or '').strip().replace(',','.')
                    v=float(rawv)
                    if math.isfinite(v) and v!=-200:out[col].append(v)
                except (TypeError,ValueError):pass
        return out
def parse_house(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name=next((x for x in z.namelist() if x.lower().endswith('.txt') or x.lower().endswith('.csv')),None)
        if not name:raise ValueError('Household power text absent')
        stream=io.TextIOWrapper(z.open(name),encoding='utf-8',errors='replace')
        rd=csv.DictReader(stream,delimiter=';');out={k:[] for k in SOURCES[1][2]}
        for row in rd:
            for col in out:
                if len(out[col])>=MAX_VALUES:continue
                try:
                    v=float(row.get(col,''))
                    if math.isfinite(v):out[col].append(v)
                except (TypeError,ValueError):pass
            if all(len(v)>=MAX_VALUES for v in out.values()):break
        return out
series=[]; provenance=[]
for ds,url,cols in SOURCES:
    try:
        raw=download_capped(url);parsed=parse_air(raw) if ds=='air' else parse_house(raw)
        sha=hashlib.sha256(raw).hexdigest()
        count=0
        for col in cols:
            a=np.asarray(parsed.get(col,[])[:MAX_VALUES],dtype='<f8')
            if len(a)<1024:continue
            name=f'{ds}_{re.sub("[^A-Za-z0-9]+","_",col)}';file=DATADIR/(name+'.f64.bin')
            file.write_bytes(a.tobytes());assert np.array_equal(np.frombuffer(file.read_bytes(),dtype='<f8').view('<u8'),a.view('<u8'))
            series.append({'name':name,'column':col,'dataset':ds,'origin':'REAL_UCI','count':len(a),'bytes':file.stat().st_size,'file':str(file.resolve()),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
            count+=1
        provenance.append({'dataset':ds,'origin':'REAL_UCI','url':url,'archive_sha256':sha,'column_count':count,'status':'READY' if count==3 else 'PARTIAL'})
        stage('data_'+ds,'PASS' if count==3 else 'PARTIAL',f'{count} real series')
    except Exception as e:
        provenance.append({'dataset':ds,'origin':'UNAVAILABLE','url':url,'column_count':0,'status':'FAILED','error':repr(e)})
        stage('data_'+ds,'FAILED',repr(e))
savecsv('dataset_provenance.csv',provenance);savecsv('series_manifest.csv',series)
print('Real series:',len(series),'(expected: 6)')


# 03 — Run official ALP once for all available real series, using LF configuration
alp_rows=[];alp_status={'status':'NOT_RUN','expected_real_series':6,'available_real_series':len(series),'matched_rows':0,'commit':commit}
if binary_ok and series:
    config=REPO/'benchmarks'/'your_own_dataset.csv';result=REPO/'benchmarks'/'your_own_dataset_result.csv'
    head='id,column_name,data_type,path,file_type\n'
    for i,s in enumerate(series):
        assert ',' not in s['name'] and ',' not in s['file']
        head+=f"{i},{s['name']},double,{s['file']},binary\n"
    config.write_bytes(head.encode()); assert b'\r' not in config.read_bytes()
    shutil.copy2(config,OUT/'official_alp_dataset_config.csv')
    if result.exists():result.unlink()
    r=run([str(BIN)],'official_alp_six_series',BENCH_TIMEOUT,cwd=BUILD)
    if result.is_file():
        shutil.copy2(result,OUT/'official_alp_original_result.csv')
        with result.open(newline='') as f:alp_rows=list(csv.DictReader(f))
        # Match rows to registered input series, avoid stale/partial record misclassification
        names={s['name'] for s in series}; matched=[row for row in alp_rows if row.get('column') in names]
        alp_status.update({'status':'EXECUTED_COMPLETE' if r['ok'] and len(matched)==len(series) else 'EXECUTED_PARTIAL',
                           'matched_rows':len(matched),'returned_rows':len(alp_rows),'returncode':r.get('returncode'),
                           'all_six_real':len(series)==6 and len(matched)==6})
    else:alp_status.update({'status':'FAILED_NO_RESULT','returncode':r.get('returncode')})
else:stage('official_alp','BLOCKED','Missing executable or real input')
savecsv('official_alp_parsed.csv',alp_rows);savejson('official_alp_status.json',alp_status)
print(json.dumps(alp_status,indent=2))


# 04 — Generic codec baselines; exact reverse transforms and whole-stream overhead
import zlib,lzma
try:import zstandard as zstd
except ImportError:zstd=None
try:import lz4.frame as lz4frame
except ImportError:lz4frame=None
# NO pip installation required. Extra codecs included only if already available.
BACKENDS=['zlib6','lzma1']+(['zstd1','zstd3'] if zstd is not None else [])+(['lz4'] if lz4frame is not None else [])
def forward(blob,mode):
    u=np.frombuffer(blob,dtype='<u8').copy()
    if mode=='identity':return blob
    if mode=='xor':
        v=u.copy();v[1:]=u[1:]^u[:-1];return v.tobytes()
    if mode=='delta':
        v=u.copy();v[1:]=u[1:]-u[:-1];return v.tobytes()
    if mode=='shuffle':return u.view(np.uint8).reshape(-1,8).T.copy().tobytes()
    raise ValueError(mode)
def inverse(encoded,mode):
    if mode=='identity':return encoded
    if mode=='shuffle':
        a=np.frombuffer(encoded,dtype=np.uint8).reshape(8,-1).T.copy().reshape(-1)
        return a.tobytes()
    r=np.frombuffer(encoded,dtype='<u8').copy();v=r.copy()
    if mode=='xor':
        # reconstruction is sequential; XOR accumulator is exact and decoder-synchronous
        for i in range(1,len(r)):v[i]=v[i-1]^r[i]
    elif mode=='delta':
        v=np.add.accumulate(r,dtype=np.uint64)
    else:raise ValueError(mode)
    return v.tobytes()
def enc(raw,backend):
    if backend=='zlib6':return zlib.compress(raw,6)
    if backend=='lzma1':return lzma.compress(raw,preset=1)
    if backend=='zstd1':return zstd.ZstdCompressor(level=1).compress(raw)
    if backend=='zstd3':return zstd.ZstdCompressor(level=3).compress(raw)
    if backend=='lz4':return lz4frame.compress(raw)
    raise ValueError(backend)
def dec(raw,backend):
    if backend=='zlib6':return zlib.decompress(raw)
    if backend=='lzma1':return lzma.decompress(raw)
    if backend.startswith('zstd'):return zstd.ZstdDecompressor().decompress(raw)
    if backend=='lz4':return lz4frame.decompress(raw)
    raise ValueError(backend)
rows=[];errors=[]
for s in series:
    original=pathlib.Path(s['file']).read_bytes();n=s['count']
    for mode in ('identity','xor','delta','shuffle'):
        try:
            t0=time.perf_counter();transformed=forward(original,mode); transform_s=time.perf_counter()-t0
            for backend in BACKENDS:
                try:
                    t0=time.perf_counter();packed=enc(transformed,backend);encode_s=time.perf_counter()-t0+transform_s
                    t0=time.perf_counter();restored=inverse(dec(packed,backend),mode);decode_s=time.perf_counter()-t0
                    exact=bool(restored==original)
                    if not exact:raise AssertionError('Bit-exact reconstruction failed')
                    # diagnostic full container budget = 16-byte fixed header + UTF8 tags (conservative, protocol not standardized)
                    header=16+len(mode.encode())+len(backend.encode())
                    rows.append({'name':s['name'],'origin':'REAL_UCI','values':n,'transform':mode,'backend':backend,
                                 'payload_bytes':len(packed),'estimated_container_bytes':len(packed)+header,
                                 'header_bytes':header,'bits_per_value_estimate':8*(len(packed)+header)/n,
                                 'encode_seconds_single_run':encode_s,'decode_seconds_single_run':decode_s,
                                 'exact_ok':exact})
                except Exception as exc:errors.append({'name':s['name'],'mode':mode,'backend':backend,'error':repr(exc)})
        except Exception as exc:errors.append({'name':s['name'],'mode':mode,'error':repr(exc)})
    savecsv('generic_baselines.csv',rows);savecsv('generic_errors.csv',errors) # incremental checkpoints
print('Baseline records:',len(rows),'Errors:',len(errors),'Backends:',BACKENDS)
# Timings are single-run diagnostics, NOT statistically comparable to ALP CPU cycles/value.


# 05 — Compression-gap report: avoid hindsight/deployable confusion and differing protocol comparisons
alp_by={r.get('column'):r for r in alp_rows}; summary=[]
for s in series:
    subset=[r for r in rows if r['name']==s['name'] and r['exact_ok']]
    native=[r for r in subset if r['transform']=='identity']
    best_fixed=min(native,key=lambda r:r['estimated_container_bytes']) if native else None
    hindsight=min(subset,key=lambda r:r['estimated_container_bytes']) if subset else None
    official=alp_by.get(s['name'],{})
    try:bpv=float(official.get('size',''))
    except (TypeError,ValueError):bpv=None
    summary.append({'series':s['name'],'n':s['count'],'alp_reported_bits_per_value':bpv,
         'alp_compression_cycles_per_value':official.get('compression_speed(cycles_per_value)',None),
         'alp_decompression_cycles_per_value':official.get('decompression_speed(cycles_per_value)',None),
         'best_generic_untransformed':best_fixed['backend'] if best_fixed else '',
         'best_generic_untransformed_bpv_est':best_fixed['bits_per_value_estimate'] if best_fixed else '',
         'best_generic_hindsight':(hindsight['transform']+'+'+hindsight['backend']) if hindsight else '',
         'best_generic_hindsight_bpv_est':hindsight['bits_per_value_estimate'] if hindsight else '',
         'generic_hindsight_gain_pct':round(100*(1-hindsight['estimated_container_bytes']/best_fixed['estimated_container_bytes']),4) if best_fixed and hindsight else '',
         'important':'ALP native bpv and generic estimated container bpv use different protocols; not a strict head-to-head'})
savecsv('compression_gap_summary.csv',summary)
conclusion={'benchmark_ready_for_gap_study':bool(alp_status.get('all_six_real',False) and len(rows)>0),
 'official_alp_rows':alp_status['matched_rows'],'real_series':len(series),
 'limitations':['ALP native size and generic container estimates differ in metadata and benchmarking protocol',
   'Generic times are single-run; ALP reports CPU cycles/value','Best transform+codec per held-out series is hindsight, not an online selector',
   'Bit-exactness applies to parsed float64 arrays, not original CSV source bytes',
   'ALP official benchmark metrics alone do not establish byte-exact decoder correctness within this notebook'],
 'next_research_step':'Investigate observed coding gaps and integrate matched protocol; do not assert novel codec performance from this benchmark.'}
savejson('research_decision.json',conclusion)
print(json.dumps(conclusion,indent=2))


# 06 — Always export a portable evidence ZIP, including failures and setup logs
savejson('runtime.json',{'python':platform.python_version(),'platform':platform.platform(),'seconds_elapsed':round(time.time()-START,2),'numpy':np.__version__,'alp_commit':commit})
manifest=[]
for f in sorted(OUT.iterdir()):
    if f.is_file() and f.name!='sha256_manifest.json':
        manifest.append({'file':f.name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
savejson('sha256_manifest.json',manifest)
RESULT=ROOT/'SGX_NB20_Official_ALP_Multidata_results.zip'
with zipfile.ZipFile(RESULT,'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.iterdir()):
        if f.is_file():z.write(f,f.name)
print('Result ZIP:',RESULT,'size:',RESULT.stat().st_size,'bytes')
try:
    from google.colab import files
    files.download(str(RESULT))
except ImportError:print('Outside Colab; result remains at',RESULT)

