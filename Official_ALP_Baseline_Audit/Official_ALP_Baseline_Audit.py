#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX-NB19 — Official ALP Baseline & Compression Gap Audit

**Purpose:** Obtain a verifiable baseline using the **official** [cwida/ALP](https://github.com/cwida/ALP) source and its `bench_your_dataset` benchmark, alongside local conventional-codec measurements on identical IEEE-754 binary64 streams.

**Important distinctions:** ALP's reported `size` metric is bits/value; it is *not necessarily a standalone, byte-exact compressed-file size*. ALP's native benchmark timing is cycles/value; do **not** compare it directly with Python end-to-end MiB/s. Official ALP is marked `COMPLETE` only if the real target builds, executes on our custom data, and produces identifiable result records. A successful build alone is not enough. The local conventional-codec branch independently tests exact byte reconstruction; the official benchmark is not assumed to prove byte-exact reconstruction of arbitrary NaN payloads.

Free Colab **CPU runtime**. No paid API. If the official repository/build is unavailable, this notebook **exports an informative results ZIP anyway** and records `NOT_RUN`.
"""

# 00 — Configuration. Reduce caps if RAM is constrained.
MAX_ROWS_PER_SERIES = 32768
RUN_OFFICIAL_ALP = True
ALP_CLONE_TIMEOUT = 90
ALP_BUILD_TIMEOUT = 240
ALP_RUN_TIMEOUT = 120
OUTPUT = '/content/SGX_NB19_outputs' if __import__('os').path.exists('/content') else '/mnt/data/SGX_NB19_outputs'


# 01 — Standard-library helpers; no pip installation required for the base audit.
import os, sys, csv, json, time, hashlib, pathlib, shutil, subprocess, platform, zipfile, io, urllib.request, re, glob, zlib, lzma, statistics
import numpy as np
OUT=pathlib.Path(OUTPUT); OUT.mkdir(parents=True,exist_ok=True)
START=time.time()
STATUS=[]
def record(stage,status,detail=''):
    row={'stage':stage,'status':status,'detail':str(detail)[:2000]};STATUS.append(row)
    print(stage,status,str(detail)[:240])
def writecsv(name,rows,columns=None):
    p=OUT/name;rows=list(rows)
    if columns is None:columns=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    return p
def call(args,cwd=None,timeout=90,log_name='process.log'):
    try:
        p=subprocess.run(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout,check=False)
        txt=p.stdout or ''
        (OUT/log_name).write_text(txt[-500000:],encoding='utf8',errors='replace')
        return {'ok':p.returncode==0,'code':p.returncode,'output':txt,'error':''}
    except (Exception) as e:
        (OUT/log_name).write_text(type(e).__name__+': '+str(e))
        return {'ok':False,'code':None,'output':'','error':f'{type(e).__name__}: {e}'}
record('environment','OK',platform.platform()+'; Python '+platform.python_version())


# 02 — Obtain actual public data. Synthetic is a clearly labeled fallback, never called real data.
# UCI Air Quality and Household Electric Power Consumption. Raw input is parsed to float64;
# exact round-trip claims apply to *these serialized float64 arrays*, NOT the source CSV text.
SOURCES=[
 ('uci_air_quality','https://archive.ics.uci.edu/static/public/360/air+quality.zip', ['T','RH','CO(GT)']),
 ('uci_household_power','https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip',['Global_active_power','Voltage','Global_intensity'])]
series=[];provenance=[]
def download_limited(url,max_bytes=30_000_000,timeout=35):
    req=urllib.request.Request(url,headers={'User-Agent':'SGX-NB19-research/1.0'})
    with urllib.request.urlopen(req,timeout=timeout) as r:
        buf=io.BytesIO();total=0
        while True:
            part=r.read(1024*1024)
            if not part:break
            total+=len(part)
            if total>max_bytes:raise ValueError('Dataset exceeds download byte cap')
            buf.write(part)
    return buf.getvalue()
def parse_air(data):
    z=zipfile.ZipFile(io.BytesIO(data)); name=next((x for x in z.namelist() if x.lower().endswith('.csv')),None)
    if not name:raise ValueError('air-quality archive has no CSV')
    rows=[]
    with z.open(name) as f:
        for s in io.TextIOWrapper(f,encoding='utf-8-sig',errors='replace'):
            if ';' not in s:continue
            arr=s.rstrip('\r\n').split(';'); rows.append(arr)
    names=rows[0];out={k:[] for k in ['T','RH','CO(GT)']}
    for row in rows[1:]:
        for col in out:
            if col not in names:continue
            try:
                v=float(row[names.index(col)].strip().replace(',','.'))
                if np.isfinite(v) and v!=-200:out[col].append(v)
            except Exception:pass
    return out
def parse_house(data):
    z=zipfile.ZipFile(io.BytesIO(data));name=next((x for x in z.namelist() if x.lower().endswith('.txt') or x.lower().endswith('.csv')),None)
    if not name:raise ValueError('household archive has no TXT/CSV')
    out={k:[] for k in ['Global_active_power','Voltage','Global_intensity']}
    with z.open(name) as raw:
        f=io.TextIOWrapper(raw,encoding='utf8',errors='replace')
        reader=csv.DictReader(f,delimiter=';')
        for row in reader:
            for col in out:
                if len(out[col])>=MAX_ROWS_PER_SERIES:continue
                try:
                    v=float(row[col]);
                    if np.isfinite(v):out[col].append(v)
                except Exception:pass
            if all(len(x)>=MAX_ROWS_PER_SERIES for x in out.values()):break
    return out
for ds_name,url,columns in SOURCES:
    try:
        binary=download_limited(url)
        digest=hashlib.sha256(binary).hexdigest()
        parsed=parse_air(binary) if ds_name=='uci_air_quality' else parse_house(binary)
        count=0
        for col in columns:
            values=np.asarray(parsed.get(col,[])[:MAX_ROWS_PER_SERIES],dtype='<f8')
            if len(values)<512:continue
            name=ds_name+'__'+re.sub(r'[^a-zA-Z0-9]+','_',col)
            path=OUT/(name+'.f64.bin'); path.write_bytes(values.tobytes())
            series.append({'name':name,'path':str(path),'values':int(len(values)),'origin':'REAL_UCI','source':url})
            count+=1
        provenance.append({'dataset':ds_name,'url':url,'archive_sha256':digest,'series_count':count,'status':'REAL_UCI'})
        record(ds_name,'REAL_UCI',f'{count} valid float64 series')
    except Exception as e:
        provenance.append({'dataset':ds_name,'url':url,'archive_sha256':'','series_count':0,'status':'DOWNLOAD_OR_PARSE_FAILED','error':repr(e)})
        record(ds_name,'NOT_AVAILABLE',repr(e))
# Always include diagnostic synthetic controls, explicitly segregated.
rng=np.random.default_rng(17)
for name,arr in [('synthetic_walk',np.cumsum(rng.normal(0,.1,8192))),('synthetic_random_bits',rng.integers(0,2**64,size=8192,dtype=np.uint64).view(np.float64))]:
    p=OUT/(name+'.f64.bin');p.write_bytes(np.asarray(arr,dtype='<f8').tobytes())
    series.append({'name':name,'path':str(p),'values':len(arr),'origin':'SYNTHETIC_DIAGNOSTIC','source':'generated'})
writecsv('dataset_provenance.csv',provenance)
writecsv('series_manifest.csv',series)


# 03 — Local *full byte-stream* compressed sizes and decoding tests; single codec, no exhaustive search.
# Format is exact IEEE-754 little-endian binary64. Pure reversible byte transforms only.
def transform(raw,mode):
    if mode=='identity':return raw
    a=np.frombuffer(raw,dtype=np.uint8).reshape(-1,8)
    if mode=='shuffle':return a.T.copy().tobytes()
    if mode=='xor':
        u=np.frombuffer(raw,dtype='<u8');v=np.empty_like(u);v[0]=u[0];v[1:]=u[1:]^u[:-1]
        return v.tobytes()
    raise ValueError(mode)
def inverse(raw,mode):
    if mode=='identity':return raw
    if mode=='shuffle':return np.frombuffer(raw,dtype=np.uint8).reshape(8,-1).T.copy().tobytes()
    if mode=='xor':
        u=np.frombuffer(raw,dtype='<u8');return np.bitwise_xor.accumulate(u).astype('<u8',copy=False).tobytes()
    raise ValueError(mode)
backends={'zlib6':(lambda b:zlib.compress(b,6),zlib.decompress),'lzma1':(lambda b:lzma.compress(b,preset=1),lzma.decompress)}
try:
 import zstandard as zstd
 backends['zstd1']=(lambda b:zstd.ZstdCompressor(level=1).compress(b),lambda b:zstd.ZstdDecompressor().decompress(b))
except Exception:record('zstd','NOT_INSTALLED','Optional module not available; install separately to enable.')
local=[]
for s in series:
    original=pathlib.Path(s['path']).read_bytes()
    for mode in ['identity','shuffle','xor']:
        transformed=transform(original,mode)
        for cname,(enc,dec) in backends.items():
            try:
                st=time.perf_counter();compressed=enc(transformed);seconds=time.perf_counter()-st
                st=time.perf_counter();decoded=inverse(dec(compressed),mode);dt=time.perf_counter()-st
                assert decoded==original
                local.append({'name':s['name'],'origin':s['origin'],'transform':mode,'codec':cname,'original_bytes':len(original),'compressed_payload_bytes':len(compressed),'header_bytes_accounted':0,'bits_per_value':8*len(compressed)/s['values'],'encode_seconds':seconds,'decode_seconds':dt,'exact':True,'note':'payload-only; no container metadata'})
            except Exception as e:
                record('local_'+s['name']+'_'+mode+'_'+cname,'FAILED',repr(e))
writecsv('local_codec_baselines.csv',local)
record('local_baselines','COMPLETE',f'{len(local)} exact round-trip comparisons (payload-only; no container metadata)')


# 04 — Obtain official ALP code and build its official dataset benchmark target.
repo=OUT/'official_ALP'
alp={'repository':'https://github.com/cwida/ALP','commit':'','clone_status':'NOT_RUN','build_status':'NOT_RUN','benchmark_status':'NOT_RUN','dataset_config':'','result_csvs':[],'notes':[]}
if RUN_OFFICIAL_ALP:
    r=call(['git','clone','--depth','1',alp['repository'],str(repo)],timeout=ALP_CLONE_TIMEOUT,log_name='alp_clone.log')
    if r['ok']:
        alp['clone_status']='COMPLETE'
        rev=call(['git','rev-parse','HEAD'],cwd=str(repo),timeout=10,log_name='alp_commit.log')
        alp['commit']=rev['output'].strip() if rev['ok'] else 'UNKNOWN'
        cfg=call(['cmake','-DALP_BUILD_BENCHMARKING=ON','-DALP_ENABLE_VERBOSE_OUTPUT=ON','-DCMAKE_BUILD_TYPE=Release','-S','.','-B','build'],cwd=str(repo),timeout=ALP_BUILD_TIMEOUT,log_name='alp_cmake.log')
        if cfg['ok']:
            build=call(['cmake','--build','build','--target','bench_your_dataset','-j','2'],cwd=str(repo),timeout=ALP_BUILD_TIMEOUT,log_name='alp_build.log')
            alp['build_status']='COMPLETE' if build['ok'] else 'FAILED'
            if not build['ok']:alp['notes'].append('Official bench_your_dataset target failed to build. See log.')
        else:
            alp['build_status']='FAILED';alp['notes'].append('CMake configuration failed; see alp_cmake.log')
    else:
        alp['clone_status']='FAILED';alp['notes'].append('Clone failed; no official ALP comparisons are available')
else:alp['notes'].append('Official ALP disabled by RUN_OFFICIAL_ALP setting')
record('official_ALP_build',alp['build_status'],'commit '+alp['commit'])


# 05 — Discover, rather than guess, the ALP benchmark's dataset CSV path.
# The repo is versioned and paths may change. We only run when a config path can be grounded in checkout.
# Keep archive of the inspected source and configuration for independent audit.
import re
if alp['build_status']=='COMPLETE':
    source_files=list((repo/'benchmarks').rglob('*your*dataset*.cpp')) + list((repo/'benchmarks').rglob('*your*dataset*.hpp'))
    if not source_files:
        source_files=[p for p in (repo/'benchmarks').rglob('*.cpp') if 'bench_your_dataset' in p.name]
    candidate_paths=[]
    for source in source_files:
        src=source.read_text(errors='replace')
        (OUT/('inspected_'+source.name+'.txt')).write_text(src[:200000])
        for token in re.findall(r'["\']([^"\']+\.csv)["\']',src):
            if 'result' not in token.lower() and 'output' not in token.lower():
                candidate_paths.append((source,token))
    (OUT/'alp_config_candidates.json').write_text(json.dumps([{'source':str(p.relative_to(repo)),'csv_string':v} for p,v in candidate_paths],indent=2))
    # Only use an existing repo CSV with the five header fields from ALP's official documentation.
    matches=[]
    for p in repo.rglob('*.csv'):
        try:
            line=p.open(errors='replace').readline().strip().lower().replace(' ','')
            if all(x in line for x in ['column_name','data_type','path','file_type']):matches.append(p)
        except Exception:pass
    if len(matches)==1:
        config=matches[0]
    elif len(matches)>1:
        # Prefer paths explicitly referenced in the benchmark source, otherwise fail closed.
        valid=[]
        for p in matches:
            if any(p.name==pathlib.Path(tok).name for _,tok in candidate_paths):valid.append(p)
        config=valid[0] if len(valid)==1 else None
    else:config=None
    if config is None:
        alp['notes'].append('Unable to uniquely identify official dataset CSV configuration. No official benchmark executed; inspect alp_config_candidates.json and repository docs.')
        record('alp_config','NOT_FOUND',f'{len(matches)} CSV config matches; do not fabricate measurement')
    else:
        alp['dataset_config']=str(config.relative_to(repo))
        rows=[]
        for i,s in enumerate(series):
            rows.append({'id':i,'column_name':s['name'],'data_type':'double','path':s['path'],'file_type':'binary'})
        # Preserve original config before replacement.
        shutil.copy2(config,OUT/'alp_original_dataset_config.csv')
        writecsv('alp_dataset_config.csv',rows,['id','column_name','data_type','path','file_type'])
        shutil.copy2(OUT/'alp_dataset_config.csv',config)
        record('alp_config','COMPLETE',alp['dataset_config'])


# 06 — Execute official benchmark only if build AND config are ready; capture newly written files.
if alp['build_status']=='COMPLETE' and alp['dataset_config']:
    executable=repo/'build'/'benchmarks'/'bench_your_dataset'
    if not executable.exists():
        found=list((repo/'build').rglob('bench_your_dataset'))
        executable=next((p for p in found if p.is_file() and os.access(p,os.X_OK)),executable)
    if executable.exists():
        snapshot={str(p):p.stat().st_mtime_ns for p in repo.rglob('*.csv')}
        run=call([str(executable)],cwd=str(repo/'build'),timeout=ALP_RUN_TIMEOUT,log_name='alp_benchmark.log')
        latest=[]
        for p in repo.rglob('*.csv'):
            try:
                if str(p) not in snapshot or p.stat().st_mtime_ns>snapshot[str(p)]:
                    if p.resolve()!=pathlib.Path(alp['dataset_config']).resolve():latest.append(p)
            except Exception:pass
        for i,p in enumerate(latest):
            dest=OUT/('official_ALP_result_%02d.csv'%i)
            shutil.copy2(p,dest)
            alp['result_csvs'].append(dest.name)
        # Require identifiable numerical benchmark rows in official results or stdout.
        evidence='\n'.join([p.read_text(errors='replace')[:100000] for p in map(lambda x:OUT/x,alp['result_csvs'])]) + run['output']
        marker=(('bits' in evidence.lower() or 'size' in evidence.lower()) and ('column' in evidence.lower() or 'dataset' in evidence.lower()))
        alp['benchmark_status']='COMPLETE' if run['ok'] and marker else 'NOT_VERIFIED'
        if not marker:alp['notes'].append('Run did not provide identifiable per-column size evidence; do not claim ALP measured.')
        record('alp_benchmark',alp['benchmark_status'],f'exit={run["code"]} files={len(latest)}')
    else:
        alp['benchmark_status']='NOT_RUN';alp['notes'].append('Expected benchmark executable missing')
else:
    alp['benchmark_status']='NOT_RUN'
record('official_alp_measurement',alp['benchmark_status'],'See logs and official result CSV files')


# 07 — Research gap report with no artificial "improvement" claims.
real=[r for r in local if r['origin']=='REAL_UCI']
summary=[]
for name in sorted(set(r['name'] for r in local)):
    q=[r for r in local if r['name']==name]
    base=[r for r in q if r['transform']=='identity']
    best_baseline=min(base,key=lambda r:r['compressed_payload_bytes']) if base else None
    best_any=min(q,key=lambda r:r['compressed_payload_bytes']) if q else None
    summary.append({'name':name,'origin':q[0]['origin'],'best_untransformed_codec':best_baseline['codec'] if best_baseline else '', 'best_untransformed_bytes':best_baseline['compressed_payload_bytes'] if best_baseline else '', 'best_tested_transform':best_any['transform'] if best_any else '', 'best_tested_codec':best_any['codec'] if best_any else '', 'best_tested_bytes':best_any['compressed_payload_bytes'] if best_any else '', 'hindsight_only':True,'external_specialized_baseline_ready':alp['benchmark_status']=='COMPLETE'})
writecsv('compression_gap_summary.csv',summary)
(OUT/'official_alp_status.json').write_text(json.dumps(alp,indent=2))
(OUT/'research_decision.json').write_text(json.dumps({'decision':'BASELINE_AUDIT_ONLY','official_alp_verifiably_ran':alp['benchmark_status']=='COMPLETE','n_real_series':len([s for s in series if s['origin']=='REAL_UCI']),'n_synthetic_controls':len([s for s in series if s['origin']!='REAL_UCI']),'novelty':'NOT_ESTABLISHED','reason':'A benchmark audit cannot establish a new compression mechanism; official results must be compared carefully against complete container accounting.'},indent=2))
print('Official ALP:',alp['benchmark_status'],'; Real numerical series:',sum(s['origin']=='REAL_UCI' for s in series))


# 08 — Export ZIP with manifest, including failure logs. Never package cloned source or downloaded large data.
manifest=[]
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name!='manifest.json':
        manifest.append({'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT/'manifest.json').write_text(json.dumps({'files':manifest,'python':platform.python_version(),'platform':platform.platform(),'elapsed_seconds':round(time.time()-START,2),'status':STATUS},indent=2))
zip_path=pathlib.Path('/content/SGX_NB19_Official_ALP_Audit_results.zip' if pathlib.Path('/content').exists() else '/mnt/data/SGX_NB19_Official_ALP_Audit_results.zip')
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():
        if p.is_file() and p.suffix!='.bin':z.write(p,p.name)
print('Results:',zip_path, 'bytes:',zip_path.stat().st_size)
if pathlib.Path('/content').exists():
    from google.colab import files
    files.download(str(zip_path))

