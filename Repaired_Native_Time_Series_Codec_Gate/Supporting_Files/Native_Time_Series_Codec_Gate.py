#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB88 — Native time-series codec gate (PM2.5/PM10)

**Q1 research criterion:** compare the previously promising PM2.5/PM10 streams with *actual third-party codec implementations* and independent environmental data, while preserving raw float64 bits. This is a **benchmark gate, not a new algorithm**.

The notebook attempts to install: `floatbungler==0.1.2` (Rust Gorilla, Chimp, Chimp128, Patas), `pcodec==1.0.3`, and `zstandard==0.25.0`. These are research/third-party implementations; no equivalence to the authors' official Chimp test harness is asserted. Native codec names are used only when their installed encoder **and decoder** run and pass byte-identical reconstruction. If a dependency is unavailable, it is recorded, not replaced with an invented native result.

**Dataset scope:** 6 Beijing air-quality stations, including two already examined in NB87, and up to 4 *separately sourced US EPA* PM2.5 monitors. Beijing PM2.5/PM10 are the *preselected primary streams*. Beijing SO2/NO2 are negative workload controls. EPA PM2.5 from the 2024 monitor file is a different country/source; this is not a fully independent test of new *algorithmic* novelty.

**Accounting:** All tested payloads are encapsulated in the exact same 22-byte `SGX88` file envelope (magic, method ID, float64 count, payload length), including any native internal-format overhead. All outputs are decoder-checkable. This standardized envelope may duplicate internal length headers for some native formats; no claims of optimal format engineering are made. Input is defined as little-endian float64 values after published-source CSV parsing (not raw CSV bytes or timestamps). No lossy imputation is performed.

**Selection protocol:** chronological 25% calibration and 75% held-out, no held-out codec selection for deployable-policy claims. Held-out hindsight-best rows are diagnostics only. If a native baseline is missing or fails round-trip, the specialized comparison gate is **INCOMPLETE**. Python-to-Rust conversion overhead is included in measured entrypoint time, and all measurements are CPU/system specific.

**Target outcomes:** (i) exact round-trips, (ii) PM2.5/PM10 advantages over the **calibration-selected native portfolio**, (iii) shuffled-time negative controls, (iv) no large regressions and consistent benefit on the separate EPA source. Even success would establish *a workload opportunity*, not algorithmic originality.
"""

import os, sys, subprocess, json, time, platform, importlib.metadata
SMOKE = os.environ.get('SGX88_SMOKE','0') == '1'  # Internal workflow smoke test; all smoke results explicitly labeled.
SKIP_INSTALL = os.environ.get('SGX88_SKIP_INSTALL','0') == '1'
INSTALL_LOG = {}
if not SKIP_INSTALL:
    for dist,version in [('floatbungler','0.1.2'), ('pcodec','1.0.3'), ('zstandard','0.25.0')]:
        try:
            p = subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','-q',f'{dist}=={version}'], text=True, capture_output=True, timeout=240)
            INSTALL_LOG[dist]={'returncode':p.returncode,'stderr_tail':p.stderr[-1800:]}
        except Exception as exc:
            INSTALL_LOG[dist]={'error':repr(exc)}
print('Dependency installation:', INSTALL_LOG)


import io, os, csv, json, math, time, zipfile, hashlib, pathlib, struct, zlib, lzma, urllib.request
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
ROOT = pathlib.Path('/content' if pathlib.Path('/content').exists() else '/mnt/data')
OUT = ROOT/'SGX_NB88_outputs';OUT.mkdir(parents=True,exist_ok=True)
TARGET = ROOT/'SGX_NB88_results.zip'
def dumpcsv(name,rows):
    if not rows:
        rows=[{'status':'NO_ROWS'}]
    keys=list(dict.fromkeys(k for row in rows for k in row.keys()))
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def dumpjson(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=str))
def raw(v):return np.ascontiguousarray(v,dtype='<f8').tobytes()
def arr(b):
    if len(b)%8:raise ValueError('not an 8-byte multiple')
    return np.frombuffer(b,dtype='<f8').copy()
def sha(b):return hashlib.sha256(b).hexdigest()
def codec_import(name):
    try:return __import__(name)
    except Exception as exc:return None
zstd = codec_import('zstandard')
pcodec = codec_import('pcodec')
try:
    from floatbungler import gorilla,chimp,chimp128,patas
    fb={'gorilla':gorilla,'chimp':chimp,'chimp128':chimp128,'patas':patas}
except Exception as e:
    fb={}; fb_import_error=repr(e)
print('Loaded packages:', 'zstd',bool(zstd),'pcodec',bool(pcodec),'floatbungler',list(fb))


# Explicit self-delimiting 22-byte common file header. Payload may itself use a native container.
MAGIC=b'SGX88'; HEAD_FMT='<5sBQQ'; HEAD_LEN=struct.calcsize(HEAD_FMT);assert HEAD_LEN==22
METHODS=['raw','zlib6','lzma1','zstd3','shuffle_zstd3','xor_shuffle_zstd3','pcodec','gorilla','chimp','chimp128','patas']
MID={m:i for i,m in enumerate(METHODS)}
def availability(m):
    if m=='zstd3' or m in ('shuffle_zstd3','xor_shuffle_zstd3'):return zstd is not None
    if m=='pcodec':return pcodec is not None
    if m in ('gorilla','chimp','chimp128','patas'):return m in fb
    return True
def shuffle(b):
    n=len(b)//8
    return np.frombuffer(b,dtype=np.uint8).reshape(n,8).T.copy().tobytes()
def unshuffle(b,n):
    if len(b)!=n*8:raise ValueError('unshuffle byte count')
    return np.frombuffer(b,dtype=np.uint8).reshape(8,n).T.copy().tobytes()
def xor(b):
    x=np.frombuffer(b,dtype='<u8'); y=x.copy()
    if len(x)>1:y[1:]=x[1:]^x[:-1]
    return y.tobytes()
def unxor(b):
    x=np.frombuffer(b,dtype='<u8').copy()
    np.bitwise_xor.accumulate(x,out=x)
    return x.tobytes()
def packbody(m,b):
    if m=='raw':return b
    if m=='zlib6':return zlib.compress(b,6)
    if m=='lzma1':return lzma.compress(b,preset=1)
    if m=='zstd3':return zstd.ZstdCompressor(level=3).compress(b)
    if m=='shuffle_zstd3':return zstd.ZstdCompressor(level=3).compress(shuffle(b))
    if m=='xor_shuffle_zstd3':return zstd.ZstdCompressor(level=3).compress(shuffle(xor(b)))
    if m=='pcodec':return bytes(pcodec.standalone.simple_compress(arr(b),pcodec.ChunkConfig()))
    if m in fb:return bytes(fb[m].encode(arr(b).tolist()))
    raise ValueError(f'unknown or unavailable method {m}')
def unpackbody(m,payload,n):
    if m=='raw':return payload
    if m=='zlib6':return zlib.decompress(payload)
    if m=='lzma1':return lzma.decompress(payload)
    if m=='zstd3':return zstd.ZstdDecompressor().decompress(payload)
    if m=='shuffle_zstd3':return unshuffle(zstd.ZstdDecompressor().decompress(payload),n)
    if m=='xor_shuffle_zstd3':return unxor(unshuffle(zstd.ZstdDecompressor().decompress(payload),n))
    if m=='pcodec':return raw(pcodec.standalone.simple_decompress(payload))
    if m in fb:return raw(np.asarray(fb[m].decode(payload,n),dtype='<f8'))
    raise ValueError(f'unknown or unavailable method {m}')
def encode(m,b):
    if len(b)%8:raise ValueError('float64 byte count')
    body=packbody(m,b)
    return struct.pack(HEAD_FMT,MAGIC,MID[m],len(b)//8,len(body))+body
def decode(file):
    if len(file)<HEAD_LEN:raise ValueError('truncated file header')
    magic,i,n,ln=struct.unpack(HEAD_FMT,file[:HEAD_LEN])
    if magic!=MAGIC or i>=len(METHODS):raise ValueError('invalid SGX88 header')
    if len(file)!=HEAD_LEN+ln:raise ValueError('bad payload size')
    ret=unpackbody(METHODS[i],file[HEAD_LEN:],n)
    if len(ret)!=n*8:raise ValueError('decoded count mismatch')
    return ret
def measure(m,b,rep=2):
    times=[]
    for _ in range(rep):
        t=time.perf_counter_ns(); f=encode(m,b); e=(time.perf_counter_ns()-t)/1e6
        t=time.perf_counter_ns(); recovered=decode(f);d=(time.perf_counter_ns()-t)/1e6
        if recovered!=b:raise AssertionError('NOT BIT EXACT')
        times.append((len(f),e,d))
    return {'bytes':times[0][0],'encode_ms':float(np.median([x[1] for x in times])),'decode_ms':float(np.median([x[2] for x in times])),'verified':True}


# No fallback masquerading as a native codec. Check all special byte patterns and record failures explicitly.
EDGE_WORDS=[0x0000000000000000,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0x7ff80000000001ab,0x0000000000000001,0xffffffffffffffff]
edge_data=np.array(EDGE_WORDS,dtype='<u8').tobytes()
edge_data+=np.random.default_rng(8801).integers(0,2**64,size=50,dtype='<u8').tobytes()
edge_results=[]
for m in METHODS:
    if not availability(m):
        edge_results.append({'method':m,'state':'UNAVAILABLE','error':''});continue
    try:
        f=encode(m,edge_data);decoded=decode(f)
        state='PASS_BITWISE' if decoded==edge_data else 'FAIL_BITWISE'
        edge_results.append({'method':m,'state':state,'error':''})
    except Exception as exc:edge_results.append({'method':m,'state':'FAIL_OR_INAPPLICABLE','error':repr(exc)[:500]})
dumpcsv('edge_cases.csv',edge_results)
print('Edge cases:',edge_results)


# Data retrieval: official UCI Beijing archive and separately sourced US EPA AirData 2024.
# Station split retains missing values in Beijing as parsed float64 NaNs (no imputation).
BEIJING_URL='https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip'
EPA_URL='https://aqs.epa.gov/aqsweb/airdata/hourly_88502_2024.zip'
BEIJING_STATIONS=['Aotizhongxin','Changping','Dongsi','Guanyuan','Gucheng','Shunyi','Wanliu','Wanshouxigong']
BEIJING_COLS=['PM2.5','PM10','SO2','NO2']
MAX_POINTS=32000
MIN_POINTS=1600
sources=[];load_errors=[];streams=[]
def fetch(url,key):
    cached=ROOT/key
    if cached.exists() and cached.stat().st_size>1000:return cached.read_bytes()
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SGX88-academic-repro/1.0'}),timeout=160) as resp:
        blob=resp.read()
    cached.write_bytes(blob)
    return blob
def addstream(family,name,values,source_id,is_primary,is_known):
    v=np.ascontiguousarray(values,dtype='<f8')
    if len(v)<MIN_POINTS:return
    v=v[:MAX_POINTS]
    streams.append({'family':family,'stream':name,'values':v,'source_id':source_id,'primary':bool(is_primary),'previously_examined':bool(is_known)})
if SMOKE:
    rng=np.random.default_rng(88)
    for sta in ['smoke_Beijing_A','smoke_Beijing_B','smoke_US']:
        for col in ['PM2.5','PM10','SO2']:
            x=np.round(np.cumsum(rng.normal(size=1800))/4,1)
            addstream('SIMULATED_SMOKE_TEST',sta+'__'+col,x,'SIMULATED',col in ['PM2.5','PM10'],False)
    sources.append({'family':'SIMULATED_SMOKE_TEST','status':'SIMULATED_NOT_RESEARCH_DATA','url':'','sha256':'','bytes':0})
else:
    try:
        blob=fetch(BEIJING_URL,'SGX88_Beijing_UCI_original.zip')
        sources.append({'family':'beijing','status':'REAL_DOWNLOAD','url':BEIJING_URL,'sha256':sha(blob),'bytes':len(blob)})
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for station in BEIJING_STATIONS:
                candidates=[n for n in z.namelist() if n.endswith('.csv') and ('PRSA_Data_'+station+'_') in n]
                if len(candidates)!=1:
                    load_errors.append({'family':'beijing','stream':station,'error':f'Expected one station CSV, found {len(candidates)}'});continue
                with z.open(candidates[0]) as f:df=pd.read_csv(f)
                df=df.sort_values(['year','month','day','hour'],kind='stable')
                for col in BEIJING_COLS:
                    if col not in df:continue
                    v=pd.to_numeric(df[col],errors='coerce').to_numpy(dtype='<f8')
                    addstream('beijing',station+'__'+col,v,candidates[0],col in ['PM2.5','PM10'],station in ['Aotizhongxin','Changping'])
    except Exception as exc:load_errors.append({'family':'beijing','stream':'ALL','error':repr(exc)})
    try:
        blob=fetch(EPA_URL,'SGX88_EPA_2024_hourly_88502.zip')
        sources.append({'family':'us_epa','status':'REAL_DOWNLOAD','url':EPA_URL,'sha256':sha(blob),'bytes':len(blob)})
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            csv_name=next((n for n in z.namelist() if n.endswith('.csv')),None)
            if not csv_name:raise ValueError('no csv in EPA zip')
            # Memory bounded: only keep top 4 monitors, ordered local observations.
            use=['State Code','County Code','Site Num','POC','Date Local','Time Local','Sample Measurement']
            counts=Counter()
            with z.open(csv_name) as f:
                for chunk in pd.read_csv(f,usecols=use,low_memory=False,chunksize=75000,dtype={'State Code':str,'County Code':str,'Site Num':str,'POC':str}):
                    keys=chunk[['State Code','County Code','Site Num','POC']].fillna('NA').astype(str).agg('-'.join,axis=1)
                    counts.update(keys.value_counts().to_dict())
            top=[k for k,c in counts.most_common(25) if c>=MIN_POINTS][:4]
            collect=defaultdict(list)
            with z.open(csv_name) as f:
                for chunk in pd.read_csv(f,usecols=use,low_memory=False,chunksize=75000,dtype={'State Code':str,'County Code':str,'Site Num':str,'POC':str}):
                    keys=chunk[['State Code','County Code','Site Num','POC']].fillna('NA').astype(str).agg('-'.join,axis=1)
                    piece=chunk.loc[keys.isin(top)].copy()
                    if piece.empty:continue
                    piece['_key']=keys.loc[piece.index].values
                    piece['_time']=piece['Date Local'].astype(str)+' '+piece['Time Local'].astype(str)
                    for k,group in piece.groupby('_key'):
                        collect[k].extend(zip(group['_time'].tolist(),pd.to_numeric(group['Sample Measurement'],errors='coerce').tolist()))
            for k in top:
                tuples=sorted(collect.get(k,[]),key=lambda t:t[0]);vals=np.asarray([v for _,v in tuples],dtype='<f8')
                addstream('us_epa','EPA_2024_PM25__'+k,vals,csv_name,True,False)
    except Exception as exc:load_errors.append({'family':'us_epa','stream':'ALL','error':repr(exc)})
dumpcsv('dataset_provenance.csv',sources);dumpcsv('download_errors.csv',load_errors)
print('Real/Simulated streams:',len(streams),'by family:',Counter(x['family'] for x in streams))


# Frozen, chronological calibration policy; all eligible methods measured but deployed choice made before held-out evaluation.
# Eligibility is based on an actual round-trip for that stream's calibration bytes, not just import success.
measurements=[];decisions=[];problems=[];negative_controls=[]
GENERIC=['zlib6','lzma1','zstd3','shuffle_zstd3']
SPECIALIZED=['pcodec','gorilla','chimp','chimp128','patas']
NB87=['xor_shuffle_zstd3']
for idx,s in enumerate(streams):
    v=s['values']; n=len(v);cut=max(256,n//4)
    cal=raw(v[:cut]);held=raw(v[cut:]);full=raw(v)
    ok={};trial_ms=0
    for m in METHODS:
        if not availability(m):
            problems.append({'family':s['family'],'stream':s['stream'],'method':m,'stage':'availability','error':'DEPENDENCY_UNAVAILABLE'});continue
        try:
            t=time.perf_counter_ns();calrow=measure(m,cal,rep=1);trial_ms+=(time.perf_counter_ns()-t)/1e6
            ok[m]=calrow
        except Exception as exc:
            problems.append({'family':s['family'],'stream':s['stream'],'method':m,'stage':'calibration','error':repr(exc)[:350]})
    if 'zlib6' not in ok:
        problems.append({'family':s['family'],'stream':s['stream'],'method':'ALL','stage':'gate','error':'no zlib baseline'});continue
    def choose(options):return min(options,key=lambda x:ok[x]['bytes']) if options else None
    generic=choose([m for m in GENERIC if m in ok])
    specialized=choose([m for m in SPECIALIZED if m in ok])
    old=choose([m for m in NB87 if m in ok])
    any_method=choose([m for m in METHODS if m in ok and m!='raw'])
    selected={'generic':generic,'specialized':specialized,'old_xor':old,'portfolio':any_method,'fixed_zlib':'zlib6'}
    res={}
    for m in sorted(set(x for x in selected.values() if x)):
        try:
            r=measure(m,held,rep=3)
            res[m]=r
            measurements.append({'family':s['family'],'stream':s['stream'],'group':'heldout','method':m,'n_values':n-cut,'complete_frame_bytes':r['bytes'],'encode_ms':r['encode_ms'],'decode_ms':r['decode_ms'],'bit_exact':r['verified'],'heldout_sha256':sha(held)})
        except Exception as exc:problems.append({'family':s['family'],'stream':s['stream'],'method':m,'stage':'heldout','error':repr(exc)[:350]})
    row={'family':s['family'],'stream':s['stream'],'n_total':n,'n_cal':cut,'n_heldout':n-cut,'primary':s['primary'],'previously_examined':s['previously_examined'],'input_sha256':sha(full),'heldout_sha256':sha(held),'cal_trial_ms':round(trial_ms,4)}
    for k,m in selected.items():
        row[k+'_codec']=m or ''
        row[k+'_bytes']=res[m]['bytes'] if m in res else None
    if row['generic_bytes'] and row['specialized_bytes']:
        row['specialized_saving_pct']=100*(row['generic_bytes']-row['specialized_bytes'])/row['generic_bytes']
    if row['generic_bytes'] and row['old_xor_bytes']:
        row['old_xor_saving_pct']=100*(row['generic_bytes']-row['old_xor_bytes'])/row['generic_bytes']
    if row['specialized_bytes'] and row['old_xor_bytes']:
        row['old_xor_vs_specialized_pct']=100*(row['specialized_bytes']-row['old_xor_bytes'])/row['specialized_bytes']
    decisions.append(row)
    print(f"{idx+1}/{len(streams)}", s['stream'], 'generic',generic,'specialized',specialized,'old',old)
    # Negative-control: test a PREREGISTERED subset, without retuning selections after shuffle.
    if s['family']=='beijing' and s['stream'].startswith(('Aotizhongxin__PM2.5','Changping__PM2.5','Dongsi__PM2.5','Shunyi__PM2.5')):
        rng=np.random.default_rng(88088);order=rng.permutation(n-cut);sh=raw(v[cut:][order])
        try:
            a=measure(generic,sh,rep=1)
            b=measure(old,sh,rep=1) if old else None
            c=measure(specialized,sh,rep=1) if specialized else None
            negative_controls.append({'stream':s['stream'],'generic_choice':generic,'specialized_choice':specialized,'xor_choice':old,'shuffled_generic_bytes':a['bytes'],'shuffled_specialized_bytes':c['bytes'] if c else None,'shuffled_old_xor_bytes':b['bytes'] if b else None,'bit_exact':True,'seed':88088})
        except Exception as exc:problems.append({'family':s['family'],'stream':s['stream'],'method':'negative_control','stage':'shuffled','error':repr(exc)[:350]})
dumpcsv('heldout_methods.csv',measurements)
dumpcsv('frozen_decisions.csv',decisions)
dumpcsv('inapplicable_and_errors.csv',problems)
dumpcsv('shuffle_controls.csv',negative_controls)


# Primary decisions require successful third-party native baselines and data from the separate US EPA source.
family_summary=[]
for family in sorted(set(r['family'] for r in decisions)):
    grp=[r for r in decisions if r['family']==family]
    for test in ['primary','controls','all']:
        if test=='primary':subset=[r for r in grp if r['primary']]
        elif test=='controls':subset=[r for r in grp if not r['primary']]
        else:subset=grp
        rows=[r for r in subset if r.get('generic_bytes') and r.get('specialized_bytes')]
        if not rows:continue
        gb=sum(r['generic_bytes'] for r in rows);sp=sum(r['specialized_bytes'] for r in rows)
        oldrows=[r for r in rows if r.get('old_xor_bytes')]
        og=sum(r['generic_bytes'] for r in oldrows);ox=sum(r['old_xor_bytes'] for r in oldrows)
        family_summary.append({'family':family,'partition':test,'streams':len(rows),'generic_bytes':gb,'specialized_bytes':sp,'specialized_saving_pct':100*(gb-sp)/gb,'old_xor_eligible_streams':len(oldrows),'old_xor_saving_pct':100*(og-ox)/og if og else None,'specialized_wins':sum(r['specialized_bytes']<r['generic_bytes'] for r in rows),'old_xor_wins_vs_specialized':sum(r.get('old_xor_bytes',math.inf)<r['specialized_bytes'] for r in rows)})
dumpcsv('family_summary.csv',family_summary)
native_available=bool(pcodec) and all(m in fb for m in ('gorilla','chimp','chimp128','patas')) and (zstd is not None)
expected=set(['beijing','us_epa']);completed=set(r['family'] for r in decisions)
completed_primary=sum(bool(r['primary']) for r in decisions)
real_data=(not SMOKE) and expected.issubset(completed)
native_qualified=sum(bool(r.get('specialized_bytes')) for r in decisions)
old_qualified=sum(bool(r.get('old_xor_bytes')) for r in decisions if r['primary'])
old_beats_specialized=sum(bool(r.get('old_xor_bytes') and r.get('specialized_bytes') and r['old_xor_bytes']<r['specialized_bytes']) for r in decisions if r['primary'])
failed_bitwise=sum(1 for r in problems if 'NOT BIT EXACT' in r.get('error',''))
if SMOKE:state='SIMULATED_WORKFLOW_ONLY'
elif not real_data:state='INCOMPLETE_EXTERNAL_DATA'
elif not native_available or native_qualified<completed_primary or old_qualified<completed_primary:state='INCOMPLETE_NATIVE_SPECIALIZED_BASELINES'
elif failed_bitwise:state='INCOMPLETE_SOME_CODECS_NOT_BIT_EXACT'
elif old_beats_specialized==0:state='STOP_OLD_XOR_NOT_COMPETITIVE_WITH_SPECIALIZED_CODECS'
else:state='WORKLOAD_OPPORTUNITY_ONLY_NOT_A_NEW_ALGORITHM'
result={'state':state,'number_of_streams':len(decisions),'real_data_two_sources':bool(real_data),'native_package_available':native_available,'native_specialized_valid_streams':native_qualified,'old_xor_valid_primary_streams':old_qualified,'primary_streams':completed_primary,'old_xor_wins_over_native_specialized_count':old_beats_specialized,'bitwise_failures_recorded':failed_bitwise,'source_issues':load_errors,'precision_scope':'parsed float64 value bits, no original CSV preservation','study_limitations':['no official authors Chimp native test harness','selection setup cost is not included in heldout encode_ms','native runtime package conversion included; runtime architecture-dependent','no new compression algorithm in NB88','Beijing station holdouts come from same UCI source; EPA is independent source/country','EPA monitor sampling gaps are retained (timestamps not compressed)','complete SGX88 22-byte common envelope deliberately includes duplicated internal format metadata when native codecs have headers']}
dumpjson('decision.json',result)
versions={}
for dist in ['floatbungler','pcodec','zstandard','numpy','pandas']:
    try:versions[dist]=importlib.metadata.version(dist)
    except Exception:versions[dist]='UNAVAILABLE'
dumpjson('environment.json',{'python':sys.version,'platform':platform.platform(),'versions':versions,'install_log':INSTALL_LOG,'smoke':SMOKE})
dumpjson('protocol.json',{'calibration_fraction':0.25,'max_points_per_stream':MAX_POINTS,'required_native_methods':SPECIALIZED,'methods':METHODS,'negative_control_seed':88088,'selection':'calibration only','file_header':'SGX88 22 bytes equal for all codecs','source_scope':'Beijing UCI station and US EPA hourly PM2.5, parser float64 values','no_novelty_claim':True})
with zipfile.ZipFile(TARGET,'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.glob('*')):
        if f.is_file():z.write(f,f.name)
print('DECISION',json.dumps(result,indent=2)[:3000])
print('ZIP',TARGET,TARGET.stat().st_size)
try:
    from google.colab import files;files.download(str(TARGET))
except ImportError:pass

