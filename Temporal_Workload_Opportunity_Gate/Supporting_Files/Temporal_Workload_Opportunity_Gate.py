#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB87 — Temporal Workload Opportunity Gate

**Research goal:** Decide whether time-ordered numerical sensor signals contain a reproducible compression opportunity that remains after strong *known* transforms and portable metadata accounting. This is **not** an invention or a Q1-ready algorithm.

We previously rejected NB81 DSEMR, NB85 exception-vocabulary coding, and NB86 binary-precision stratification. This single notebook is a deliberate **workload-selection study** before another algorithm is written.

## Experimental commitments (registered before results)

1. **Real chronologically ordered workloads:** EEG Eye State (UCI 264) and Beijing Multi-Site Air Quality (UCI 501). **Non-temporal negative control:** model-generated Electrical Grid Stability (UCI 471). Preserve source bytes' SHA-256 and the parsed float64 stream's SHA-256. No synthetic substitute for download failures.
2. **Fair size units:** all codecs use the same 22-byte reversible SGX87 container. Compare the exact same held-out values and lengths. Every compressed representation decodes back to the original float64 bit pattern.
3. **Methods already known:** zlib, LZMA, Zstandard, byte-stream split+Zstandard, XOR+Zstandard, XOR+byte-stream split+Zstandard. **These are controls, not a newly proposed algorithm.**
4. **No hindsight policy choice:** Select best non-temporal and best temporal representation on the first chronological 25% only; evaluate the remaining 75% without changing the decisions. Log calibration trial cost.
5. **Negative control:** apply a deterministic random permutation to the held-out sequence (the same rows, different order). Compare whether the temporal method loses its apparent advantage. This is evidence about sequential structure, not proof of novel coding.
6. **Gate for a workload worth investigating:** both *independent real time-series families* must show at least 5% aggregate saving for the preselected temporal method over the preselected non-temporal method, plus at least 2 percentage points stronger improvement in original order than shuffled control. No bitwise mismatches, no worse-than-10% stream regression. Passing only licenses **mechanism research**, not a publication or originality claim.

**Prior-art exclusions:** XOR/delta floating-point coding and prediction are established in Gorilla, Chimp, FPC, Elf-family methods. ALP covers adaptive decimal and front-bit compression. There is **no novelty claim** for XOR+Zstandard or this experiment's selector.

Key sources:
- ALP (SIGMOD 2024): https://ir.cwi.nl/pub/33334
- Elf* (IEEE IoT J., 2025): https://ieeexplore.ieee.org/document/10949160/
- UCI EEG Eye State: https://archive.ics.uci.edu/dataset/264/eeg+eye+state
- UCI Beijing Air Quality: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
- UCI Grid Stability (simulated): https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data

**Colab:** Runtime → Run all. The notebook downloads real datasets itself and writes `SGX_NB87_results.zip`. Set `SGX87_SMOKE=1` in your environment *only* to test the notebook with labeled simulated data. That mode can **never** pass the research gate.
"""

# ---- Standalone imports and configuration ----
import os, sys, time, json, io, csv, zipfile, struct, zlib, lzma, hashlib, traceback, subprocess, urllib.request, platform, statistics
from pathlib import Path
import numpy as np
import pandas as pd

SMOKE = os.environ.get('SGX87_SMOKE', '0') == '1'
OUTPUT_NAME = 'SGX_NB87_SMOKE' if SMOKE else 'SGX_NB87'
ROOT = Path('/content') if Path('/content').exists() else Path('/mnt/data')
OUT = ROOT / (OUTPUT_NAME + '_outputs')
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / 'SGX_NB87_source_downloads'
CACHE.mkdir(parents=True, exist_ok=True)
MAX_ROWS = 12000
SEED = 870126
rng = np.random.default_rng(SEED)

try:
    import zstandard as zstd
except ImportError:
    zstd = None
    if not SMOKE:
        try:
            r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'zstandard>=0.22,<1'], capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                import zstandard as zstd
            else:
                print('zstandard install failed:', r.stderr[-750:])
        except Exception as e:
            print('zstandard install unavailable:', str(e))

GENERAL = ['zlib6', 'lzma1'] + (['zstd3', 'shuffle_zstd3'] if zstd else [])
TEMPORAL = ['xor_zstd3', 'xor_shuffle_zstd3'] if zstd else ['xor_zlib6']
ALL_METHODS = ['raw'] + GENERAL + TEMPORAL
print('TEST_MODE:', 'SIMULATED_SMOKE_ONLY' if SMOKE else 'REAL_EXTERNAL_DATA', 'zstandard:', bool(zstd), 'methods:', ALL_METHODS)

def save_csv(name, rows, columns=None):
    path = OUT/name
    if columns is None:
        columns = list(rows[0].keys()) if rows else ['status']
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        if rows: w.writerows(rows)

def sha(data): return hashlib.sha256(data).hexdigest()
def raw_f64(v): return np.ascontiguousarray(v, dtype='<f8').tobytes()

def shuffle(b): return np.frombuffer(b, dtype=np.uint8).reshape(-1,8).T.copy().tobytes()
def unshuffle(b): return np.frombuffer(b, dtype=np.uint8).reshape(8,-1).T.copy().tobytes()
def xor_forward(b):
    a = np.frombuffer(b, dtype='<u8')
    result = a.copy()
    if len(a)>1: result[1:] = a[1:] ^ a[:-1]
    return result.astype('<u8').tobytes()
def xor_backward(b):
    a = np.frombuffer(b, dtype='<u8')
    return np.bitwise_xor.accumulate(a).astype('<u8').tobytes()

MAGIC = b'SGX87'
METHOD_ID = {m:i for i,m in enumerate(['raw','zlib6','lzma1','zstd3','shuffle_zstd3','xor_zstd3','xor_shuffle_zstd3','xor_zlib6'])}
ID_METHOD = {v:k for k,v in METHOD_ID.items()}

def compress_payload(m,raw):
    if m=='raw': return raw
    if m=='zlib6': return zlib.compress(raw,6)
    if m=='lzma1': return lzma.compress(raw,preset=1)
    if m=='zstd3': return zstd.ZstdCompressor(level=3).compress(raw)
    if m=='shuffle_zstd3': return zstd.ZstdCompressor(level=3).compress(shuffle(raw))
    if m=='xor_zstd3': return zstd.ZstdCompressor(level=3).compress(xor_forward(raw))
    if m=='xor_shuffle_zstd3': return zstd.ZstdCompressor(level=3).compress(shuffle(xor_forward(raw)))
    if m=='xor_zlib6': return zlib.compress(xor_forward(raw),6)
    raise KeyError(m)

def decompress_payload(m,body):
    if m=='raw': return body
    if m=='zlib6': return zlib.decompress(body)
    if m=='lzma1': return lzma.decompress(body)
    if m=='zstd3': return zstd.ZstdDecompressor().decompress(body)
    if m=='shuffle_zstd3': return unshuffle(zstd.ZstdDecompressor().decompress(body))
    if m=='xor_zstd3': return xor_backward(zstd.ZstdDecompressor().decompress(body))
    if m=='xor_shuffle_zstd3': return xor_backward(unshuffle(zstd.ZstdDecompressor().decompress(body)))
    if m=='xor_zlib6': return xor_backward(zlib.decompress(body))
    raise KeyError(m)

def encode(m,raw):
    if len(raw)%8: raise ValueError('raw input must have full float64 words')
    body = compress_payload(m,raw)
    return MAGIC + struct.pack('<BQQ',METHOD_ID[m],len(raw)//8,len(body)) + body

def decode(blob):
    if len(blob)<22 or blob[:5] != MAGIC: raise ValueError('invalid/truncated SGX87 frame')
    mid,n,bodysize = struct.unpack('<BQQ',blob[5:22])
    if len(blob)-22!=bodysize or mid not in ID_METHOD: raise ValueError('invalid metadata or truncated body')
    raw = decompress_payload(ID_METHOD[mid],blob[22:])
    if len(raw)!=8*n: raise ValueError('decoded raw byte count mismatch')
    return raw

def measure(m,raw,repeats=1):
    times = []
    for _ in range(repeats):
        start=time.perf_counter_ns();c=encode(m,raw);et=(time.perf_counter_ns()-start)/1e6
        start=time.perf_counter_ns();d=decode(c);dt=(time.perf_counter_ns()-start)/1e6
        if d != raw: raise AssertionError('BITWISE MISMATCH '+m)
        times.append((len(c),et,dt))
    return {'bytes':times[0][0], 'encode_ms':float(statistics.median(x[1] for x in times)), 'decode_ms':float(statistics.median(x[2] for x in times))}

def test_edges():
    patterns=[0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,
              0x7ff8000000000001,0x7ff8000000004321,0x0000000000000001,
              0x8000000000000001,0x7fefffffffffffff,0xffffffffffffffff]
    cases=[('empty', b''),('special_ieee',np.array(patterns,dtype='<u8').tobytes()),
           ('repeat_negative_zero',np.array([0x8000000000000000]*100,dtype='<u8').tobytes())]
    rows=[]
    for label,b in cases:
        for m in ALL_METHODS:
            try:
                c=encode(m,b)
                ok=(decode(c)==b)
                rows.append({'case':label,'method':m,'passed':ok,'reason':''})
            except Exception as e:
                rows.append({'case':label,'method':m,'passed':False,'reason':str(e)})
    try:
        decode(encode('zlib6',b'')[:-1]); rows.append({'case':'truncation','method':'zlib6','passed':False,'reason':'truncated data not rejected'})
    except ValueError:
        rows.append({'case':'truncation','method':'zlib6','passed':True,'reason':'truncation rejected'})
    save_csv('edge_cases.csv',rows)
    assert all(r['passed'] for r in rows), 'edge test failed'
    return len(rows)


# ---- UCI archive retrieval, provenance, and exact float64 stream extraction ----
SOURCES={
 'eeg_eye_state': {'url':'https://archive.ics.uci.edu/static/public/264/eeg+eye+state.zip', 'type':'arff', 'temporally_ordered':True, 'doi':'10.24432/C57G7J'},
 'beijing_air_quality': {'url':'https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip', 'type':'csv_nested', 'temporally_ordered':True,'doi':'10.24432/C5RK5G'},
 'grid_stability_simulated': {'url':'https://archive.ics.uci.edu/static/public/471/electrical+grid+stability+simulated+data.zip', 'type':'csv', 'temporally_ordered':False, 'doi':'10.24432/C5PG66'}
}

def get_archive(name):
    url = SOURCES[name]['url'];dest = CACHE/(name+'.zip')
    if dest.is_file():
        body=dest.read_bytes()
        if zipfile.is_zipfile(io.BytesIO(body)):return body
        dest.unlink()
    last=''
    for trial in range(2):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 SGX87 scientific reproducibility'})
            with urllib.request.urlopen(req,timeout=75) as r:
                body=r.read(35_000_001)
            if len(body)>35_000_000: raise ValueError('download too large for protocol')
            if not zipfile.is_zipfile(io.BytesIO(body)): raise ValueError('download is not ZIP')
            dest.write_bytes(body)
            return body
        except Exception as e:
            last=repr(e)
    raise RuntimeError(f'dataset download failed ({name}): {last}')

def members(blob, depth=0):
    if depth>3:return
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for info in z.infolist():
            if info.is_dir() or info.file_size>55_000_000:continue
            path=info.filename; lower=path.lower()
            if lower.endswith('.zip'):
                try:yield from members(z.read(info),depth+1)
                except Exception:continue
            elif lower.endswith(('.csv','.arff')):
                yield path,z.read(info)

def parse_eeg(blob):
    candidates=[(p,b) for p,b in members(blob) if p.lower().endswith('.arff')]
    if not candidates:raise ValueError('EEG zip contained no .arff')
    path,data=sorted(candidates,key=lambda x:len(x[1]),reverse=True)[0]
    t=data.decode('utf-8-sig','replace').splitlines()
    ind=next(i for i,ln in enumerate(t) if ln.strip().lower()=='@data')
    dat='\n'.join(ln.strip() for ln in t[ind+1:] if ln.strip() and not ln.strip().startswith('%'))
    arr=np.genfromtxt(io.StringIO(dat),delimiter=',',dtype=np.float64,missing_values='?',filling_values=np.nan)
    if arr.ndim!=2 or arr.shape[1]<15:raise ValueError('EEG data shape unexpected '+str(arr.shape))
    names=['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    return {f'EEG_{name}':arr[:MAX_ROWS,j].copy() for j,name in enumerate(names[:8])},path

def parse_beijing(blob):
    candidates=sorted([(p,b) for p,b in members(blob) if p.lower().endswith('.csv') and 'prsa_data_' in p.lower()],key=lambda x:x[0].lower())
    if len(candidates)<1:raise ValueError('Beijing archive has no recognized station .csv')
    streams={};files=[]
    for path,b in candidates[:2]:
        d=pd.read_csv(io.BytesIO(b),nrows=MAX_ROWS)
        names=['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM']
        chosen=[c for c in names if c in d.columns][:7]
        if len(chosen)<3: raise ValueError('Beijing numeric columns missing')
        station=Path(path).stem.replace('PRSA_Data_','')[:22]
        for c in chosen:
            vals=pd.to_numeric(d[c],errors='coerce').to_numpy(dtype=np.float64)
            streams[f'{station}__{c}']=vals
        files.append(path)
    return streams,';'.join(files)

def parse_grid(blob):
    candidates=[(p,b) for p,b in members(blob) if p.lower().endswith('.csv')]
    if not candidates:raise ValueError('grid zip contained no csv')
    path,b=max(candidates,key=lambda v:len(v[1]));d=pd.read_csv(io.BytesIO(b),nrows=MAX_ROWS)
    names=['tau1','tau2','tau3','tau4','p1','p2','p3','p4','g1','g2','g3','g4','stab']
    present=[c for c in names if c in d.columns]
    if len(present)<5: raise ValueError('grid numeric columns missing '+str(list(d.columns)[:10]))
    return {f'GRID_{c}':pd.to_numeric(d[c],errors='coerce').to_numpy(dtype=np.float64) for c in present[:8]},path

def simulated_dataset_inputs():
    # This branch is exclusively for smoke testing; never accepted as external evidence.
    r=np.random.default_rng(42)
    x=np.cumsum(r.standard_normal(256))
    y=np.repeat(r.normal(size=32),8)
    return {
      'SIMULATED_EEG': {'SIM_EEG_1':x,'SIM_EEG_2':y},
      'SIMULATED_BEIJING':{'SIM_AIR_1':x+0.001*r.normal(size=len(x)),'SIM_AIR_2':y},
      'SIMULATED_GRID':{'SIM_GRID_1':r.uniform(size=256),'SIM_GRID_2':r.integers(-3,3,size=256).astype(float)}
    }

def load_workloads():
    sources=[];errors=[];data={}
    if SMOKE:
        data=simulated_dataset_inputs()
        for fam,cols in data.items():
            sources.append({'family':fam,'status':'SIMULATED_SMOKE','url':'','source_sha256':'','archive_member':'','stream_count':len(cols),'temporally_ordered':fam!='SIMULATED_GRID','doi':''})
        return data,sources,errors
    parser={'eeg_eye_state':parse_eeg,'beijing_air_quality':parse_beijing,'grid_stability_simulated':parse_grid}
    for fam,spec in SOURCES.items():
        try:
            body=get_archive(fam)
            arrs,member=parser[fam](body)
            if not arrs:raise ValueError('no usable float64 columns')
            data[fam]=arrs
            sources.append({'family':fam,'status':'REAL_UCI','url':spec['url'],'source_sha256':sha(body),'archive_member':member,'stream_count':len(arrs),'temporally_ordered':spec['temporally_ordered'],'doi':spec['doi']})
            print('LOADED',fam,'streams',len(arrs),'members',member[:120],'archive_sha',sha(body)[:12])
        except Exception as e:
            msg=repr(e)
            errors.append({'family':fam,'stage':'download_or_parse','error':msg})
            sources.append({'family':fam,'status':'FAILED_NO_SUBSTITUTION','url':spec['url'],'source_sha256':'','archive_member':'','stream_count':0,'temporally_ordered':spec['temporally_ordered'],'doi':spec['doi']})
            print('DATASET FAILED (no substitute):',fam,msg)
    return data,sources,errors


# ---- Frozen calibration, held-out evaluation, matched permutation controls ----
edge_count=test_edges()
loaded, provenance, errors=load_workloads()
save_csv('dataset_provenance.csv',provenance)

policy_rows=[]; method_rows=[]; oracle_rows=[]
for family, columns in loaded.items():
    for name, vals in columns.items():
        try:
            vals=np.asarray(vals,dtype=np.float64)[:MAX_ROWS]
            if len(vals)<64:raise ValueError('too few samples for 25% calibration')
            cut=max(32,len(vals)//4)
            cal=raw_f64(vals[:cut]);hold=raw_f64(vals[cut:]);n_hold=len(vals)-cut
            trial_ms=0.;cal_metrics={}
            for m in ALL_METHODS:
                t=time.perf_counter_ns()
                cal_metrics[m]=measure(m,cal)
                trial_ms+=(time.perf_counter_ns()-t)/1e6
            baseline=min(GENERAL,key=lambda m:cal_metrics[m]['bytes'])
            temporal=min(TEMPORAL,key=lambda m:cal_metrics[m]['bytes'])
            # Both method choices are frozen before heldout evaluation.
            hold_results={m:measure(m,hold,repeats=3 if m in (baseline,temporal) else 1) for m in ALL_METHODS}
            # Unavailable methods were not silently substituted: ALL_METHODS is explicit.
            best_baseline_oracle=min(GENERAL,key=lambda m:hold_results[m]['bytes'])
            best_temporal_oracle=min(TEMPORAL,key=lambda m:hold_results[m]['bytes'])
            for m,d in hold_results.items():
                method_rows.append({'family':family,'stream':name,'method':m,'cal_selected_baseline':baseline,'cal_selected_temporal':temporal,
                                    'n_heldout':n_hold,'bytes':d['bytes'],'enc_ms':d['encode_ms'],'dec_ms':d['decode_ms'],'heldout_sha256':sha(hold),'bit_exact':True})
            shuffled=vals[cut:].copy()
            rnd=np.random.default_rng(SEED + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little'))
            shuffled=shuffled[rnd.permutation(len(shuffled))]
            shuffled_raw=raw_f64(shuffled)
            b_shuf=measure(baseline,shuffled_raw,1);t_shuf=measure(temporal,shuffled_raw,1)
            base=hold_results[baseline];cand=hold_results[temporal]
            gain=100*(base['bytes']-cand['bytes'])/base['bytes']
            shuffled_gain=100*(b_shuf['bytes']-t_shuf['bytes'])/b_shuf['bytes']
            policy_rows.append({'family':family,'stream':name,'n_total':len(vals),'n_cal':cut,'n_heldout':n_hold,
                                'is_simulated':SMOKE or family=='grid_stability_simulated','chronological_family':family in ['eeg_eye_state','beijing_air_quality'],
                                'raw_sha256':sha(raw_f64(vals)),'heldout_sha256':sha(hold),
                                'baseline_cal_choice':baseline,'temporal_cal_choice':temporal,
                                'baseline_heldout_bytes':base['bytes'],'temporal_heldout_bytes':cand['bytes'],
                                'gain_percent':gain,'shuffled_baseline_bytes':b_shuf['bytes'],'shuffled_temporal_bytes':t_shuf['bytes'],
                                'shuffled_gain_percent':shuffled_gain,'sequential_lift_pp':gain-shuffled_gain,
                                'baseline_encode_ms':base['encode_ms'],'temporal_encode_ms':cand['encode_ms'],
                                'baseline_decode_ms':base['decode_ms'],'temporal_decode_ms':cand['decode_ms'],
                                'calibration_trial_ms':trial_ms,'bit_exact':True})
            oracle_rows.append({'family':family,'stream':name,'best_baseline_with_hindsight':best_baseline_oracle,
                                'best_baseline_bytes':hold_results[best_baseline_oracle]['bytes'],
                                'best_temporal_with_hindsight':best_temporal_oracle,
                                'best_temporal_bytes':hold_results[best_temporal_oracle]['bytes'],
                                'note':'DIAGNOSTIC ONLY; method chosen AFTER observing held-out data'})
        except Exception as e:
            errors.append({'family':family,'stream':name,'stage':'measurement','error':repr(e)})
            print('STREAM ERROR',family,name,repr(e))
save_csv('frozen_stream_decisions.csv',policy_rows)
save_csv('all_heldout_methods.csv',method_rows)
save_csv('hindsight_oracle_diagnostic_only.csv',oracle_rows)
save_csv('errors.csv',errors,columns=['family','stream','stage','error'])

families=[]
for family in sorted(set(r['family'] for r in policy_rows)):
    block=[r for r in policy_rows if r['family']==family]
    b=sum(r['baseline_heldout_bytes'] for r in block);t=sum(r['temporal_heldout_bytes'] for r in block)
    bs=sum(r['shuffled_baseline_bytes'] for r in block);ts=sum(r['shuffled_temporal_bytes'] for r in block)
    gain=100*(b-t)/b if b else 0.;sgain=100*(bs-ts)/bs if bs else 0.
    families.append({'family':family,'streams':len(block),'baseline_bytes':b,'temporal_bytes':t,'net_gain_percent':gain,
                     'shuffled_baseline_bytes':bs,'shuffled_temporal_bytes':ts,'shuffled_gain_percent':sgain,
                     'sequential_lift_pp':gain-sgain,'candidate_stream_wins':sum(r['gain_percent']>0 for r in block),
                     'worst_stream_gain_pct':min(r['gain_percent'] for r in block),
                     'total_calibration_ms':sum(r['calibration_trial_ms'] for r in block),
                     'temporally_ordered':family in ['eeg_eye_state','beijing_air_quality']})
save_csv('family_summary.csv',families)


# ---- Conservative research gate and exports ----
real_temporal=[r for r in families if r['temporally_ordered']]
control=next((r for r in families if r['family']=='grid_stability_simulated'),None)
no_mismatch=all(r['bit_exact'] for r in policy_rows) and edge_count>0
complete_real = (len(real_temporal)==2 and control is not None and not errors and bool(zstd) and not SMOKE)
family_passes=[r for r in real_temporal if r['net_gain_percent']>=5.0 and r['sequential_lift_pp']>=2.0 and r['worst_stream_gain_pct']>=-10.0]
# Non-time-series grid dataset cannot count as a chronological signal family.
strong_control=bool(control and control['net_gain_percent']>=5.0 and control['sequential_lift_pp']>=2.0)
workload_go=bool(complete_real and no_mismatch and len(family_passes)==2 and not strong_control)
state=('INCOMPLETE_SIMULATED_SMOKE' if SMOKE else
       'INCOMPLETE_MISSING_DATA_OR_CODEC' if not complete_real else
       'ADVANCE_WORKLOAD_TO_NOVEL_MECHANISM_DESIGN' if workload_go else
       'NO_STRONG_UNEXPLAINED_TEMPORAL_OPPORTUNITY')
decision={
 'notebook':'SGX_NB87','research_stage':'domain opportunity discovery, not an original compressor',
 'data_type':'SIMULATED_SMOKE_ONLY' if SMOKE else 'REAL_PUBLIC_DATA_WHERE_DOWNLOADED',
 'sources_attempted':len(provenance),'sources_succeeded':sum(x['status']=='REAL_UCI' for x in provenance),
 'families_measured':len(families),'streams_measured':len(policy_rows),
 'zstandard_available':bool(zstd),'bit_exact_checks_passed':no_mismatch,'edge_checks':edge_count,
 'temporal_families_passing_predefined_gate':[r['family'] for r in family_passes],
 'simulated_negative_control_has_temporal_signature':strong_control,
 'research_gate_state':state,'workload_opportunity_go':workload_go,
 'Q1_originality_proven':False,'Q1_manuscript_ready':False,
 'interpretation':'Even a positive result would only identify a data domain where known temporal transforms help. It does not establish a novel algorithm or a benefit over Gorilla/Chimp/Elf/FPC/ALP.',
 'limitations':['Only two distinct chronological families; insufficient for Q1 generalization.','Best preselected temporal transformation is prior art.','Cross-library native comparison not included.','Recorded stream-wise calibration trials must be charged for one-shot throughput.','Mixed types and source parsing define the exact float64 benchmark representation.']
}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
protocol={'methods':ALL_METHODS,'frame':'22 bytes: 5-byte magic, 1-byte method id, 8-byte count, 8-byte compressed payload length','split':'first 25% chronological calibration; remaining 75% held-out; grid control has no temporal claim',
          'input_bit_exact_definition':'little-endian float64 bytes after published source parsing','random_seed':SEED,
          'selection':'best complete-frame bytes on calibration within each method class','oracle':'heldout best-on-result only, never used for deployed policy',
          'primary_gate': 'each real temporal family >=5% net gain and >=2 percentage points shuffle-control lift, worst per-stream >= -10%, negative non-temporal grid lacks signature, zero errors',
          'no_novelty_claim':True}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
(OUT/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'zstandard':getattr(zstd,'__version__',None),'smoke':SMOKE},indent=2))

target=ROOT/(OUTPUT_NAME+'_results.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file():z.write(p,p.name)
print('\nRESEARCH GATE:',state)
print('FAMILY SUMMARY:')
for r in families:
    print(r['family'],'|',r['streams'],'streams | gain %.2f%%'%r['net_gain_percent'],'| sequence lift %.2f pp'%r['sequential_lift_pp'])
print('RESULT ZIP:',target,'bytes:',target.stat().st_size)
try:
    from google.colab import files
    files.download(str(target))
except ImportError:
    pass

