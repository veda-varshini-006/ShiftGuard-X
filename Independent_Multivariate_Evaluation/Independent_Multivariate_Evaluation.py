#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB63 — Independent Multivariate Evaluation

**Question.** Does the frozen cross-column candidate generalize to newly selected, public multivariate numerical datasets?

Datasets: **UCI Yacht Hydrodynamics** (ID 243), **UCI Energy Efficiency** (ID 242), and **UCI Parkinsons Telemonitoring** (ID 189). Data are downloaded directly from UCI, with source URLs, SHA-256 hashes, shapes, and parsed-matrix hashes saved. Failed downloads remain **explicit failures**: the notebook never replaces real datasets with generated data.

The first 25% of rows fits codec/reference/mask choices. The remaining 75% tests complete file sizes, exact float64 bit-pattern decoding, encode/decode times, and codec-selection time. Includes row-major and columnwise Zlib/LZMA/Zstd (if installed), plus the earlier exact decimal-lattice method. Selection requires a **10% calibration margin** before a joint method is used. A hindsight-best baseline is reported **for diagnosis only**, not as a deployable policy.

**Limitations:** no head-to-head native ALP/Sprintz implementation; no assertion of algorithmic originality; parsed IEEE-754 values are preserved, not the original CSV/XLSX textual formatting. Output: `SGX_NB63_results.zip`.

Sources: https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics ; https://archive.ics.uci.edu/dataset/242/energy+efficiency ; https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring
"""

# Install the optional Zstandard comparator only when running in Colab.
# If installation is unavailable, the experiment records that Zstandard was not tested.
import sys, importlib.util, subprocess
from pathlib import Path
if Path('/content').is_dir() and importlib.util.find_spec('zstandard') is None:
    attempt=subprocess.run([sys.executable,'-m','pip','install','-q','zstandard'],
                           capture_output=True,text=True,timeout=120)
    print('Optional zstandard installation:', 'PASS' if attempt.returncode==0 else 'SKIPPED (unavailable)')


OUT_NAME = 'SGX_NB63_outputs'


import io, os, csv, json, math, time, struct, zlib, lzma, hashlib, statistics, zipfile, platform, sys, urllib.request, urllib.error, traceback
from pathlib import Path
import numpy as np

RUN_ROOT = Path('/content') if Path('/content').is_dir() else Path('/mnt/data')
RUN_ROOT.mkdir(exist_ok=True,parents=True)
OUT = RUN_ROOT / OUT_NAME
OUT.mkdir(exist_ok=True,parents=True)
TEST_MODE = os.environ.get('SGX_LOCAL_SMOKE', '0') == '1'
try:
    import zstandard as zstd
except ImportError:
    zstd = None

METHODS = {'row_zlib':0,'row_lzma':1,'row_zstd':2,'col_zlib':3,'col_lzma':4,'col_zstd':5,
           'joint_xor_zlib':6,'joint_xor_zstd':7,'joint_hybrid_zlib':8,'joint_hybrid_zstd':9,'col_lattice':10}
REVERSE = {v:k for k,v in METHODS.items()}
MAGIC = b'SXJ1'
SCALES = [1,10,100,1000,10000]

def write_csv(filename,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k in row.keys()))
    with (OUT/filename).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

def save_json(filename,obj): (OUT/filename).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
def package(number):
    p=RUN_ROOT/f'SGX_NB{number}_results.zip'
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob('*')):
            if f.is_file(): z.write(f,f.relative_to(OUT))
    print('RESULTS_ZIP:',p,'BYTES:',p.stat().st_size)
    try:
        from google.colab import files
        files.download(str(p))
    except ImportError: pass
    return p

def to_matrix(a):
    arr=np.asarray(a,dtype='<f8')
    if arr.ndim!=2 or arr.shape[1]<2: raise ValueError('Expected 2D float64 matrix with at least 2 columns')
    if arr.shape[0]>100000000 or arr.shape[1]>65535: raise ValueError('Matrix dimensions out of range')
    return np.ascontiguousarray(arr)

def bytes_matrix(a): return to_matrix(a).tobytes(order='C')
def words_matrix(a): return to_matrix(a).view('<u8').reshape(a.shape)
def col_bytes(words,j): return np.ascontiguousarray(words[:,j],dtype='<u8').tobytes()

def compress(raw,which):
    if which=='zlib':return zlib.compress(raw,6)
    if which=='lzma':return lzma.compress(raw,preset=1)
    if which=='zstd':
        if zstd is None:raise RuntimeError('zstandard not installed')
        return zstd.ZstdCompressor(level=3).compress(raw)
    raise ValueError('Unknown compressor: '+which)

def decompress(blob,which):
    if which=='zlib':return zlib.decompress(blob)
    if which=='lzma':return lzma.decompress(blob)
    if which=='zstd':
        if zstd is None:raise RuntimeError('zstandard not installed')
        return zstd.ZstdDecompressor().decompress(blob)
    raise ValueError('Unknown compressor: '+which)

# The decimal-lattice baseline is ported from SGX NB52 and retains its frame format.
# It is NOT a claim of novelty; it captures the strongest previous SGX direction.
def bitpack(values,width):
    acc=0; pending=0; out=bytearray()
    for v in values:
        v=int(v)
        if v<0 or (width and v >=(1<<width)) or (width==0 and v): raise ValueError('Packed value out of bounds')
        acc |= v << pending
        pending += width
        while pending>=8:
            out.append(acc&255);acc >>=8;pending-=8
    if pending:out.append(acc&255)
    return bytes(out)

def bitunpack(buf,width,count):
    acc=0; pending=0; it=iter(buf); out=[]
    for _ in range(count):
        while pending<width:
            try: acc |= next(it) << pending
            except StopIteration: raise ValueError('Truncated packing')
            pending += 8
        out.append(acc & ((1<<width)-1));acc >>=width;pending -=width
    if any(it): raise ValueError('Trailing packed bytes')
    return out

def lattice_pack(raw,scale):
    if len(raw)%8: raise ValueError('Unaligned raw values')
    values=np.frombuffer(raw,dtype='<f8')
    accept=[];integer=[];exceptions=[]
    for i,v in enumerate(values):
        v=float(v);found=False;k=0
        if math.isfinite(v) and -(2**31) <= v*scale <= 2**31-1:
            candidate=int(round(v*scale))
            if -(2**31)<=candidate<=2**31-1:
                try: found=struct.pack('<d',float(candidate)/scale)==raw[8*i:8*i+8]
                except OverflowError: found=False
                if found:k=candidate
        if found:accept.append(1);integer.append(k)
        else:accept.append(0);exceptions.append(raw[8*i:8*i+8])
    minimum=min(integer) if integer else 0
    delta=[x-minimum for x in integer]
    width=max(delta,default=0).bit_length()
    streams=[zlib.compress(bitpack(accept,1),6),zlib.compress(bitpack(delta,width),6),zlib.compress(b''.join(exceptions),6)]
    return b'QPK1'+struct.pack('<BIIqBIII',SCALES.index(scale),len(values),len(integer),minimum,width,*[len(c) for c in streams])+b''.join(streams)

def lattice_unpack(blob):
    if blob[:4]!=b'QPK1' or len(blob)<34:raise ValueError('Invalid lattice header')
    k,n,naccepted,minimum,width,p,q,r=struct.unpack('<BIIqBIII',blob[4:34])
    if k>=len(SCALES) or width>64 or naccepted>n:raise ValueError('Invalid lattice header fields')
    if len(blob)!=34+p+q+r:raise ValueError('Lattice length mismatch')
    offset=34
    mask=bitunpack(zlib.decompress(blob[offset:offset+p]),1,n);offset+=p
    delta=bitunpack(zlib.decompress(blob[offset:offset+q]),width,naccepted);offset+=q
    exceptions=zlib.decompress(blob[offset:offset+r])
    if sum(mask)!=naccepted or len(exceptions)!=(n-naccepted)*8:raise ValueError('Invalid lattice exceptions')
    out=bytearray();di=0;ei=0
    for keep in mask:
        if keep:
            out.extend(struct.pack('<d',(delta[di]+minimum)/SCALES[k]));di+=1
        else:
            out.extend(exceptions[ei:ei+8]);ei+=8
    return bytes(out)

def encode(a,spec):
    a=to_matrix(a);n,d=a.shape
    name=spec['mode']
    if name not in METHODS:raise ValueError('Unknown mode '+name)
    ref=int(spec.get('ref',0))
    if not 0<=ref<d:raise ValueError('Reference index out of range')
    header=MAGIC+struct.pack('<IHHH',n,d,METHODS[name],ref)
    isrow=name.startswith('row_')
    if isrow:
        body=compress(a.tobytes(order='C'),name[4:])
        return header+struct.pack('<Q',len(body))+body
    w=words_matrix(a)
    if name=='col_lattice':
        scales=spec.get('scales')
        if scales is None or len(scales)!=d or any(x not in SCALES for x in scales):raise ValueError('Lattice scale metadata missing')
        blocks=[lattice_pack(col_bytes(w,j),int(scales[j])) for j in range(d)]
        mask=b''
    else:
        alg=name.rsplit('_',1)[-1]
        if name.startswith('joint_'):
            anchor=w[:,ref]
            if name.startswith('joint_hybrid_'):
                switches=spec.get('switches')
                if switches is None or len(switches)!=d or any(x not in (0,1) for x in switches):raise ValueError('Hybrid switch metadata missing')
                if switches[ref]:raise ValueError('Reference column cannot be XOR-transformed')
                mask=bytes(switches)
            else:
                switches=[int(j!=ref) for j in range(d)];mask=b''
        else:
            switches=[0]*d;mask=b''
        blocks=[]
        for j in range(d):
            raw=col_bytes(w,j)
            if switches[j]:raw=np.bitwise_xor(w[:,j],anchor).astype('<u8').tobytes()
            blocks.append(compress(raw,alg))
    parts=[header,mask]
    for b in blocks:parts.append(struct.pack('<Q',len(b)))
    parts.extend(blocks)
    return b''.join(parts)

def decode(blob):
    if len(blob)<14 or blob[:4]!=MAGIC:raise ValueError('Invalid or truncated matrix frame')
    n,d,idx,ref=struct.unpack('<IHHH',blob[4:14])
    if idx not in REVERSE or d<2 or n>100000000 or ref>=d:raise ValueError('Invalid dimensions or codec')
    name=REVERSE[idx];at=14
    if name.startswith('row_'):
        if len(blob)<at+8:raise ValueError('Truncated row payload')
        (length,)=struct.unpack('<Q',blob[at:at+8]);at+=8
        if len(blob)-at!=length:raise ValueError('Row length mismatch')
        raw=decompress(blob[at:],name[4:])
        if len(raw)!=n*d*8:raise ValueError('Row decoded length')
        return np.frombuffer(raw,dtype='<f8').copy().reshape(n,d)
    switches=bytes(d)
    if name.startswith('joint_hybrid_'):
        if len(blob)<at+d:raise ValueError('Truncated hybrid switches')
        switches=blob[at:at+d];at+=d
        if any(x not in (0,1) for x in switches) or switches[ref]:raise ValueError('Invalid hybrid switches')
    if len(blob)<at+d*8:raise ValueError('Truncated column lengths')
    sizes=struct.unpack('<'+'Q'*d,blob[at:at+d*8]);at+=d*8
    cols=[]
    for j,size in enumerate(sizes):
        if size>len(blob)-at:raise ValueError('Truncated compressed column')
        part=blob[at:at+size];at+=size
        if name=='col_lattice':raw=lattice_unpack(part)
        else:raw=decompress(part,name.rsplit('_',1)[-1])
        if len(raw)!=n*8:raise ValueError('Decoded column size mismatch')
        cols.append(np.frombuffer(raw,dtype='<u8').copy())
    if at!=len(blob):raise ValueError('Trailing payload bytes')
    if name.startswith('joint_xor_'):switches=bytes([int(j!=ref) for j in range(d)])
    if name.startswith('joint_'):
        reference=cols[ref]
        for j in range(d):
            if switches[j]:cols[j]=np.bitwise_xor(cols[j],reference)
    uw=np.ascontiguousarray(np.column_stack(cols),dtype='<u8')
    return uw.view('<f8').reshape(n,d)

def select_scales(cal):
    words=words_matrix(cal)
    chosen=[]
    for j in range(words.shape[1]):
        raw=col_bytes(words,j)
        lengths=[len(lattice_pack(raw,scale)) for scale in SCALES]
        chosen.append(SCALES[int(np.argmin(lengths))])
    return chosen

def reference_rank(cal,codec='zlib',max_refs=4):
    w=words_matrix(cal);d=w.shape[1]
    normal=[len(compress(col_bytes(w,j),codec)) for j in range(d)]
    scored=[]
    for ref in range(d):
        gain=0
        for j in range(d):
            if j==ref:continue
            transformed=np.bitwise_xor(w[:,j],w[:,ref]).astype('<u8').tobytes()
            gain += normal[j]-len(compress(transformed,codec))
        scored.append((gain,ref))
    return [r for _,r in sorted(scored,key=lambda p:(-p[0],p[1]))[:max_refs]]

def switches_for(cal,ref,codec='zlib'):
    w=words_matrix(cal);d=w.shape[1];flags=[]
    for j in range(d):
        if j==ref:flags.append(0);continue
        raw=col_bytes(w,j)
        xor=np.bitwise_xor(w[:,j],w[:,ref]).astype('<u8').tobytes()
        flags.append(int(len(compress(xor,codec)) < len(compress(raw,codec))))
    return flags

def candidates(cal):
    bases=[]; joints=[]
    codecs=['zlib','lzma']+(['zstd'] if zstd is not None else [])
    for codec in codecs:
        bases.extend([{'mode':'row_'+codec},{'mode':'col_'+codec}])
    bases.append({'mode':'col_lattice','scales':select_scales(cal)})
    # Reference rank is based ONLY on calibration rows.
    for codec in ['zlib']+(['zstd'] if zstd is not None else []):
        for ref in reference_rank(cal,'zlib',max_refs=min(4,cal.shape[1])):
            joints.append({'mode':'joint_xor_'+codec,'ref':ref})
            joints.append({'mode':'joint_hybrid_'+codec,'ref':ref,'switches':switches_for(cal,ref,codec)})
    return bases,joints

def choose_policy(cal,margin=0.10):
    st=time.perf_counter()
    bases,joints=candidates(cal)
    records=[]
    for spec in bases+joints:
        b=encode(cal,spec)
        if bytes_matrix(decode(b))!=bytes_matrix(cal):raise AssertionError('Calibration reconstruction mismatch')
        records.append((len(b),spec))
    base=min((r for r in records if not r[1]['mode'].startswith('joint_')),key=lambda p:p[0])
    joint=min((r for r in records if r[1]['mode'].startswith('joint_')),key=lambda p:p[0])
    chosen=joint if joint[0] <= (1-margin)*base[0] else base
    return {'baseline':base[1],'joint_best':joint[1],'chosen':chosen[1],
            'baseline_cal_bytes':base[0],'joint_cal_bytes':joint[0],
            'fit_ms':(time.perf_counter()-st)*1000,'tested_candidates':len(records),
            'margin':margin}

def measure(a,spec,reps=3):
    original=bytes_matrix(a); size=None;enc=[];dec=[]
    for k in range(reps+1):
        ts=time.perf_counter_ns();blob=encode(a,spec);te=(time.perf_counter_ns()-ts)/1e6
        ts=time.perf_counter_ns();got=decode(blob);td=(time.perf_counter_ns()-ts)/1e6
        if bytes_matrix(got)!=original:raise AssertionError('Bit mismatch '+repr(spec))
        if size is None:size=len(blob)
        if k:enc.append(te);dec.append(td)
    return {'bytes':size,'bpv':8*size/(a.shape[0]*a.shape[1]) if a.shape[0] else None,
            'encode_ms':statistics.median(enc),'decode_ms':statistics.median(dec),'bit_exact':True}

def edge_cases():
    bit_patterns=np.array([0,0x8000000000000000,0x7ff8000000000001,0x7ff8000000000033,0x7ff0000000000000,
                 0xfff0000000000000,0x0000000000000001,0xffffffffffffffff],dtype='<u8')
    a=np.tile(bit_patterns,(9,1)).view('<f8')
    cases={'special_patterns':a,'empty':np.empty((0,3),dtype='<f8'),
           'integers':np.column_stack((np.arange(33),np.arange(33),np.arange(33))),
           'fractional':np.column_stack((np.arange(37)/10,np.arange(37)/10+1,np.arange(37)/10))}
    outcomes=[]
    for label,x in cases.items():
        x=to_matrix(x)
        try:scale=select_scales(x)
        except Exception as e:scale=None
        specs=[{'mode':'row_zlib'},{'mode':'col_zlib'},{'mode':'joint_xor_zlib','ref':0},
               {'mode':'joint_hybrid_zlib','ref':0,'switches':[0]+[1]*(x.shape[1]-1)}]
        if scale is not None:specs.append({'mode':'col_lattice','scales':scale})
        for spec in specs:
            try:
                packed=encode(x,spec);correct=bytes_matrix(decode(packed))==bytes_matrix(x)
                status='PASS' if correct else 'FAIL'
            except Exception as exc:status='ERROR:'+repr(exc)
            outcomes.append({'case':label,'mode':spec['mode'],'status':status})
    if any(x['status']!='PASS' for x in outcomes):raise AssertionError('Edge-case tests not all passing: '+repr(outcomes))
    return outcomes

def download_bytes(url,timeout=45):
    req=urllib.request.Request(url,headers={'User-Agent':'ShiftGuardX-research/1.0'})
    with urllib.request.urlopen(req,timeout=timeout) as res: return res.read()

def environment():
    save_json('environment.json',{'python':sys.version,'numpy':np.__version__,'zstandard':None if zstd is None else zstd.__version__,
                'platform':platform.platform(),'test_mode':TEST_MODE,
                'codec':'SGX joint XOR with explicit ref/mask, complete self-describing frame',
                'novelty':'not established','comparability':'CPU Python generic codecs, no native ALP/Sprintz benchmark'})


import pandas as pd
write_csv('edge_cases.csv',edge_cases());environment()
SOURCES={
 'yacht_hydrodynamics':{'url':'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data','kind':'whitespace'},
 'energy_efficiency':{'url':'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx','kind':'excel'},
 'parkinsons_telemonitoring':{'url':'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data','kind':'csv'}}
provenance=[];summary=[];benchmarks=[];errors=[]

def load_external(name,source):
    if TEST_MODE:
        from sklearn.datasets import load_wine
        raw_matrix=load_wine().data[:120,:min(8,load_wine().data.shape[1])]
        return to_matrix(raw_matrix),{'source_status':'SMOKE_TEST_SKLEARN_NOT_UCI','source_sha256':'SMOKE_ONLY','download_bytes':0}
    body=download_bytes(source['url'],timeout=50)
    digest=hashlib.sha256(body).hexdigest()
    if source['kind']=='whitespace':
        matrix=np.loadtxt(io.StringIO(body.decode('utf-8')),dtype=np.float64)
    elif source['kind']=='excel':
        df=pd.read_excel(io.BytesIO(body),engine='openpyxl')
        matrix=df.apply(pd.to_numeric,errors='coerce').dropna(axis=1,how='all').dropna(axis=0,how='all').to_numpy(dtype=np.float64)
    elif source['kind']=='csv':
        df=pd.read_csv(io.BytesIO(body))
        matrix=df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    else:raise ValueError('Unknown source kind')
    if matrix.ndim!=2 or matrix.shape[0]<64 or matrix.shape[1]<2:raise ValueError('Unexpected dataset dimensions '+str(matrix.shape))
    if not np.all(np.isfinite(matrix)):raise ValueError('Unexpected non-finite value in external dataset; no silent missing-value substitution')
    return to_matrix(matrix),{'source_status':'REAL_UCI','source_sha256':digest,'download_bytes':len(body)}

for name,src in SOURCES.items():
    try:
        full,sourceinfo=load_external(name,src)
        n,d=full.shape
        provenance.append({'dataset':name,'url':src['url'],'n':n,'d':d,
                           'parsed_matrix_sha256':hashlib.sha256(bytes_matrix(full)).hexdigest(),**sourceinfo})
        cut=max(32,n//4)
        cal=full[:cut];test=full[cut:]
        fitted=choose_policy(cal,margin=0.10)
        bas=fitted['baseline'];joint=fitted['joint_best'];selected=fitted['chosen']
        for label,spec in [('calibration_baseline',bas),('best_joint_calibration',joint),('frozen_policy',selected)]:
            got=measure(test,spec,reps=5)
            benchmarks.append({'dataset':name,'variant':label,'mode':spec['mode'],
                               'reference_column':spec.get('ref',''),'n_holdout':len(test),'d':d,**got})
        a=benchmarks[-3]['bytes'];j=benchmarks[-2]['bytes'];b=benchmarks[-1]['bytes']
        # For honest hindsight baseline diagnostics, ALL generic/decimal baselines are evaluated
        # on test, but only the calibration-selected baseline participates in the deployment comparison.
        base_candidates,_=candidates(cal)
        per_baseline=[(spec['mode'],measure(test,spec,reps=1)['bytes']) for spec in base_candidates]
        oracle_name,oracle_size=min(per_baseline,key=lambda x:x[1])
        summary.append({'dataset':name,'status':'SMOKE_ONLY' if TEST_MODE else 'REAL_UCI',
            'n_calibration':len(cal),'n_holdout':len(test),'d':d,'selected_mode':selected['mode'],
            'selected_ref':selected.get('ref',''),'calibration_baseline_mode':bas['mode'],
            'holdout_baseline_bytes':a,'holdout_best_joint_bytes':j,'holdout_policy_bytes':b,
            'gain_pct_deployable_baseline':100*(a-b)/a,'hindsight_best_nonjoint_mode':oracle_name,
            'hindsight_best_nonjoint_bytes':oracle_size,'gain_pct_hindsight_oracle':100*(oracle_size-b)/oracle_size,
            'fit_ms':fitted['fit_ms'],'selected_encode_ms':benchmarks[-1]['encode_ms'],
            'selected_decode_ms':benchmarks[-1]['decode_ms'],
            'compression_plus_fit_ms':fitted['fit_ms']+benchmarks[-1]['encode_ms'],
            'holdout_sha256':hashlib.sha256(bytes_matrix(test)).hexdigest()})
        print(name,'selected',selected['mode'],'savings_vs_calibrated_baseline',round(100*(a-b)/a,2),'%')
    except Exception as exc:
        errors.append({'dataset':name,'url':src['url'],'error':repr(exc),'traceback':traceback.format_exc()[-4000:]})
        print('DATASET FAILURE (no synthetic fallback):',name,repr(exc))
write_csv('dataset_provenance.csv',provenance);write_csv('frozen_external_decisions.csv',summary)
write_csv('heldout_codec_measurements.csv',benchmarks);write_csv('download_or_codec_errors.csv',errors)
real=all(r['status']=='REAL_UCI' for r in summary)
valid=(len(summary)==len(SOURCES) and not errors and real)
if summary:
    base_total=sum(x['holdout_baseline_bytes'] for x in summary)
    policy_total=sum(x['holdout_policy_bytes'] for x in summary)
    gain=100*(base_total-policy_total)/base_total
else:gain=None
state='COMPLETED_REAL_EXTERNAL' if valid else 'SMOKE_ONLY' if TEST_MODE and len(summary)==len(SOURCES) else 'INCOMPLETE'
save_json('research_gate.json',{'state':state,'requested_external_families':list(SOURCES),'successful_families':[x['dataset'] for x in summary],
    'total_external_heldout_gain_pct':gain,'joint_selected':sum(x['selected_mode'].startswith('joint_') for x in summary),
    'regressed_families':sum(x['gain_pct_deployable_baseline']<0 for x in summary),
    'test_mode':TEST_MODE,'no_synthetic_substitution':not TEST_MODE,
    'conclusion':'If zero joint selection or nonpositive savings, redesign. Positive results alone do not prove novelty.',
    'prior_art':['ALP 2024 SIGMOD decimal/sampling','Sprintz 2018 multivariate predictive compression',
                 'FPC and predictive XOR floating-point lossless compression'],
    'limits':['No native ALP/Sprintz head-to-head; decoder verifies parsed float64 bytes, not original CSV or XLSX text.',
              'Selections are made on first 25% of each dataset, and evaluated on the remaining 75%.',
              'These public families were selected before the notebook was run; not a preregistered blind study.']})
print('NB63',state,'results',len(summary),'errors',len(errors),'gain',gain)
package(63)

