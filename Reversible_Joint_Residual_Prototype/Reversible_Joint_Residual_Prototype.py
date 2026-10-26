#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB62 — Reversible Joint-Residual Prototype

**Question.** Does the proposed bit-exact joint-XOR coding of float64 matrices beat strong independent, row-major and prior SGX decimal-lattice baselines on held-out rows?

The self-describing SGX joint frame encodes matrix shape, codec, XOR reference column, a hybrid routing mask when needed, all compressed section sizes and all column bodies. It supports signed zero, NaN payloads and other special bit patterns, verified **bit-for-bit**.

The prototype tests **joint XOR** and **hybrid raw/XOR coding**, with the reference column and any routing mask learned only from the *first 25%* of rows. It falls back to the calibration-selected baseline unless a joint mode is at least 10% smaller on calibration. The last 75% is never used to fit the method. An independent-column decimal-lattice implementation from NB52 is included as a strong prior-SGX baseline.

This is a **candidate mechanism, not a novelty claim**: predictive XOR and multivariate compression are established research areas. All code is standalone and exports `SGX_NB62_results.zip`.
"""

# Install the optional Zstandard comparator only when running in Colab.
# If installation is unavailable, the experiment records that Zstandard was not tested.
import sys, importlib.util, subprocess
from pathlib import Path
if Path('/content').is_dir() and importlib.util.find_spec('zstandard') is None:
    attempt=subprocess.run([sys.executable,'-m','pip','install','-q','zstandard'],
                           capture_output=True,text=True,timeout=120)
    print('Optional zstandard installation:', 'PASS' if attempt.returncode==0 else 'SKIPPED (unavailable)')


OUT_NAME = 'SGX_NB62_outputs'


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


from sklearn.datasets import load_diabetes,load_wine,load_breast_cancer,load_digits
write_csv('edge_cases.csv',edge_cases());environment()
DATA=[('diabetes',load_diabetes().data),('wine_classification',load_wine().data),
      ('breast_cancer',load_breast_cancer().data),('digits',load_digits().data)]
summary=[];ablation=[];errors=[]
for name,matrix in DATA:
    try:
        matrix=to_matrix(matrix)
        if TEST_MODE:matrix=matrix[:min(170,len(matrix)),:min(12,matrix.shape[1])]
        cut=max(32,len(matrix)//4)
        cal=matrix[:cut];hold=matrix[cut:]
        model=choose_policy(cal,margin=0.10)
        base=model['baseline'];joint=model['joint_best'];selected=model['chosen']
        # Held-out evaluation of choices frozen using ONLY the calibration prefix.
        bm=measure(hold,base,3);jm=measure(hold,joint,3);sm=measure(hold,selected,3)
        bases,_=candidates(cal)
        baseline_oracle=min(measure(hold,b,1)['bytes'] for b in bases)  # diagnostic hindsight; never used for policy
        summary.append({'dataset':name,'n_cal':len(cal),'n_holdout':len(hold),'d':matrix.shape[1],
             'cal_baseline_size':model['baseline_cal_bytes'],'cal_joint_size':model['joint_cal_bytes'],
             'selected_mode':selected['mode'],'selected_ref':selected.get('ref',''),
             'baseline_selected_mode':base['mode'],'joint_mode':joint['mode'],
             'heldout_selected_bytes':sm['bytes'],'heldout_calibration_baseline_bytes':bm['bytes'],
             'heldout_joint_bytes':jm['bytes'],'heldout_baseline_oracle_bytes_diagnostic_only':baseline_oracle,
             'saving_vs_cal_baseline_pct':100*(bm['bytes']-sm['bytes'])/bm['bytes'],
             'saving_vs_hindsight_oracle_pct':100*(baseline_oracle-sm['bytes'])/baseline_oracle,
             'policy_fit_ms':model['fit_ms'],'policy_encode_ms':sm['encode_ms'],
             'policy_decode_ms':sm['decode_ms'],'holdout_sha256':hashlib.sha256(bytes_matrix(hold)).hexdigest()})
        # Ablation: compare the frozen choices, including self-describing routing/reference metadata.
        for label,spec in [('calibration_baseline',base),('calibration_best_joint',joint),('frozen_policy',selected)]:
            m=measure(hold,spec,1)
            ablation.append({'dataset':name,'variant':label,'mode':spec['mode'],'ref':spec.get('ref',''),**m})
        print(name, 'policy=',selected['mode'],'bytes=',sm['bytes'],'baseline=',bm['bytes'])
    except Exception as exc:
        errors.append({'dataset':name,'error':repr(exc),'traceback':traceback.format_exc()[-4000:]})
write_csv('frozen_decisions.csv',summary);write_csv('heldout_ablation.csv',ablation);write_csv('errors.csv',errors)
if summary:
    bas=sum(r['heldout_calibration_baseline_bytes'] for r in summary)
    pol=sum(r['heldout_selected_bytes'] for r in summary)
    savings=100*(bas-pol)/bas
else:savings=None
save_json('decision.json',{'state':'COMPLETED' if len(summary)==len(DATA) and not errors else 'INCOMPLETE',
    'streams':len(summary),'joint_selected':sum('joint' in r['selected_mode'] for r in summary),
    'aggregate_savings_pct_against_deployable_baseline':savings,
    'scientific_conclusion':'exploratory algorithm test, cross-column XOR is not new',
    'known_data':'These sklearn dataset families appeared earlier in SGX; test new datasets in NB63',
    'smoke_test':TEST_MODE})
print('NB62:',len(summary),'datasets; errors:',len(errors),'aggregate_savings_pct=',savings)
package(62)

