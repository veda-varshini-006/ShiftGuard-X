#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB119 — Real Arithmetic-Path Compression Cost Gate

**Decisive question:** Does observed IEEE-754 arithmetic-path diversity (NB118) translate into real, bit-exact **byte savings**, including selector and fallback information, on frozen February data? No previous ZIP upload required.

**Scientific boundaries:** This is a selected nine-column float64 numeric matrix plus VendorID, **not** a complete original Parquet file. This is a predictor-family engineering benchmark **not a novelty proof**. Comparison includes single-path, best January single predictor, January-frozen vendor predictor, two-path selector (prior art), and five-path selector. Five path choices are fixed before examining any February outcomes. Read the final prior-art gate even if there is a compression gain.
"""

"""
## 1. Dependencies and locked real datasets
"""

import os,sys,subprocess,platform,hashlib,json,csv,math,time,zipfile,traceback
from pathlib import Path
import numpy as np
try:
 import requests
except ImportError:
 subprocess.check_call([sys.executable,'-m','pip','-q','install','requests'])
 import requests
try:
 import pyarrow.parquet as pq
except ImportError:
 subprocess.check_call([sys.executable,'-m','pip','-q','install','pyarrow'])
 import pyarrow.parquet as pq

OUT=Path('/content/SGX_NB119_outputs');OUT.mkdir(parents=True,exist_ok=True)
DATA=Path('/content/SGX_NB119_public_data');DATA.mkdir(parents=True,exist_ok=True)
MAX_ROWS=125_000
SOURCES={
 'yellow_2024_01': 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet',
 'yellow_2024_02': 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet',
 'green_2024_02': 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-02.parquet',
}
BASE_FIELDS=['fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge']
OPTIONAL_FIELDS=['congestion_surcharge','Airport_fee'] # valid for the defined 2024 slice; no 2025 CBD fee
PATH_NAMES=['left_to_right','reverse','balanced','tip_last','taxes_fees_first']
assert len(set(PATH_NAMES))==5
print('Study:',len(SOURCES),'real sources; max rows per source:',MAX_ROWS)
print('NumPy:',np.__version__,'PyArrow:',pq.__version__ if hasattr(pq,'__version__') else 'present')

try:
 import zstandard as zstd
except ImportError:
 subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "zstandard"])
 import zstandard as zstd
C=zstd.ZstdCompressor(level=9);D=zstd.ZstdDecompressor()


"""
## 2. Identical five IEEE-754 evaluation orders as NB118
"""

def ieee_bits(x):
    a=np.asarray(x,dtype=np.float64)
    return np.ascontiguousarray(a).view(np.uint64)

def sequential_sum(terms):
    s=np.zeros_like(terms[0],dtype=np.float64)
    for t in terms:
        s=np.add(s,t,dtype=np.float64)
    return s

def balanced_sum(terms):
    v=list(terms)
    while len(v)>1:
        v=[np.add(v[i],v[i+1],dtype=np.float64) if i+1<len(v) else v[i]
           for i in range(0,len(v),2)]
    return v[0]

def fixed_paths(components):
    # Each of the 8 named inputs is float64. Selection is fixed in notebook source.
    names=BASE_FIELDS+OPTIONAL_FIELDS
    v=[components[n] for n in names]
    assert len({len(x) for x in v})==1
    path0=sequential_sum(v)
    path1=sequential_sum(list(reversed(v)))
    path2=balanced_sum(v)
    # tip_amount is placed at the end. No target values are inspected.
    path3=sequential_sum([components[n] for n in names if n!='tip_amount']+[components['tip_amount']])
    # deterministic charges-first ordering of the same source components
    path4=sequential_sum([components[n] for n in ['mta_tax','improvement_surcharge','congestion_surcharge','Airport_fee','extra','tolls_amount','fare_amount','tip_amount']])
    return dict(zip(PATH_NAMES,[path0,path1,path2,path3,path4]))

def wilson(k,n,z=1.95996398454005):
    if n==0:return [None,None]
    p=k/n; den=1+z*z/n;mid=(p+z*z/(2*n))/den
    rad=(z/den)*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return [round(max(0,mid-rad),9),round(min(1,mid+rad),9)]

def analyze_real_rows(components,target,vendor=None,dataset='unnamed'):
    # Explicit IEEE-754 bit comparisons; +0.0 and -0.0 are different.
    paths=fixed_paths(components); y=ieee_bits(target)
    pi={name:ieee_bits(result) for name,result in paths.items()}
    p0=pi[PATH_NAMES[0]]
    disagreements=np.column_stack([pi[name]!=p0 for name in PATH_NAMES[1:]])
    exact=np.column_stack([pi[name]==y for name in PATH_NAMES])
    extra=(~exact[:,0]) & exact[:,1:].any(axis=1)
    anymatch=exact.any(axis=1)
    n=len(y);k=int(extra.sum());disagree=int(disagreements.any(axis=1).sum())
    result={
      'dataset':dataset,'qualified_rows':n,'paths':PATH_NAMES,
      'p0_exact_target':int(exact[:,0].sum()),'any_path_exact_target':int(anymatch.sum()),
      'any_path_disagrees_p0':disagree,
      'additional_exact_target_by_other_path':k,
      'additional_exact_fraction':k/n if n else 0.0,
      'additional_exact_CI95_Wilson':wilson(k,n),
      'unexplained_by_any_path':int((~anymatch).sum()),
      'each_path_exact_target':{name:int(exact[:,i].sum()) for i,name in enumerate(PATH_NAMES)},
      'each_path_disagrees_p0':{name:int((pi[name]!=p0).sum()) for name in PATH_NAMES},
      'per_vendor':[],
    }
    if vendor is not None:
      vv=np.asarray(vendor)
      for val in np.unique(vv):
        sel=vv==val; nn=int(sel.sum());kk=int(extra[sel].sum())
        if nn:
          result['per_vendor'].append({'vendor':str(val),'rows':nn,'additional_exact':kk,'rate':kk/nn})
    # Minimal examples: public numerical amounts only. No dates/locations/record IDs.
    examples=[]
    for i in np.flatnonzero(extra)[:12]:
      examples.append({'sample_offset':int(i),'recorded_total':float(target[i]),
         'path0':float(paths[PATH_NAMES[0]][i]),
         'matching_alternatives':[name for name in PATH_NAMES[1:] if pi[name][i]==y[i]],
         'component_values':{name:float(components[name][i]) for name in BASE_FIELDS+OPTIONAL_FIELDS}})
    return result,examples


"""
## 3. Same official TLC download and parsing as NB118
"""

def download_verified(url,dest):
    if dest.exists() and dest.stat().st_size>0:
        print('Using local public data file',dest.name)
    else:
        tmp=dest.with_suffix(dest.suffix+'.part')
        try:
            # Request is streaming: downloads manageable on Colab; never substitute synthetic data on failure.
            with requests.get(url,stream=True,timeout=(20,180),headers={'User-Agent':'ShiftGuard-X-research-reproduction/1.0'}) as r:
                r.raise_for_status()
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(chunk_size=1<<20):
                        if chunk:f.write(chunk)
            tmp.replace(dest)
        finally:
            if tmp.exists():tmp.unlink()
    if dest.stat().st_size<1000:
        raise RuntimeError('Downloaded public source is unexpectedly small')
    with dest.open('rb') as fh:
        digest=hashlib.file_digest(fh,'sha256').hexdigest()
    return {'url':url,'file_bytes':dest.stat().st_size,'sha256':digest}

def read_prefix(path,max_rows=MAX_ROWS):
    pf=pq.ParquetFile(path)
    available=pf.schema_arrow.names
    missing_req=sorted(set(BASE_FIELDS+['total_amount'])-set(available))
    if missing_req:
        raise RuntimeError(f'Required NYC columns missing: {missing_req}; source schema: {available}')
    chosen=BASE_FIELDS+['total_amount']+[n for n in OPTIONAL_FIELDS if n in available]
    if 'VendorID' in available:chosen.append('VendorID')
    batches=[];total=0
    for b in pf.iter_batches(columns=chosen,batch_size=32768):
        df=b.to_pandas()
        needed=max_rows-total
        df=df.iloc[:needed].copy()
        batches.append(df)
        total+=len(df)
        if total>=max_rows:break
    if not batches:raise RuntimeError('Parquet contains no rows')
    import pandas as pd
    frame=pd.concat(batches,ignore_index=True)
    n=len(frame)
    zero_fill={}
    valid=np.ones(n,dtype=bool)
    # Missing optional column means no reported charge; declare zero assumption.
    # Present null optional cells are imputed as zero and counted: results are conditional on this assumption.
    comp={}
    for field in BASE_FIELDS+OPTIONAL_FIELDS:
        if field in frame:
            ar=pd.to_numeric(frame[field],errors='coerce').to_numpy(dtype=np.float64)
            if field in OPTIONAL_FIELDS:
                c=int(np.isnan(ar).sum())
                zero_fill[field]=c
                ar=np.where(np.isnan(ar),0.0,ar)
            comp[field]=ar
        else:
            assert field in OPTIONAL_FIELDS
            comp[field]=np.zeros(n,dtype=np.float64)
            zero_fill[field]='COLUMN_MISSING_ASSUMED_ZERO'
        valid &= np.isfinite(comp[field])
    target=pd.to_numeric(frame['total_amount'],errors='coerce').to_numpy(dtype=np.float64)
    valid &= np.isfinite(target)
    vendor=frame['VendorID'].to_numpy() if 'VendorID' in frame else None
    info={'source_rows_read':n,'nonfinite_or_missing_required_filtered':int((~valid).sum()),
          'optional_fees_zero_assumptions':zero_fill,'schema_columns':available,
          'parquet_row_count':pf.metadata.num_rows,'parquet_row_groups':pf.num_row_groups}
    comp={k:v[valid] for k,v in comp.items()}
    return comp,target[valid],vendor[valid] if vendor is not None else None,info


"""
## 4. Certified codec: file frames, metadata and all decoding
"""

# Binary format [uint32_le metadata_length][JSON metadata][compressor streams]
# All codecs use a deterministic little-endian uint64 bit view.
# Frame headers, selector, reconstruction metadata, and compressed bytes all count.
import struct
MAGIC='SGX119-v1'

def _bits(x):
    return np.ascontiguousarray(np.asarray(x,dtype=np.float64)).view(np.uint64)

def _lane_shuffle(words):
    b=np.ascontiguousarray(words,dtype='<u8').view(np.uint8).reshape(-1,8)
    return b.T.copy().tobytes()

def _lane_unshuffle(blob,n):
    ar=np.frombuffer(blob,dtype=np.uint8).reshape(8,n).T.copy()
    return ar.view('<u8').reshape(-1).copy()

def _encode_words(bits,mode):
    if mode=='raw':return C.compress(np.asarray(bits,dtype='<u8').tobytes())
    if mode=='lanes':return C.compress(_lane_shuffle(bits))
    raise ValueError('Unsupported byte packing')

def _decode_words(payload,n,mode):
    body=D.decompress(payload)
    assert len(body)==8*n
    if mode=='raw':return np.frombuffer(body,dtype='<u8').copy()
    if mode=='lanes':return _lane_unshuffle(body,n)
    raise ValueError('Unsupported packing')

def pack_bytes(header,streams):
    meta=dict(header);meta['magic']=MAGIC;meta['lengths']=[len(a) for a in streams]
    h=json.dumps(meta,sort_keys=True,separators=(',',':')).encode('utf-8')
    return len(h).to_bytes(4,'little')+h+b''.join(streams)

def unpack_bytes(frame):
    m=int.from_bytes(frame[:4],'little')
    assert 0<m<20000
    meta=json.loads(frame[4:4+m]);assert meta['magic']==MAGIC
    chunks=[];off=4+m
    for count in meta['lengths']:
        chunks.append(frame[off:off+count]);off+=count
    assert off==len(frame)
    return meta,chunks

def encode_common(comps,vendor):
    n=len(next(iter(comps.values())))
    blocks=[];desc=[]
    for name in BASE_FIELDS+OPTIONAL_FIELDS:
        bits=_bits(comps[name]); assert len(bits)==n
        candidates=[(_encode_words(bits,mode),mode) for mode in ['raw','lanes']]
        payload,mode=min(candidates,key=lambda x:(len(x[0]),x[1]))
        blocks.append(payload);desc.append({'name':name,'mode':mode})
    v=np.asarray(vendor,dtype='<i8');assert len(v)==n
    blocks.append(C.compress(v.tobytes()))
    frame=pack_bytes({'type':'parents','n':n,'columns':desc,'vendor_format':'int64'},blocks)
    return frame

def decode_common(frame):
    meta,blocks=unpack_bytes(frame);assert meta['type']=='parents'
    n=meta['n'];comps={}
    for i,col in enumerate(meta['columns']):
        comps[col['name']]=_decode_words(blocks[i],n,col['mode']).view(np.float64).copy()
    vendbytes=D.decompress(blocks[-1]); assert len(vendbytes)==8*n
    vendor=np.frombuffer(vendbytes,dtype='<i8').copy()
    return comps,vendor

def predictors(comp):
    return {name:_bits(arr) for name,arr in fixed_paths(comp).items()}

def _choice_for_model(model,pi,vendor):
    n=len(next(iter(pi.values())));typ=model['type'];names=model.get('paths',[])
    if typ in ['raw','path']:
        return np.zeros(n,dtype=np.uint8)
    if typ=='vendor':
        lookup=model['vendor_to_path'];
        paths=np.asarray([lookup.get(str(int(v)),model['default']) for v in vendor])
        return paths
    if typ in ['pair','five']:
        cand=np.stack([pi[k] for k in names],axis=0)
        return cand
    raise ValueError('Unknown type')

def _pack_codes(code,k):
    return b''.join(np.packbits((code>>j)&1,bitorder='little').tobytes() for j in range(k))

def _unpack_codes(blob,n,k):
    size=(n+7)//8;assert len(blob)==size*k
    out=np.zeros(n,dtype=np.uint8)
    for j in range(k):out|=(np.unpackbits(np.frombuffer(blob[j*size:(j+1)*size],dtype=np.uint8),bitorder='little')[:n]<<j)
    return out

def encode_target(bits,comp,vendor,model):
    target=np.asarray(bits,dtype=np.uint64);n=len(target);pi=predictors(comp)
    typ=model['type'];streams=[]
    if typ=='raw':
        residual=target.copy();selector_bits=0
    elif typ=='path':
        residual=target^pi[model['paths'][0]];selector_bits=0
    elif typ=='vendor':
        pred=np.zeros(n,dtype=np.uint64)
        lookup=model['vendor_to_path']
        for v in np.unique(vendor):
            k=lookup.get(str(int(v)),model['default'])
            ids=vendor==v;pred[ids]=pi[k][ids]
        residual=target^pred;selector_bits=0
    else:
        names=model['paths'];predstack=np.stack([pi[k] for k in names],axis=0)
        errors=np.bitwise_xor(predstack,target[None,:])
        # Pick predictor with most leading bits agreeing; ties resolve to lowest ID.
        bitlen=np.empty_like(errors,dtype=np.int16)
        # NumPy isatleast -- deterministic non-vectorized bit_length avoids dependence on new numpy APIs.
        for j in range(len(names)):
            bitlen[j]=np.fromiter((int(x).bit_length() for x in errors[j]),count=n,dtype=np.int16)
        choice=np.argmin(bitlen,axis=0).astype(np.uint8)
        residual=errors[choice,np.arange(n)]
        selector_bits=1 if len(names)==2 else 3
        streams.append(C.compress(_pack_codes(choice,selector_bits)))
    # For each model choose cheapest valid lossless word layout; 1 byte mode stored in header.
    candidates=[(_encode_words(residual,m),m) for m in ['raw','lanes']]
    payload,mode=min(candidates,key=lambda x:(len(x[0]),x[1]))
    streams.append(payload)
    return pack_bytes({'type':'target','n':n,'model':model,'residual_mode':mode,'selector_bits':selector_bits},streams)

def decode_target(frame,comp,vendor):
    meta,streams=unpack_bytes(frame);assert meta['type']=='target'
    n=meta['n'];model=meta['model'];typ=model['type'];pi=predictors(comp)
    residual=_decode_words(streams[-1],n,meta['residual_mode'])
    if typ=='raw':return residual
    if typ=='path':return residual^pi[model['paths'][0]]
    if typ=='vendor':
        pred=np.zeros(n,dtype=np.uint64)
        for v in np.unique(vendor):
            k=model['vendor_to_path'].get(str(int(v)),model['default']);sel=(vendor==v);pred[sel]=pi[k][sel]
        return residual^pred
    choice=_unpack_codes(D.decompress(streams[0]),n,meta['selector_bits'])
    assert choice.max(initial=0)<len(model['paths'])
    selected=np.zeros(n,dtype=np.uint64)
    for j,k in enumerate(model['paths']):
        sel=choice==j;selected[sel]=pi[k][sel]
    return residual^selected

def fixed_models(comp,target,vendor):
    """Train ONLY on January; frozen models are applied without inspecting Feb labels."""
    y=_bits(target)
    best_single=min(PATH_NAMES,key=lambda k:(len(encode_target(y,comp,vendor,{'type':'path','paths':[k]})),PATH_NAMES.index(k)))
    # Pre-register pair containing left_to_right and another candidate; optimize on Jan only.
    pair0=min([k for k in PATH_NAMES if k!='left_to_right'],key=lambda k:(len(encode_target(y,comp,vendor,{'type':'pair','paths':['left_to_right',k]})),PATH_NAMES.index(k)))
    labels={}
    for v in np.unique(vendor):
        sel=vendor==v;sub={k:x[sel] for k,x in comp.items()}
        labels[str(int(v))]=min(PATH_NAMES,key=lambda k:(len(encode_target(y[sel],sub,vendor[sel],{'type':'path','paths':[k]})),PATH_NAMES.index(k)))
    return [
        ('raw_zstd',{'type':'raw'}),
        ('left_to_right',{'type':'path','paths':['left_to_right']}),
        ('best_single_january',{'type':'path','paths':[best_single]}),
        ('vendor_january',{'type':'vendor','vendor_to_path':labels,'default':best_single}),
        ('two_path_january',{'type':'pair','paths':['left_to_right',pair0]}),
        ('five_path_fixed',{'type':'five','paths':PATH_NAMES}),
    ]

def validate_dataset(comp,target,vendor,models):
    y=_bits(target);rawvendor=np.asarray(vendor,dtype='<i8')
    context_frame=encode_common(comp,rawvendor)
    context_copy,vendor_copy=decode_common(context_frame)
    assert np.array_equal(rawvendor,vendor_copy)
    for k in comp:assert np.array_equal(_bits(comp[k]),_bits(context_copy[k]))
    rows=[]
    for name,model in models:
        target_frame=encode_target(y,comp,rawvendor,model)
        decoded=decode_target(target_frame,context_copy,vendor_copy)
        assert np.array_equal(y,decoded),f'Bit-exact decoder FAILED: {name}'
        rows.append({'method':name,'target_bytes':len(target_frame),'parent_bytes':len(context_frame),'numeric_subset_total_bytes':len(context_frame)+len(target_frame),'bit_exact':True})
    return rows,context_frame


"""
## 5. Decoder-only round trip control (includes signed zero, infinities, NaN payload)
"""

# Offline-style positive, negative and metadata-integrity controls on constructed bit patterns.
# Synthetic inputs are CONTROLS ONLY and never appear in the real-data results.
n=128;rng=np.random.default_rng(119)
com={k:rng.choice(np.array([0.0,0.5,1.25,1.0e16,-1.0e16]),size=n) for k in BASE_FIELDS+OPTIONAL_FIELDS}
vend=rng.choice(np.array([1,2],dtype=np.int64),size=n)
vals=fixed_paths(com)['left_to_right'].copy()
vals[0]=np.float64(-0.0)
vals[1]=np.float64(np.inf);vals[2]=np.float64(-np.inf)
vals.view(np.uint64)[3]=np.uint64(0x7ff800000000002f) # NaN payload
control_models=[('raw',{'type':'raw'}),('p0',{'type':'path','paths':['left_to_right']}),
                ('vendor',{'type':'vendor','vendor_to_path':{'1':'reverse','2':'balanced'},'default':'left_to_right'}),
                ('pair',{'type':'pair','paths':['left_to_right','reverse']}),
                ('five',{'type':'five','paths':PATH_NAMES})]
cr,_=validate_dataset(com,vals,vend,control_models)
assert all(r['bit_exact'] for r in cr)
assert len({r['target_bytes'] for r in cr})>0
print('Special IEEE-754 bit patterns, all codec types, and common-parent frames round-trip: PASS')


"""
## 6. January calibration; frozen February cross-domain benchmark
"""

# January is calibration. February tests never alter selected rules.
# All scenarios use exactly the same ordered 9-column numeric matrix, plus encoded vendor metadata.
OUT.mkdir(parents=True,exist_ok=True)
results=[];errors={};provenance={};jan_models=None;jan_hash=None
for label,url in SOURCES.items():
    print('Source:',label,flush=True)
    try:
        dest=DATA/(label+'.parquet');prov=download_verified(url,dest)
        comps,target,vendor,info=read_prefix(dest)
        if vendor is None:raise RuntimeError('VendorID required for the vendor-based baseline')
        vi=np.asarray(vendor).astype('int64');assert len(np.unique(vi))>=2
        if label=='yellow_2024_01':
            jan_models=fixed_models(comps,target,vi)
            jan_hash=hashlib.sha256(json.dumps(jan_models,sort_keys=True).encode()).hexdigest()
        if jan_models is None:raise RuntimeError('Calibration data unavailable: cannot test February')
        rows,common=validate_dataset(comps,target,vi,jan_models)
        for row in rows:
            row.update({'dataset':label,'rows':len(target),'model_sha256':jan_hash})
            results.append(row)
        prov.update(info);provenance[label]=dict(prov)
        print('Results:',[(r['method'],r['numeric_subset_total_bytes']) for r in rows])
    except Exception as e:
        errors[label]=type(e).__name__+': '+str(e)
        print('Dataset FAILED:',errors[label],flush=True)
        # Do not replace unavailable sources with synthetic data.
        if label=='yellow_2024_01':break

complete=(len(provenance)==len(SOURCES) and not errors)
rmap={(r['dataset'],r['method']):r for r in results}
comparisons={}
for label in SOURCES:
    if (label,'raw_zstd') not in rmap:continue
    rows=[r for r in results if r['dataset']==label]
    best=min(rows,key=lambda r:r['numeric_subset_total_bytes'])
    best_path=min([r for r in rows if r['method'] in ('left_to_right','best_single_january')],key=lambda r:r['target_bytes'])
    comparisons[label]={'best_method':best['method'],'best_total_bytes':best['numeric_subset_total_bytes'],
       'best_path_reference_bytes':best_path['target_bytes'],'best_path_reference_method':best_path['method'],
       'multi_path_beats_best_frozen_single':min(r['target_bytes'] for r in rows if r['method'] in ('two_path_january','five_path_fixed'))<best_path['target_bytes'],
       'vendor_beats_best_frozen_single':rmap[(label,'vendor_january')]['target_bytes']<best_path['target_bytes']}
decision={'notebook':'NB119','status':'COMPLETE' if complete else 'INCOMPLETE',
          'source_complete':complete,'all_bit_exact':all(r['bit_exact'] for r in results) if complete else False,
          'frozen_model_from_january':jan_models,'calibration_sha256':jan_hash,
          'february_multi_path_better_than_frozen_single':{x:comparisons[x]['multi_path_beats_best_frozen_single'] for x in comparisons if '_02' in x},
          'data_matrix_scope':'Eight numerical parent columns + total_amount + VendorID; not the original whole Parquet file',
          'compression_superiority_vs_state_of_art_established':False,'originality_established':False,
          'prior_art_overlap':'FPC uses two predictors plus selector; Corra uses cross-column redundancy; this experiment is an engineering test, not proof of a new mechanism.',
          'errors':errors}
with (OUT/'comparison.csv').open('w',newline='') as fh:
    if results:
        w=csv.DictWriter(fh,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    else:fh.write('dataset,method,error\n')
for name,obj in [('decision.json',decision),('source_provenance.json',provenance),('comparisons.json',comparisons),('errors.json',errors)]:
    (OUT/name).write_text(json.dumps(obj,indent=2,default=str))
(OUT/'prior_art_gate.md').write_text("""# NB119 interpretation and originality gate\n\nThe compression unit is the exact IEEE-754 bit pattern of nine selected numerical columns, not the original Parquet/CSV text. All nine parent/target streams, VendorID and per-method headers are counted. Every result is decoded from the serialized frames and verified against the original source bits.\n\n**Candidate does not imply novelty.** FPC (Burtscher/Ratanaworabhan, IEEE TC 2009, DOI 10.1109/TC.2008.131) already uses paired predictors, selector coding and XOR residuals: https://userweb.cs.txstate.edu/~burtscher/papers/tr08.pdf . Corra (VLDB Workshop 2024) exploits cross-column relationships: https://arxiv.org/abs/2403.17229 .\n\nFor a defensible novelty claim, demonstrate a distinct mathematical coding principle, formal invertibility, compression gains against faithful published baselines, independent domains, and comprehensive size/runtime accounting. Differences in evaluation order alone or 1-bit predictor selection do not satisfy this condition.\n\nJanuary is calibration; the 2024 February yellow and green sources are frozen external tests. First 125,000 records per source are a deterministic prefix, not a random sample. Missing optional fees are imputed zero and logged. The numeric subset is not the original entire source file.\n""")
print('NB119 STATUS:',decision['status'],'BITE EXACT:',decision['all_bit_exact'])
print('SIZE TABLE:',comparisons)


"""
## 7. Automatic results ZIP export
"""

ZIP=Path('/content/SGX_NB119_results.zip')
with zipfile.ZipFile(ZIP,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file():z.write(p,arcname=p.relative_to(OUT))
with zipfile.ZipFile(ZIP) as z:assert z.testzip() is None
print('Auto-generated output:',str(ZIP),'bytes:',ZIP.stat().st_size)
try:
    from google.colab import files
    files.download(str(ZIP))
except ImportError:
    print('Not in Colab; file saved to',ZIP)

