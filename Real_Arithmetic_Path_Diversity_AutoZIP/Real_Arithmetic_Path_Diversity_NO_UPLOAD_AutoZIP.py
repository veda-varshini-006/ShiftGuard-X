#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB118 — Natural IEEE-754 Arithmetic-Path Diversity Gate

**Purpose:** Find out whether an *independently recorded real-world target column* has sufficiently frequent, reproducible bit-level matches to different floating-point evaluation orders. **Not a compression benchmark, not a novelty proof.**

- **Public sources:** [NYC TLC official trip data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), Yellow Taxi January 2024, Yellow Taxi February 2024, Green Taxi February 2024. Published itemized taxi charge columns and a separately recorded `total_amount`.
- **No prior NB96/NB114/NB117 ZIPs and no manually uploaded data required.** Downloads 2024 monthly Parquet files, checks HTTP status and SHA-256, reads the **first 125,000 records** per source (may be fewer), and handles only actual real-data source rows.
- **Locked before inspecting targets:** five arithmetic pathways: left-to-right, reverse, balanced pairwise, tip-last, and taxes/fees-first. All use the **same fixed source columns**; no target-derived expressions, rescaling, path training, or row-specific column selection.
- **Core signal:** target matches an alternative path bit-for-bit while *not* matching the left-to-right path. A mere disagreement between two synthetic reconstructions **does not** establish that the published dataset was calculated via either path.
- **Caveat:** The TLC data dictionary describes itemized charges but does *not* guarantee a particular addition expression or a complete accounting identity for every row. Vendor corrections, missing fees, refunds, and differences in reported totals may prevent either path from matching. This experiment is a **hypothesis screen**.

**Stop/go (pre-registered exploratory cutoff):** Continue to an actual codec only if ≥500 additional exact target matches and ≥0.5% of qualified rows occur in the external February Yellow and February Green sources; otherwise reject *this data/domain hypothesis*. The threshold is a project decision criterion, not a statistical significance or novelty test.
"""

"""
## 1. Dependencies and locked experiment definition
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

OUT=Path('/content/SGX_NB118_outputs');OUT.mkdir(parents=True,exist_ok=True)
DATA=Path('/content/SGX_NB118_public_data');DATA.mkdir(parents=True,exist_ok=True)
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


"""
## 2. Fixed binary64 evaluation paths, independent of the recorded total
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
## 3. Positive and negative controls (not included in real-world evidence)
"""

def control_tests():
    # Positive: different parenthesizations of representable IEEE-754 operands.
    comps={name:np.zeros(4,dtype=np.float64) for name in BASE_FIELDS+OPTIONAL_FIELDS}
    comps['fare_amount'][:]=1e16
    comps['extra'][:]=-1e16
    comps['mta_tax'][:]=1.0
    # Left sum -> 1.0, reverse -> 0.0; true binary64 non-associativity.
    paths=fixed_paths(comps)
    assert ieee_bits(paths['left_to_right'])[0]!=ieee_bits(paths['reverse'])[0]
    pos,ex=analyze_real_rows(comps,paths['reverse'],dataset='ENGINEERED_POSITIVE_CONTROL')
    assert pos['additional_exact_target_by_other_path']>0
    assert pos['any_path_disagrees_p0']>0
    # Negative: identity with all addends exactly zero.
    zz={n:np.zeros(20,dtype=np.float64) for n in BASE_FIELDS+OPTIONAL_FIELDS}
    zz['fare_amount'][:]=np.arange(20,dtype=np.float64)
    neg,_=analyze_real_rows(zz,zz['fare_amount'],dataset='ENGINEERED_NEGATIVE_CONTROL')
    assert neg['any_path_disagrees_p0']==0
    assert neg['additional_exact_target_by_other_path']==0
    print('Internal positive and negative controls: PASS')
    return {'positive':pos,'negative':neg}

CONTROL=control_tests()


"""
## 4. Download official NYC TLC monthly files and read deterministic row prefixes
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
## 5. Run all real source months; record failures instead of fabricating a result
"""

results=[];provenance={};examples={};errors={}
for label,url in SOURCES.items():
    print('\nREAL SOURCE',label,url,flush=True)
    try:
        local=DATA/(label+'.parquet')
        prov=download_verified(url,local)
        comps,target,vendor,info=read_prefix(local)
        outcome,example=analyze_real_rows(comps,target,vendor=vendor,dataset=label)
        assert outcome['qualified_rows']>=1000, 'insufficient qualifying real rows'
        provenance[label]={**prov,**info}
        examples[label]=example
        results.append(outcome)
        print('rows',outcome['qualified_rows'], 'path disagreement',outcome['any_path_disagrees_p0'],
              'additional exact alternative',outcome['additional_exact_target_by_other_path'],
              'rate',f"{100*outcome['additional_exact_fraction']:.5f}%")
    except Exception as exc:
        errors[label]={'error_type':type(exc).__name__,'message':str(exc),
                       'traceback_tail':traceback.format_exc()[-1200:]}
        print('REAL SOURCE UNAVAILABLE:',repr(exc))
# No success or novelty can be declared for any unavailable real source.
assert len(results)+len(errors)==len(SOURCES)


"""
## 6. Pre-registered independent-domain gate, confidence intervals and reproducibility package
"""

rmap={x['dataset']:x for x in results}
def passes(label):
    r=rmap.get(label)
    if not r:return False
    return (r['qualified_rows']>=10000 and
            r['additional_exact_target_by_other_path']>=500 and
            r['additional_exact_fraction']>=0.005)
pass_yellow_feb=passes('yellow_2024_02')
pass_green_feb=passes('green_2024_02')
pass_jan=passes('yellow_2024_01')
all_sources_ok=len(results)==len(SOURCES)
# Discovery support requires January, February and Green external sources to pass locked prevalence gate.
gate=all_sources_ok and pass_jan and pass_yellow_feb and pass_green_feb
verdict={
 'notebook':'NB118',
 'status':'COMPLETE' if all_sources_ok else 'INCOMPLETE_PUBLIC_SOURCE_ACCESS',
 'all_three_real_sources_available':all_sources_ok,
 'january_discovery_gate_pass':pass_jan,
 'february_yellow_external_gate_pass':pass_yellow_feb,
 'february_green_external_gate_pass':pass_green_feb,
 'arithmetic_path_domain_hypothesis_supports_followup':bool(gate),
 'novel_algorithm_established':False,
 'compression_superiority_established':False,
 'interpretation':'Potential natural numerical redundancy only; matching a total does not establish original evaluation order.',
 'no_data_fabrication':True,'error_sources':list(errors),
 'pre_registered_min_additional_exact':500,'pre_registered_min_fraction':0.005,
}
(OUT/'decision.json').write_text(json.dumps(verdict,indent=2))
(OUT/'real_source_results.json').write_text(json.dumps(results,indent=2))
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
(OUT/'examples.json').write_text(json.dumps(examples,indent=2))
(OUT/'errors.json').write_text(json.dumps(errors,indent=2))
(OUT/'synthetic_internal_controls.json').write_text(json.dumps(CONTROL,indent=2))
flat=[]
for r in results:
 flat.append({k:r[k] for k in ['dataset','qualified_rows','p0_exact_target','any_path_exact_target','any_path_disagrees_p0','additional_exact_target_by_other_path','additional_exact_fraction','unexplained_by_any_path']})
with (OUT/'comparison.csv').open('w',newline='') as f:
 columns=['dataset','qualified_rows','p0_exact_target','any_path_exact_target','any_path_disagrees_p0','additional_exact_target_by_other_path','additional_exact_fraction','unexplained_by_any_path']
 w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(flat)
notes=['# NB118 — Natural arithmetic-path discovery','',
       'Real sources: New York City TLC official Parquet, 2024 Yellow Jan/Feb and Green Feb.',
       'https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page','',
       '**Not a novel compressor.** Experimental gate: '+str(verdict['arithmetic_path_domain_hypothesis_supports_followup']),
       'Source completeness: '+verdict['status'],'',
       '## Outcomes']
for r in results:
 notes+=['- '+r['dataset']+': '+str(r['additional_exact_target_by_other_path'])+' independent recorded target matches from alternatives only / '+str(r['qualified_rows'])+' qualified rows ('+f"{100*r['additional_exact_fraction']:.4f}"+'%). 95% Wilson CI: '+str(r['additional_exact_CI95_Wilson'])+'.']
if errors:notes+=['','## Unavailable sources',json.dumps(errors,indent=2)]
notes+=['','## Limitations',
 '- This is a prevalence experiment, not an encoder/decoder or whole-file compression benchmark.',
 '- Addition-order differences are real IEEE-754 effects but do not prove the upstream vendor evaluated a specific expression.',
 '- TLC sources may omit fees, have corrections, include vendor processing, or contain data anomalies; see zero-imputation assumptions.',
 '- First 125k rows per month are deterministic but not random; Wilson intervals assume independent sampled rows and are descriptive only, not population inference. Confirm on independent larger random samples.',
 '- Source definition must not be changed in response to the held-out February outcomes.',
 '- Multiple-predictor compression, dependency coding and exception coding have substantial prior art (ALP, Corra, FPC, Virtual).',
 '- Even a passed prevalence gate would NOT establish originality or compression advantage.',
 '','## Next research action',
 'If any source is unavailable: resolve access and rerun identical locked experiment.',
 'If the gate fails: stop arithmetic-path coding for this data/domain; reassess the problem statement.',
 'If the gate passes: mathematically differentiate from prior-art multi-predictor coding before building a codec.' ]
(OUT/'README.md').write_text('\n'.join(notes)+'\n')
(OUT/'reproduction_environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'pyarrow':getattr(pq,'__version__','unknown'),'path_names':PATH_NAMES,'max_rows':MAX_ROWS,'sources':SOURCES,'utc_generated':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()},indent=2))
print('DECISION:');print(json.dumps(verdict,indent=2))


"""
## 7. Auto-export ZIP — always includes source failures and provenance, never a fake PASS
"""

ZIP=Path('/content/SGX_NB118_results.zip')
with zipfile.ZipFile(ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(OUT))
with zipfile.ZipFile(ZIP) as z:
    assert z.testzip() is None
    assert 'decision.json' in z.namelist()
print('NB118 outputs ZIP:',ZIP,'size:',ZIP.stat().st_size)
try:
    from google.colab import files
    files.download(str(ZIP))
except ImportError:
    print('Not running on Colab: inspect ZIP at',ZIP)

