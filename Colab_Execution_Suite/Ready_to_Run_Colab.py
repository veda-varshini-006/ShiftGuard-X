#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB95G — Official Virtual Bitwise Verification

**Run all cells in a fresh Google Colab CPU runtime.** This notebook fixes the Virtual expression-derived column naming issue. It downloads authentic UCI Bike Sharing and NYC Yellow Taxi data, verifies source and held-out float64 fingerprints, runs the **official** `virtual-parquet` encoder/decoder, compares every 64-bit float representation, records full Virtual bytes plus explicit schema-repair mapping, and compares sizes against earlier *archived* SGX NB95 measurements **only when the held-out hash matches**.

**Important:** This is a prior-art parity notebook, not a new codec. SGX byte counts are recorded NB95 benchmark figures, **not recomputed in this notebook**. It does not count a Virtual result unless it reconstructs the exact original IEEE-754 bit pattern, including zeros and NaNs. Positional mapping is allowed only if *all* bits match. If the official library changes or data differs, the notebook reports `INCOMPLETE` instead of fabricating results.

Outputs: `SGX_NB95G_results.zip` for upload to ChatGPT.
"""

import sys,subprocess,importlib.util
INSTALL=[]
for module,pkg in [('pandas','pandas==2.2.2'),('numpy','numpy==2.0.2'),('pyarrow','pyarrow==20.0.0'),('virtual','virtual-parquet==0.2.2')]:
    if importlib.util.find_spec(module) is None:
        r=subprocess.run([sys.executable,'-m','pip','-q','install',pkg],capture_output=True,text=True,timeout=300)
        INSTALL.append({'package':pkg,'returncode':r.returncode,'stderr':r.stderr[-500:]})
        print(pkg, 'installed' if r.returncode==0 else 'INSTALL FAILED')
    else:INSTALL.append({'package':pkg,'status':'present'})


import os,io,csv,json,time,sys,hashlib,zipfile,urllib.request,tempfile,platform,importlib.metadata,traceback
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
try:import virtual
except Exception as e:virtual=None;VIRTUAL_ERR=repr(e)
ROOT=Path('/content') if Path('/content').exists() else Path('/mnt/data')
OUT=ROOT/'SGX_NB95G_outputs';OUT.mkdir(parents=True,exist_ok=True)
CACHE=ROOT/'SGX_NB95G_source_cache';CACHE.mkdir(parents=True,exist_ok=True)
ERR=[];PROVENANCE=[];ROWS=[]
URL_BIKE='https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
URL_TAXI='https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet'
EXPECTED_HOLDS={'UCI-Bike-275':'710e66352e1d341f53797dc9535f6436589ee99bb7765f979209ee474edf1693',
'NYC-TLC-Yellow-2024-01':'3d0fc4346b1b4d3457aaa1e9941fc2729c67a76ca405bc1fb6829d7e22f8f651'}
# Archived NB95 SGX results, for reference only; not recomputed by NB95G.
ARCHIVED_SGX={'UCI-Bike-275':26053,'NYC-TLC-Yellow-2024-01':279003}

def fetch(url,path):
    if not path.exists() or not path.stat().st_size:
        req=urllib.request.Request(url,headers={'User-Agent':'ShiftGuardX-Research-NB95G'})
        with urllib.request.urlopen(req,timeout=100) as a,open(path,'wb') as b:
            while True:
                chunk=a.read(1024*1024)
                if not chunk:break
                b.write(chunk)
    data=path.read_bytes()
    return hashlib.sha256(data).hexdigest(),len(data)

def bike():
    p=CACHE/'uci_bike_275.zip';digest,size=fetch(URL_BIKE,p)
    with zipfile.ZipFile(p) as z:
        name=next(n for n in z.namelist() if n.lower().endswith('hour.csv'))
        df=pd.read_csv(io.BytesIO(z.read(name)))
    keys=['casual','registered','cnt']
    vals={k:np.ascontiguousarray(df[k].to_numpy(dtype='<f8')) for k in keys}
    return 'UCI-Bike-275',vals,URL_BIKE,digest,size

def taxi():
    p=CACHE/'yellow_tripdata_2024-01.parquet';digest,size=fetch(URL_TAXI,p)
    pf=pq.ParquetFile(str(p));keys=['fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge','congestion_surcharge','airport_fee','total_amount']
    mapping={k.lower():k for k in pf.schema_arrow.names}
    if any(k not in mapping for k in keys):raise ValueError('Missing required taxi columns')
    frames=[];count=0
    for batch in pf.iter_batches(batch_size=25000,columns=[mapping[k] for k in keys]):
        if count>=120000:break
        f=batch.to_pandas().iloc[:120000-count];frames.append(f);count+=len(f)
    df=pd.concat(frames,ignore_index=True)
    vals={k:np.ascontiguousarray(pd.to_numeric(df[mapping[k]],errors='coerce').to_numpy(dtype='<f8')) for k in keys}
    return 'NYC-TLC-Yellow-2024-01',vals,URL_TAXI,digest,size

def fingerprint(vals,names):
    h=hashlib.sha256()
    for name in names:h.update(np.asarray(vals[name],dtype='<f8').copy().tobytes())
    return h.hexdigest()

def check_exact(original,got,names):
    if not isinstance(got,pd.DataFrame):return False,'not a DataFrame',None
    if len(got)!=len(original[names[0]]):return False,'row-count mismatch',None
    aliases=list(got.columns)
    if len(aliases)!=len(names) or len(set(aliases))!=len(names):return False,'wrong number of distinct columns',None
    # Normal name-based mapping first; positional mapping is checked, never assumed.
    pairs=list(zip(names, names)) if set(aliases)==set(names) else list(zip(names,aliases))
    for source,alias in pairs:
        try:
            actual=np.asarray(got[alias].to_numpy(dtype='<f8')).copy().view('<u8')
            expected=np.asarray(original[source],dtype='<f8').copy().view('<u8')
        except Exception as e:return False,'column conversion error: '+repr(e),dict(pairs)
        bad=np.flatnonzero(actual!=expected)
        if len(bad):
            i=int(bad[0]);return False,(f'{source} mapped to {alias}: {len(bad)} bitwise mismatches; '
                f'first row {i}, expected 0x{int(expected[i]):016x}, received 0x{int(actual[i]):016x}'),dict(pairs)
    return True,'BIT_EXACT',dict(pairs)
# Synthetic validator smoke test only, not a compression-performance measurement.
zero=np.array([0x8000000000000000,0x7ff80000000000a3],dtype='<u8').view('<f8')
ref={'a':zero,'b':np.array([1.,2.])}
assert check_exact(ref,pd.DataFrame({'(a)':zero,'(b)':ref['b']}),['a','b'])[0]
assert not check_exact(ref,pd.DataFrame({'(a)':np.array([0.,np.nan]),'(b)':ref['b']}),['a','b'])[0]
print('Schema-mapping bitwise validator PASS')


for loader in (bike,taxi):
    try:
        family,values,url,source_sha,source_bytes=loader()
        names=list(values);cut=max(50,int(len(values[names[0]])*.25))
        hold={k:v[cut:] for k,v in values.items()};hold_sha=fingerprint(hold,names)
        matched=hold_sha==EXPECTED_HOLDS[family]
        PROVENANCE.append({'family':family,'source_url':url,'source_sha256':source_sha,
            'source_bytes':source_bytes,'full_rows':len(values[names[0]]),'heldout_rows':len(hold[names[0]]),
            'heldout_sha256':hold_sha,'matched_NB93_NB95':matched})
        if not matched:
            ERR.append({'family':family,'stage':'input_parity','error':'Held-out float64 SHA-256 differs from NB95; comparison unavailable'})
            print(family,'INPUT MISMATCH; no comparison');continue
        if virtual is None:
            ERR.append({'family':family,'stage':'virtual_package','error':globals().get('VIRTUAL_ERR','virtual unavailable')});continue
        with tempfile.TemporaryDirectory(prefix='sgx95g_') as temp:
            wd=Path(temp);original_cwd=Path.cwd()
            try:
                os.chdir(wd)
                df=pd.DataFrame({k:np.asarray(hold[k],dtype='<f8') for k in names})
                start=time.perf_counter();virtual.to_format(df,'codec.parquet');enc_s=time.perf_counter()-start
                start=time.perf_counter();decoded=virtual.from_format('codec.parquet');dec_s=time.perf_counter()-start
                ok,reason,mapping=check_exact(hold,decoded,names)
                files=[p for p in Path('.').rglob('*') if p.is_file()]
                encoded_bytes=sum(x.stat().st_size for x in files)
                # Explicit recoverable source-schema mapping included in total, with 8-byte framing.
                restore={'original_names':names,'encoded_names':list(decoded.columns) if isinstance(decoded,pd.DataFrame) else [],
                    'mapping':mapping}
                schema_bytes=len(json.dumps(restore,ensure_ascii=False,separators=(',',':')).encode('utf8'))+8
                virtual_total=encoded_bytes+schema_bytes
                record={'family':family,'heldout_rows':len(df),'column_count':len(names),'heldout_sha256':hold_sha,
                    'SGX_archived_NB95_bytes':ARCHIVED_SGX[family], 'SGX_reexecuted_in_NB95G':False,
                    'virtual_codec_file_bytes':encoded_bytes,'virtual_schema_bytes':schema_bytes,'virtual_total_bytes':virtual_total,
                    'virtual_bits_verified':ok,'virtual_verification':reason,'virtual_output_columns':json.dumps(list(decoded.columns)) if isinstance(decoded,pd.DataFrame) else '',
                    'schema_mapping':json.dumps(mapping,default=str),'virtual_encode_s':enc_s,'virtual_decode_s':dec_s,
                    'virtual_output_files':len(files),'virtual_library_version':__import__('importlib').metadata.version('virtual-parquet')}
                if ok:record['SGX_archived_saving_vs_virtual_pct']=100*(virtual_total-ARCHIVED_SGX[family])/virtual_total
                else:ERR.append({'family':family,'stage':'virtual_bitwise','error':reason})
                ROWS.append(record)
                print(family,'virtual',virtual_total,'SGX archived',ARCHIVED_SGX[family],'bitwise',ok,reason)
            finally:os.chdir(original_cwd)
    except Exception as e:
        ERR.append({'family':getattr(loader,'__name__','unknown'),'stage':'experiment','error':repr(e)})
        print(loader.__name__,'ERROR',repr(e))


def write_csv(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r)) if rows else ['status']
    with open(OUT/name,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        if rows:w.writerows(rows)
        else:w.writerow({'status':'NO_ROWS'})
write_csv('matched_virtual_comparison.csv',ROWS)
write_csv('source_provenance.csv',PROVENANCE)
write_csv('run_issues.csv',ERR)
complete=len(ROWS)==2 and not ERR and all(x['virtual_bits_verified'] for x in ROWS)
report={'notebook':'NB95G','status':'COMPLETE' if complete else 'INCOMPLETE',
    'official_Virtual_bitwise_verified_both':bool(complete),'SGX_size_values_archived_not_reexecuted':True,
    'novel_algorithm_established':False,'direct_Corra_C3_comparison':'NOT_PERFORMED',
    'caution':'Only selected parsed float64 columns are validated. These development datasets are not independent blinded novelty evidence.',
    'next':'Review verified Virtual performance and identify a genuinely distinct mechanism' if complete else 'Resolve true Virtual bit mismatch or source mismatch before claiming prior-art superiority'}
(OUT/'decision.json').write_text(json.dumps(report,indent=2))
(OUT/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),
    'numpy':np.__version__,'pandas':pd.__version__,'installation':INSTALL},indent=2,default=str))
result=ROOT/'SGX_NB95G_results.zip'
with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.iterdir()):
        if f.is_file():z.write(f,arcname=f.name)
print(json.dumps(report,indent=2));print('RESULTS:',result)
try:
    from google.colab import files
    files.download(str(result))
except ImportError:pass

