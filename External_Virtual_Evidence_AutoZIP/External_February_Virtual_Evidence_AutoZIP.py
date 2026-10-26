#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB102 — External Month Virtual Bitwise Reproduction

Tests official Virtual on **February 2024 NYC taxi** rather than the January training month. Records the original and decoded 64-bit floating patterns and saves genuine evidence for NB103. This is a new time period of the **same taxi domain**, not an independent domain-family replication. A failed download or encoder run is `INCOMPLETE`; no synthetic substitute. The full frame and schema metadata are included in the Virtual size. Results automatically export as `SGX_NB102_results.zip`.
"""

import os,io,json,csv,zipfile,hashlib,sys,subprocess,urllib.request,tempfile,platform,time
from pathlib import Path
for pkg,mod in [('zstandard==0.25.0','zstandard'),('virtual-parquet==0.2.2','virtual'),('pyarrow==20.0.0','pyarrow')]:
 try:__import__(mod)
 except ImportError:subprocess.check_call([sys.executable,'-m','pip','-q','install',pkg])
import numpy as np,pandas as pd,pyarrow.parquet as pq,zstandard as zstd
import virtual
ROOT=Path('/content') if Path('/content').exists() else Path('/mnt/data')
OUT=ROOT/'SGX_NB102_outputs';OUT.mkdir(exist_ok=True)
ERRORS=[]

def export_zip():
 p=ROOT/'SGX_NB102_results.zip';items=sorted(x for x in OUT.rglob('*') if x.is_file())
 if not items:raise RuntimeError('No result files to export')
 with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
  for f in items:z.write(f,f.relative_to(OUT))
 with zipfile.ZipFile(p) as z:assert z.testzip() is None
 print('Results ZIP:',p)
 try:
  from google.colab import files;files.download(str(p))
 except ImportError:pass

def write_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=str))


# External month, not used for predictor training. Preserve decoded numeric bit patterns.
URL='https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet'
KEYS=['fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge','congestion_surcharge','airport_fee','total_amount']
cache=ROOT/'SGX_NB102_yellow_tripdata_2024-02.parquet'
original=None;decoded=None;schema=None;source_sha=None;hold_sha=None;virtual_bytes=None
try:
 if not cache.exists():
  req=urllib.request.Request(URL,headers={'User-Agent':'ShiftGuard-X-NB102'})
  with urllib.request.urlopen(req,timeout=150) as stream,open(cache,'wb') as dest:
   while True:
    chunk=stream.read(1<<20)
    if not chunk:break
    dest.write(chunk)
 source_sha=hashlib.sha256(cache.read_bytes()).hexdigest()
 pf=pq.ParquetFile(cache);lookup={x.lower():x for x in pf.schema_arrow.names}
 if any(k not in lookup for k in KEYS):raise ValueError('Missing NYC Feb schema columns')
 frames=[];rows=0
 for batch in pf.iter_batches(columns=[lookup[k] for k in KEYS],batch_size=24000):
  if rows>=120000:break
  frame=batch.to_pandas().iloc[:120000-rows];frames.append(frame);rows+=len(frame)
 dataframe=pd.concat(frames,ignore_index=True)
 if rows!=120000:raise ValueError('Fewer than 120000 rows')
 original=np.stack([pd.to_numeric(dataframe[lookup[k]],errors='coerce').to_numpy(dtype='<f8').view('<u8') for k in KEYS],axis=0).copy()
 hold_sha=hashlib.sha256(original.tobytes()).hexdigest()
 with tempfile.TemporaryDirectory(prefix='SGX102_') as tmp:
  cwd=os.getcwd()
  try:
   os.chdir(tmp)
   df=pd.DataFrame({k:original[j].view('<f8').copy() for j,k in enumerate(KEYS)})
   start=time.perf_counter();virtual.to_format(df,'external.parquet');encode_s=time.perf_counter()-start
   start=time.perf_counter();restored=virtual.from_format('external.parquet');decode_s=time.perf_counter()-start
   labels=list(restored.columns)
   if len(labels)!=len(KEYS) or len(set(labels))!=len(KEYS):raise ValueError('Virtual column count/name ambiguity')
   # Positional mapping is explicitly audited: Virtual may expose formula expressions as headings.
   # If original headings are all retained, use name mapping; otherwise use output position as in NB95G.
   matched_labels=KEYS if set(labels)==set(KEYS) else labels
   if len(restored)!=120000:raise ValueError('Virtual output row count changed')
   decoded=np.stack([np.asarray(restored[k],dtype='<f8').copy().view('<u8') for k in matched_labels],axis=0)
   schema={'original_names':KEYS,'decoded_names':labels,'mapping':dict(zip(KEYS,matched_labels))}
   virtual_bytes=sum(p.stat().st_size for p in Path('.').rglob('*') if p.is_file())
   virtual_bytes+=len(json.dumps(schema,separators=(',',':')).encode())+8
  finally:os.chdir(cwd)
except Exception as e:ERRORS.append({'stage':'external_data_or_virtual','error':repr(e)})


state='COMPLETE' if original is not None and decoded is not None and not ERRORS else 'INCOMPLETE'
report={'status':state,'family':'NYC-TLC-2024-02','independent_from_training_month':True,
 'source_url':URL,'source_sha256':source_sha,'float64_source_bits_sha256':hold_sha,
 'virtual_total_with_schema_bytes':virtual_bytes,'mismatches':int(np.count_nonzero(original!=decoded)) if state=='COMPLETE' else None,
 'per_column_mismatches':{k:int(np.count_nonzero(original[j]!=decoded[j])) for j,k in enumerate(KEYS)} if state=='COMPLETE' else None,
 'errors':ERRORS,'novelty_established':False}
write_json('decision.json',report)
write_json('environment.json',{'python':sys.version,'numpy':np.__version__,'virtual_package':'official virtual-parquet','platform':platform.platform()})
if state=='COMPLETE':
 write_json('schema.json',schema)
 bio=io.BytesIO();np.savez_compressed(bio,original=original,decoded=decoded)
 (OUT/'external_bitwise_evidence.npz').write_bytes(bio.getvalue())
write_json('errors.json',ERRORS)
print(json.dumps(report,indent=2));export_zip()

