#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB96 — Virtual floating-point mismatch forensics

Runs the **official Virtual implementation** on precisely the NB95G NYC Taxi held-out float64 columns, verifies source fingerprint, computes full 64-bit mismatch diagnostics for every original column, and exports exact evidence as `SGX_NB96_results.zip`. The archive includes the **actual Virtual compressed file**, serialized original/decoded word matrices (for scientific audit only; not counted as a compression artifact), and schema metadata. All mismatches and failures are retained. No novelty claim is made. Run this before NB97. Internet required in Colab.
"""

import subprocess,sys,importlib.util
for module,pkg in [('virtual','virtual-parquet==0.2.2'),('pyarrow','pyarrow==20.0.0'),('pandas','pandas==2.2.2'),('numpy','numpy==2.0.2')]:
    if importlib.util.find_spec(module) is None:
        r=subprocess.run([sys.executable,'-m','pip','-q','install',pkg],capture_output=True,text=True,timeout=300)
        if r.returncode:raise RuntimeError(f'Cannot install {pkg}: {r.stderr[-800:]}')
print('Dependencies ready')


import numpy as np,pandas as pd,pyarrow.parquet as pq,virtual
import tempfile,io,json,hashlib,zipfile,urllib.request,os,time,sys,platform,shutil
from pathlib import Path
ROOT=Path('/content') if Path('/content').exists() else Path('/mnt/data')
OUT=ROOT/'SGX_NB96_outputs';OUT.mkdir(exist_ok=True)
URL='https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet'
COLS=['fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge','congestion_surcharge','airport_fee','total_amount']
EXPECTED='3d0fc4346b1b4d3457aaa1e9941fc2729c67a76ca405bc1fb6829d7e22f8f651'
ERR=[];SUMMARY={}
def hashes(vals):
 h=hashlib.sha256()
 for x in vals:h.update(np.ascontiguousarray(x,dtype='<f8').tobytes())
 return h.hexdigest()

def load():
 path=ROOT/'yellow_tripdata_2024-01.parquet'
 if not path.exists():
  req=urllib.request.Request(URL,headers={'User-Agent':'ShiftGuard-X-NB96'})
  with urllib.request.urlopen(req,timeout=100) as a,open(path,'wb') as b:
   while True:
    s=a.read(1024*1024)
    if not s:break
    b.write(s)
 parquet=pq.ParquetFile(path);names={k.lower():k for k in parquet.schema_arrow.names}
 if any(k not in names for k in COLS):raise ValueError('missing required columns')
 frames=[];count=0
 for batch in parquet.iter_batches(batch_size=25000,columns=[names[k] for k in COLS]):
  if count>=120000:break
  df=batch.to_pandas().iloc[:120000-count];frames.append(df);count+=len(df)
 df=pd.concat(frames,ignore_index=True)
 vals=[pd.to_numeric(df[names[k]],errors='coerce').to_numpy(dtype='<f8') for k in COLS]
 if len(vals[0])!=120000:raise ValueError('Taxi source contains fewer than 120000 rows')
 return [np.ascontiguousarray(v[30000:],dtype='<f8') for v in vals],hashlib.sha256(path.read_bytes()).hexdigest()

def audit():
 vals,source_sha=load();fingerprint=hashes(vals)
 if fingerprint!=EXPECTED:raise ValueError('Held-out data digest differs from NB95G: '+fingerprint)
 df=pd.DataFrame({c:v for c,v in zip(COLS,vals)})
 with tempfile.TemporaryDirectory(prefix='sgx96_') as wd:
  prev=os.getcwd();os.chdir(wd)
  try:
   virtual.to_format(df,'virtual.parquet')
   decoded=virtual.from_format('virtual.parquet')
   aliases=list(decoded.columns)
   if len(aliases)!=len(COLS) or len(set(aliases))!=len(COLS):raise ValueError('Decoded schema cardinality changed')
   # Official Virtual output currently substitutes expression names. Preserve positional map, then validate values.
   mapping=dict(zip(COLS,aliases))
   virtual_bits=[];orig_bits=[];per_column=[]
   for original,alias,v in zip(COLS,aliases,vals):
    got=np.asarray(decoded[alias].to_numpy(dtype='<f8')).view('<u8').copy()
    want=v.view('<u8').copy()
    bad=np.flatnonzero(got!=want)
    virtual_bits.append(got);orig_bits.append(want)
    xor=(got^want)[bad]
    per_column.append({'column':original,'encoded_as':alias,'rows':len(want), 'bitwise_mismatches':int(len(bad)),
         'fraction':float(len(bad)/len(want)),'one_ulp_bit_distance_fraction':float(np.mean([int(x).bit_count()==1 for x in xor])) if len(xor) else None,
         'first_bad_row':int(bad[0]) if len(bad) else None,'first_expected_bits':hex(int(want[bad[0]])) if len(bad) else None,
         'first_got_bits':hex(int(got[bad[0]])) if len(bad) else None})
   np.savez_compressed(OUT/'bitwise_evidence.npz', original=np.stack(orig_bits), decoded=np.stack(virtual_bits))
   # Explicit schema metadata necessary for a standalone recovery codec.
   schema=json.dumps({'original_names':COLS,'virtual_names':aliases,'mapping':mapping},sort_keys=True,separators=(',',':')).encode('utf-8')
   (OUT/'schema.json').write_bytes(schema)
   (OUT/'virtual.parquet').write_bytes(Path('virtual.parquet').read_bytes())
   expected_virtual_file=Path('virtual.parquet').stat().st_size
   other=[str(q) for q in Path('.').rglob('*') if q.is_file() and str(q)!='virtual.parquet']
   if other:raise RuntimeError('Unexpected Virtual side files: '+repr(other))
   s={'status':'COMPLETE','source_sha256':source_sha,'hold_sha256':fingerprint,'rows':len(vals[0]),
    'virtual_parquet_bytes':expected_virtual_file,'schema_json_bytes':len(schema),'version':'virtual-parquet==0.2.2',
    'all_bits_exact':all(x['bitwise_mismatches']==0 for x in per_column),
    'columns':per_column,'na_payload_and_signed_zero_preserved_by_repair':'NOT_YET_TESTED'}
   return s
  finally:os.chdir(prev)
try:
 SUMMARY=audit()
 print('Mismatch counts:',[(x['column'],x['bitwise_mismatches']) for x in SUMMARY['columns']])
except Exception as e:
 ERR.append(repr(e));SUMMARY={'status':'INCOMPLETE','errors':ERR}
(OUT/'audit.json').write_text(json.dumps(SUMMARY,indent=2))
archive=ROOT/'SGX_NB96_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for f in OUT.iterdir():z.write(f,f.name)
print('Output:',archive,'size',archive.stat().st_size,'status',SUMMARY['status'])
try:
 from google.colab import files; files.download(str(archive))
except ImportError:pass

