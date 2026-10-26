#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# Three_Byte_Accounting_Audit

**Research use only.** Run all cells in Google Colab. The last cell exports and downloads the results ZIP. Unproven novelty is never inferred from a size comparison.
"""

"""
Requires **NB102, NB104, NB106, NB107**. Preserves recorded values, identifies whether a common three-byte shift was introduced upstream, and outputs a recomputed table. No in-place mutation of the original ZIP reports.
"""

# !pip -q install zstandard
import io, os, json, csv, zipfile, hashlib, pathlib, zlib, lzma, bz2, struct
import numpy as np
from google.colab import files
import zstandard as zstd
NB = 110
OUT = pathlib.Path(f'/content/SGX_NB{NB}_outputs'); OUT.mkdir(parents=True, exist_ok=True)
def upload_archives(required):
    print('Upload required archives:', required)
    incoming=files.upload()
    zs={}
    for name,blob in incoming.items():
        for tag in required:
            if f'NB{tag}_results' in name:
                zs[tag]=zipfile.ZipFile(io.BytesIO(blob))
                if zs[tag].testzip() is not None: raise ValueError('Corrupt ZIP '+name)
    missing=set(required)-set(zs)
    if missing: raise ValueError('Missing result ZIPs: '+str(sorted(missing)))
    return zs
def read_json(z,filename):return json.loads(z.read(filename))
def save_json(filename,obj):(OUT/filename).write_text(json.dumps(obj,indent=2,default=str))
def save_csv(filename,rows):
    if not rows:return
    with (OUT/filename).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load_bits(z,filename):
    a=np.load(io.BytesIO(z.read(filename)))
    return a['original'].astype(np.uint64),a['decoded'].astype(np.uint64)
def signed_ulp_error(original,decoded):
    # Work on IEEE-754 monotonic integer ordering, including negative numbers.
    mask=np.uint64(0x8000000000000000)
    def ordered(a):return np.where((a&mask)!=0,~a,a^mask).astype(np.uint64)
    x,y=ordered(original),ordered(decoded)
    # subtract in int64 only after checking adjacent unsigned values; no wrap dependency.
    bigger=x>y
    mag=np.where(bigger,x-y,y-x)
    if np.any(mag>1):return None
    return np.where(bigger,1,-1).astype(np.int8)


zs=upload_archives(['102','104','106','107'])
base=int(read_json(zs['102'],'decision.json')['virtual_total_with_schema_bytes'])
rows=[]
for tag in ['104','106']:
    for row in csv.DictReader(io.StringIO(zs[tag].read('comparison.csv').decode())):
        payload=int(row['repair_frame_bytes']);recorded=int(row['virtual_plus_repair_bytes']);expected=base+payload
        rows.append(dict(source=f'NB{tag}',method=row['method'],virtual_base_NB102=base,repair_frame=payload,recorded_total=recorded,recomputed_total=expected,difference=recorded-expected,recorded_bit_exact=row['bit_exact']))
reconciled=read_json(zs['107'],'decision.json')
issues=[r for r in rows if r['difference']!=0]
mins=min(r['recomputed_total'] for r in rows)
nb107_best=int(reconciled['best_observed_whole_file_bytes'])
report={'status':'COMPLETE_WITH_DISCREPANCY' if issues else 'COMPLETE','virtual_base_bytes':base,'mismatch_rows':issues,'nb107_best_reported':nb107_best,'best_recomputed_total':mins,'nb107_best_delta':nb107_best-mins,'correction_performed_in_source_archives':False,'novelty_established':False}
save_csv('unchanged_and_recomputed_totals.csv',rows);save_json('audit.json',report)
print(json.dumps(report,indent=2))


archive=pathlib.Path(f'/content/SGX_NB{NB}_results.zip')
result_files=sorted(p for p in OUT.rglob('*') if p.is_file())
if not result_files:raise RuntimeError('No output files to export')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in result_files:z.write(p,p.relative_to(OUT))
with zipfile.ZipFile(archive) as z:
    if z.testzip() is not None:raise RuntimeError('ZIP verification failed')
print('Created:',archive,'Files:',len(result_files),'Bytes:',archive.stat().st_size)
files.download(str(archive))

