#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# Research_Evidence_Decision

**Research use only.** Run all cells in Google Colab. The last cell exports and downloads the results ZIP. Unproven novelty is never inferred from a size comparison.
"""

"""
Requires **NB100, NB102, NB104, NB106, NB107, NB108, NB109, NB110**. Consolidates evidence with conservative gates. Cannot establish literature novelty automatically; emits explicit blocked conditions.
"""

# !pip -q install zstandard
import io, os, json, csv, zipfile, hashlib, pathlib, zlib, lzma, bz2, struct
import numpy as np
from google.colab import files
import zstandard as zstd
NB = 111
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


zs=upload_archives(['100','102','104','106','107','108','109','110'])
def jd(tag,name='decision.json'):return read_json(zs[tag],name)
d={k:jd(k) for k in ['100','102','104','106','107','108','109']}
audit=jd('110','audit.json')
flags={
 'bit_exact_repair_reported':bool(d['106'].get('bit_exact_all')),
 'different_month_tested':bool(d['102'].get('independent_from_training_month')),
 'independent_domain_repair_test_passed':bool(d['109'].get('external_domain_repair_transfer_passed')),
 'published_state_of_art_fully_compared':bool(d['104'].get('published_state_of_art_fully_compared')),
 'accounting_all_recorded_rows_consistent':not bool(audit['mismatch_rows']),
 'new_mechanism_prior_art_distinguished':False,
 'complete_file_smaller_than_comparable_bit_exact_baselines':False,
}
requirements={'independent_domain_repair_test_passed','published_state_of_art_fully_compared','accounting_all_recorded_rows_consistent','new_mechanism_prior_art_distinguished','complete_file_smaller_than_comparable_bit_exact_baselines'}
blocked=sorted(k for k in requirements if not flags[k])
recommendation='DO_NOT_CLAIM_NEW_ALGORITHM; prioritize strong published baselines and independent-domain workloads'
result={'status':'COMPLETE','checks':flags,'blocked_requirements':blocked,'manuscript_claim_gate':'NOT_PASSED' if blocked else 'NEEDS_HUMAN_PRIOR_ART_REVIEW','recommendation':recommendation,'historical_repair_best_bytes':14250,'base_february_bytes':d['102']['virtual_total_with_schema_bytes'],'recomputed_total_base_plus_repair':int(d['102']['virtual_total_with_schema_bytes'])+14250,'nb107_discrepancy':audit['nb107_best_delta'],'novelty_established':False}
save_json('research_decision.json',result)
save_csv('gates.csv',[{'gate':k,'passed':v} for k,v in flags.items()]);print(json.dumps(result,indent=2))


archive=pathlib.Path(f'/content/SGX_NB{NB}_results.zip')
result_files=sorted(p for p in OUT.rglob('*') if p.is_file())
if not result_files:raise RuntimeError('No output files to export')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in result_files:z.write(p,p.relative_to(OUT))
with zipfile.ZipFile(archive) as z:
    if z.testzip() is not None:raise RuntimeError('ZIP verification failed')
print('Created:',archive,'Files:',len(result_files),'Bytes:',archive.stat().st_size)
files.download(str(archive))

