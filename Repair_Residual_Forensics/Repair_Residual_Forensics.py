#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB98 — Repair Residual Forensics

Analyze signed modular ULP deltas, XOR entropy, and mismatch-run structure using the exact NB96 evidence. This is *not* an algorithmic novelty claim. Produces `SGX_NB98_results.zip`.
"""

import io,zipfile,json,csv,struct,hashlib,os,sys,math,time,platform
from pathlib import Path
import numpy as np
try:
 import zstandard as zstd
except ImportError:
 import subprocess
 subprocess.check_call([sys.executable,'-m','pip','-q','install','zstandard==0.25.0'])
 import zstandard as zstd
ROOT=Path('/content') if Path('/content').exists() else Path('/mnt/data')
from google.colab import files
p=ROOT/'SGX_NB96_results.zip'
if not p.exists():
 print('Upload SGX_NB96_results.zip:')
 upload=files.upload()
 if 'SGX_NB96_results.zip' not in upload:raise FileNotFoundError('SGX_NB96_results.zip is required')
 p.write_bytes(upload['SGX_NB96_results.zip'])
with zipfile.ZipFile(p) as z:
 audit=json.loads(z.read('audit.json'))
 schema=json.loads(z.read('schema.json'))
 raw=z.read('bitwise_evidence.npz')
 with np.load(io.BytesIO(raw)) as arr:
  original=arr['original'].astype('<u8',copy=True)
  decoded=arr['decoded'].astype('<u8',copy=True)
assert original.shape==decoded.shape==(9,90000)
assert audit['status']=='COMPLETE' and audit['source_sha256']=='c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510'
assert np.count_nonzero(original!=decoded)==21658
N=original.shape[1];colnames=schema['original_names'];assert colnames[-1]=='total_amount'
target_original=original[-1];target_decoded=decoded[-1];mask=target_original!=target_decoded
print('Loaded NB96 authentic evidence',original.shape,'bad words',mask.sum())


OUT=ROOT/'SGX_NB98_outputs';OUT.mkdir(exist_ok=True)
import collections
idx=np.flatnonzero(mask);a=target_original[mask];b=target_decoded[mask]
# Bitword differences are modulo 2^64; only finite-positive 1-ULP distances are labelled small.
delta=(a-b).astype('<u8')
signed=delta.view('<i8')
runlength=[]
if idx.size:
 start=prev=int(idx[0])
 for v in idx[1:]:
  v=int(v)
  if v!=prev+1:
   runlength.append(prev-start+1);start=v
  prev=v
 runlength.append(prev-start+1)
# The exact XOR population size is not the same as signed bitword difference.
xor=a^b
pop=np.unpackbits(xor.view(np.uint8)).reshape(-1,64).sum(axis=1)
rows=[{'metric':'rows','value':N},{'metric':'mismatch_words','value':len(idx)},
 {'metric':'mismatch_fraction','value':float(mask.mean())},
 {'metric':'contiguous_mismatch_runs','value':len(runlength)},
 {'metric':'maximum_run_length','value':max(runlength,default=0)},
 {'metric':'single_bit_xor_count','value':int(np.count_nonzero(pop==1))},
 {'metric':'signed_delta_abs_le_1','value':int(np.count_nonzero((signed>=-1)&(signed<=1)))},
 {'metric':'signed_delta_abs_le_127','value':int(np.count_nonzero((signed>=-127)&(signed<=127)))},
 {'metric':'distinct_xor_words','value':int(len(np.unique(xor)))},
 {'metric':'distinct_signed_delta_words','value':int(len(np.unique(signed)))},
 {'metric':'evidence_npz_sha256','value':hashlib.sha256(raw).hexdigest()}]
with (OUT/'mismatch_diagnostics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['metric','value']);w.writeheader();w.writerows(rows)
counts=[{'type':'xor','value':f'0x{int(v):016x}','count':int(c)} for v,c in zip(*np.unique(xor,return_counts=True))]
counts.sort(key=lambda r:-r['count']);counts=counts[:25]
with (OUT/'top_xor_words.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['type','value','count']);w.writeheader();w.writerows(counts)
np.savez_compressed(OUT/'mismatch_positions.npz',indices=idx.astype('<i4'),xor=xor.astype('<u8'),delta=delta.astype('<u8'))
result={'status':'COMPLETE','dataset':'NYC Yellow Taxi first 120,000 parquet rows / NB96 heldout',
 'input_npz_sha256':hashlib.sha256(raw).hexdigest(),'mismatch_count':int(mask.sum()),
 'other_8_columns_bitexact':bool(np.array_equal(original[:-1],decoded[:-1])),
 'novelty_proven':False,'hypothesis':'Assess residual entropy, clustering and one-ULP patterns, no invention claim.'}
(OUT/'decision.json').write_text(json.dumps(result,indent=2))
archive=ROOT/'SGX_NB98_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(OUT.iterdir()):z.write(f,f.name)
print(json.dumps(result,indent=2));files.download(str(archive))

