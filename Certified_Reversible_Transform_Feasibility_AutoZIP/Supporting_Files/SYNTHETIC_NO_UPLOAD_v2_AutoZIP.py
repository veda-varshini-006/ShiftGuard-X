#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB114 — SYNTHETIC ONLY · ZERO UPLOADS (v2)

**THIS NOTEBOOK NEVER ASKS FOR NB96 OR ANY FILE.** It generates all 90,000 rows inside Colab. Run all cells top-to-bottom.

It validates a reversible-transform prototype and compresses synthetic data; it cannot demonstrate real-world advantage or novelty.
"""

"""
### <span style="color:green">START HERE: NO FILE UPLOAD</span>
Run **Runtime → Run all**. The first code cell will create a dataset automatically. The final code cell downloads **SGX_NB114_SYNTHETIC_v2_results.zip**.
"""

import io,os,json,zipfile,struct,time,hashlib,subprocess,sys,csv
from pathlib import Path
import numpy as np
try:
 import zstandard as zstd
except ImportError:
 subprocess.check_call([sys.executable,'-m','pip','-q','install','zstandard'])
 import zstandard as zstd
from google.colab import files  # used only to DOWNLOAD the final results ZIP
np.seterr(over='ignore')
OUT=Path('/content/SGX_NB114_SYNTHETIC_v2_outputs');OUT.mkdir(exist_ok=True)
# Reproducible, generated test data. NO USER UPLOADS REQUIRED.
# IMPORTANT: Synthetic data only tests mechanism feasibility, not real-world superiority.
rng=np.random.default_rng(1142026)
N=90000
# Typical numerical dependency columns with realistic mixed precision / large magnitudes.
a=rng.integers(0,100000,size=N,dtype=np.int64)
b=rng.integers(0,100000,size=N,dtype=np.int64)
c=rng.integers(1,100,size=N,dtype=np.int64)
values=np.vstack([
 a.astype(np.float64),b.astype(np.float64),
 (a+b).astype(np.float64),(a-b).astype(np.float64),
 (a*2+b).astype(np.float64),
 np.round(a/100.0,2), np.round(b/100.0,2),
 np.round((a+b)/100.0,2),
 np.round((a+b+c)/100.0,2)
]).astype(np.float64)
original=values.view(np.uint64).copy()
# Add bit-pattern outliers to exercise specials in a subset of rows.
special=np.array([0x0000000000000000,0x8000000000000000,0x0000000000000001,0x7ff0000000000000,0x7ff8000000000042],dtype=np.uint64)
for j,pattern in enumerate(special):original[j,1000+j]=pattern
sha=hashlib.sha256(original.tobytes()).hexdigest()
print('✅ SYNTHETIC DATA GENERATED — NO UPLOAD REQUIRED:',original.shape,'SHA256:',sha)


"""
## Fully reversible integer-domain representation

Let each IEEE-754 64-bit word be treated as a ring element in $\mathbb Z/2^{64}\mathbb Z$. A target word $t$ is represented either as $t$ (RAW), $t\oplus s$ (XOR), or $(t-s)\bmod 2^{64}$ (SUB), where $s$ is a previously decoded column in the same row. Their inverses are trivial and preserve signed zeros and all NaN payloads because there is **no floating-point arithmetic**.

Encoding independently decides between modes **based on actual complete compressed bytes** for each target block. The source column must have index less than target column, so decoding is acyclic. Selector metadata and byte lengths are included. This decision itself is standard adaptive selection, not asserted novel.
"""

MAGIC=b'SGX11401';C=zstd.ZstdCompressor(level=6);D=zstd.ZstdDecompressor()
def pack_block(word_matrix):
 ncol,n=word_matrix.shape;out=bytearray();out+=MAGIC+struct.pack('<HI',ncol,n)
 # Source column indices, modes, and compressed stream lengths live in the file.
 for target in range(ncol):
  t=word_matrix[target]
  choices=[]
  for source,mode in [(255,0)]+[(s,m) for s in range(target) for m in (1,2)]:
   if mode==0:r=t
   elif mode==1:r=np.bitwise_xor(t,word_matrix[source])
   else:r=t-word_matrix[source]
   payload=C.compress(r.astype('<u8',copy=False).tobytes())
   choices.append((len(payload),source,mode,payload))
  _,source,mode,payload=min(choices,key=lambda x:(x[0],x[1],x[2]))
  out+=struct.pack('<BBI',source,mode,len(payload))+payload
 return bytes(out)
def unpack_block(blob):
 view=memoryview(blob);assert view[:8].tobytes()==MAGIC
 ncol,n=struct.unpack_from('<HI',view,8);offset=14;cols=[]
 for target in range(ncol):
  source,mode,nb=struct.unpack_from('<BBI',view,offset);offset+=6
  assert nb>=0 and offset+nb<=len(view)
  raw=D.decompress(view[offset:offset+nb]);offset+=nb
  assert len(raw)==n*8
  r=np.frombuffer(raw,dtype='<u8').astype(np.uint64,copy=True)
  if mode==0:assert source==255;t=r
  elif mode==1:assert source<target;t=np.bitwise_xor(r,cols[source])
  elif mode==2:assert source<target;t=r+cols[source]
  else:raise ValueError('Invalid mode')
  cols.append(t)
 assert offset==len(view)
 return np.stack(cols)
def compress_dataset(a,block_rows=1024):
 encoded=[]
 for i in range(0,a.shape[1],block_rows):
  block=pack_block(a[:,i:i+block_rows]);encoded.append(block)
 # length-delimited container with full overhead for stand-alone decoding
 return b'SGX114DS'+struct.pack('<II',a.shape[1],len(encoded))+b''.join(struct.pack('<I',len(x))+x for x in encoded)
def decode_dataset(blob):
 v=memoryview(blob);assert v[:8].tobytes()==b'SGX114DS'
 rows,num=struct.unpack_from('<II',v,8);offset=16;blocks=[]
 for _ in range(num):
  size=struct.unpack_from('<I',v,offset)[0];offset+=4
  assert size and offset+size<=len(v)
  blocks.append(unpack_block(v[offset:offset+size]));offset+=size
 assert offset==len(v)
 arr=np.concatenate(blocks,axis=1);assert arr.shape[1]==rows
 return arr


"""
## Hard cases and decoder-only round trips

The test mutates **raw 64-bit words** to include +0, -0, subnormals, signed infinities, several NaN payloads and arbitrary patterns. Bitwise equality is checked as uint64. The transmitted object must decode independently; decoding does not read original evidence.
"""

rng=np.random.default_rng(114)
test=rng.integers(0,2**64,size=(9,257),dtype=np.uint64)
special=np.array([0x0000000000000000,0x8000000000000000,0x0000000000000001,0x8000000000000001,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0x7ff0000000000001,0xfff800000000002a],dtype=np.uint64)
test[:, :9]=special[None,:]
encoded_test=compress_dataset(test,64)
assert np.array_equal(decode_dataset(encoded_test),test)
print('PASS all-bit-pattern round trip, special IEEE words included, bytes:',len(encoded_test))


"""
## Comparison on internally generated synthetic data

No external ZIP files, uploads, Drive mounts, or earlier notebooks are required.
"""

cal=min(27000,original.shape[1]//3)
if original.shape[1]!=90000:print('Notice: source rows differ from expected January size')
rows=[]
for label,mat in [('calibration',original[:,:cal]),('heldout',original[:,cal:])]:
 tic=time.perf_counter();ours=compress_dataset(mat,1024);et=time.perf_counter()-tic
 recovered=decode_dataset(ours);exact=bool(np.array_equal(recovered,mat));assert exact
 stand_alone_zstd=C.compress(mat.astype('<u8',copy=False).tobytes(order='C'))
 zstd_recovered=np.frombuffer(D.decompress(stand_alone_zstd),dtype='<u8').reshape(mat.shape)
 assert np.array_equal(zstd_recovered,mat)
 # Fairness note: zstd encodes a whole matrix contiguously, while the candidate has per-block headers.
 rows.append({'set':label,'rows':mat.shape[1],'candidate_full_bytes':len(ours),'whole_file_zstd_bytes':len(stand_alone_zstd),'delta_vs_zstd_bytes':len(ours)-len(stand_alone_zstd),'gain_percent':round(100*(1-len(ours)/len(stand_alone_zstd)),4),'exact':exact,'encode_seconds':round(et,3)})
 (OUT/f'{label}.sgx114').write_bytes(ours)
 print(rows[-1])


"""
## Decision and honest novelty gate

A positive Zstandard result *does not prove novelty*: this notebook tests only known reversible operators and a conventional minimum-size selector. NB115 is justified only after a distinct selection algorithm / theorem or genuinely new transform is specified and compared to prior art.
"""

with (OUT/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
prior=[
 {'method':'zfp reversible','overlap':'Integer interpretation, reversible decorrelation, special-value preservation','source':'https://zfp.readthedocs.io/en/latest/algorithm.html'},
 {'method':'Corra','overlap':'Cross-column correlation-aware compression','source':'https://arxiv.org/abs/2403.17229'},
 {'method':'ALP','overlap':'Lossless floating-point encoding and integer-domain mappings','source':'https://github.com/cwida/ALP'},
 {'method':'AFC','overlap':'Adaptive selection between compression strategies','source':'https://doi.org/10.1016/j.ins.2023.119847'}]
(OUT/'prior_art_matrix.json').write_text(json.dumps(prior,indent=2))
decision={'algorithm':'NB114 reversible modular/XOR feasibility probe','dataset_type':'SYNTHETIC_GENERATED_NO_UPLOAD','source_sha256':sha,'all_bitwise_exact':all(r['exact'] for r in rows),'holdout_gain_vs_whole_zstd_percent':rows[-1]['gain_percent'],'beat_whole_file_zstd':rows[-1]['candidate_full_bytes']<rows[-1]['whole_file_zstd_bytes'],'novelty_proven':False,'new_operator_or_selection_theorem':False,'novelty_gate':'FAIL: current operators and min-byte adaptive selection are prior-art mechanisms','full_published_benchmark_gate':'NOT RUN','recommendation':'Do not claim novelty or proceed to a novelty manuscript until a distinct mechanism is formulated and strong published baselines are run.'}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
(OUT/'README.txt').write_text('SGX NB114: stand-alone reversible transform feasibility. Bit-exactness does not prove novelty. See decision.json and prior_art_matrix.json.\n')
archive=Path('/content/SGX_NB114_SYNTHETIC_v2_results.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(OUT.rglob('*')):
  if f.is_file():z.write(f,arcname=f.relative_to(OUT))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print('Output archive:',archive,archive.stat().st_size,'bytes')
files.download(str(archive))

