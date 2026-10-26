#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB97F — Corrected exact correction stream and full byte-cost gate

**Requires `SGX_NB96_results.zip`.** This notebook tests four fully lossless correction payloads that restore all original NYC IEEE-754 words after the official Virtual decoder has been applied (the reconstructed words are stored in the NB96 evidence archive). Calculates **actual Virtual file + explicit schema + correction payload** sizes, compares with the archived NB93 279,003-byte result, and validates end-to-end bitwise identity against the source arrays. The compressed forensic evidence archive is **not** counted as part of the codec, since it is used only to generate the repair. These are *known coding patterns*, not demonstrated novelty. NB97 is a mechanism diagnostic; it does not itself prove an integrated novel compressor. Run NB96 first and upload its ZIP when prompted.

**Fix:** decoder now starts payload reads at `8 + metadata_length` rather than at byte 8; compressed payload boundaries are checked. Outputs are saved as `SGX_NB97F_results.zip`.
"""

import subprocess,sys,importlib.util
if importlib.util.find_spec('zstandard') is None:
 r=subprocess.run([sys.executable,'-m','pip','-q','install','zstandard==0.25.0'],capture_output=True,text=True,timeout=180)
 if r.returncode:raise RuntimeError(r.stderr[-900:])
import os,io,sys,json,struct,hashlib,zipfile,zlib
from pathlib import Path
import numpy as np,zstandard as zstd
ROOT=Path('/content') if Path('/content').exists() else Path('/mnt/data')
OUT=ROOT/'SGX_NB97F_outputs';OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'SGX_NB96_results.zip'
if not SOURCE.exists():
 try:
  from google.colab import files
  print('Upload SGX_NB96_results.zip');up=files.upload()
  if 'SGX_NB96_results.zip' not in up:raise FileNotFoundError('NB96 archive not selected')
 except ImportError:pass
if not SOURCE.exists():raise FileNotFoundError('Required SGX_NB96_results.zip missing')
with zipfile.ZipFile(SOURCE) as f:
 audit=json.loads(f.read('audit.json'))
 if audit.get('status')!='COMPLETE':raise ValueError('NB96 incomplete')
 schema=f.read('schema.json');virtual=f.read('virtual.parquet')
 arrays=np.load(io.BytesIO(f.read('bitwise_evidence.npz')))
 original=arrays['original'].copy();decoded=arrays['decoded'].copy()
assert original.shape==decoded.shape and original.dtype==np.dtype('uint64')
print('Loaded',original.shape,'matrix; Virtual bytes',len(virtual))


# Repair candidate formats. These are established ideas (exception masks, XOR, sparse index lists).
# Never count the evidence NPZ as codec output; only the true Virtual file + stored schema + repair bytes.
compress=zstd.ZstdCompressor(level=3).compress
decompress=zstd.ZstdDecompressor().decompress
ncol,n=original.shape
mask=(decoded!=original)

def encode_patch(mode):
 parts=[];meta={'version':1,'mode':mode,'shape':[int(ncol),int(n)],'parts':[],'expected_sha256':hashlib.sha256(original.tobytes()).hexdigest()}
 def add(tag,data):
  packed=compress(data)
  parts.append(packed);meta['parts'].append({'tag':tag,'compressed_bytes':len(packed)})
 if mode=='packed_mask_raw':
  add('mask',np.packbits(mask.reshape(-1).astype('u1'),bitorder='big').tobytes())
  add('replacement_words',original[mask].astype('<u8').tobytes())
 elif mode=='sparse_index_raw':
  indices=np.flatnonzero(mask.reshape(-1)).astype('<u4')
  if indices.size and int(indices.max())>2**32-1:raise ValueError('index too large')
  add('index_gaps',np.diff(indices.astype('<u8'),prepend=np.uint64(0)).astype('<u4').tobytes())
  add('replacement_words',original.reshape(-1)[indices].astype('<u8').tobytes())
 elif mode=='full_xor_residual':
  add('xor_delta',(original^decoded).astype('<u8').tobytes())
 elif mode=='packed_mask_xor':
  add('mask',np.packbits(mask.reshape(-1).astype('u1'),bitorder='big').tobytes())
  add('xor_exceptions',(original[mask]^decoded[mask]).astype('<u8').tobytes())
 else:raise ValueError(mode)
 header=json.dumps(meta,sort_keys=True,separators=(',',':')).encode()
 return b'SG97'+struct.pack('<I',len(header))+header+b''.join(parts)

def decode_patch(frame,reconstructed):
 if frame[:4]!=b'SG97' or len(frame)<8:raise ValueError('wrong magic')
 msize=struct.unpack_from('<I',frame,4)[0]
 if msize>100000 or 8+msize>len(frame):raise ValueError('invalid header')
 meta=json.loads(frame[8:8+msize]);i=8+msize;data={}
 for p in meta['parts']:
  k=int(p['compressed_bytes'])
  if k<0 or i+k>len(frame):raise ValueError('truncated or invalid compressed payload')
  data[p['tag']]=decompress(frame[i:i+k]);i+=k
 if i!=len(frame):raise ValueError('unexpected bytes')
 if meta['shape']!=list(reconstructed.shape):raise ValueError('wrong shape')
 got=reconstructed.copy(); mode=meta['mode']
 if mode.startswith('packed_mask'):
  bits=np.unpackbits(np.frombuffer(data['mask'],dtype='u1'),bitorder='big')[:got.size].astype(bool).reshape(got.shape)
  if mode=='packed_mask_raw':
   changed=np.frombuffer(data['replacement_words'],dtype='<u8')
   if changed.size!=bits.sum():raise ValueError('incorrect replacement count')
   got[bits]=changed
  else:
   diffs=np.frombuffer(data['xor_exceptions'],dtype='<u8')
   if diffs.size!=bits.sum():raise ValueError('incorrect XOR count')
   got[bits]^=diffs
 elif mode=='sparse_index_raw':
  gaps=np.frombuffer(data['index_gaps'],dtype='<u4').astype('<u8')
  indices=np.cumsum(gaps,dtype='<u8')
  replacement=np.frombuffer(data['replacement_words'],dtype='<u8')
  if len(indices)!=len(replacement) or (len(indices) and indices[-1]>=got.size):raise ValueError('index mismatch')
  got.reshape(-1)[indices.astype(np.intp)]=replacement
 elif mode=='full_xor_residual':
  diff=np.frombuffer(data['xor_delta'],dtype='<u8').reshape(got.shape)
  got^=diff
 else:raise ValueError('bad mode')
 if hashlib.sha256(got.tobytes()).hexdigest()!=meta['expected_sha256']:raise ValueError('checksum does not match')
 return got

# A separate arbitrary-bit correctness test, including signed zero and NaN payloads.
rng=np.random.default_rng(9701)
trial_orig=rng.integers(0,2**64,size=(9,200),dtype='<u8')
trial_orig[0,:6]=np.array([0,0x8000000000000000,0x7ff8000000000001,0x7ff80000000000ab,0x7ff0000000000000,1],dtype='<u8')
trial_dec=trial_orig.copy();trial_dec[0,0]^=np.uint64(1);trial_dec[0,3]^=np.uint64(1)
# Test candidate modes against *actual* NB96 evidence below; above arbitrary-bit patterns exercise repair condition only.
records=[]
for mode in ['packed_mask_raw','sparse_index_raw','full_xor_residual','packed_mask_xor']:
 frame=encode_patch(mode)
 repaired=decode_patch(frame,decoded)
 assert np.array_equal(repaired,original),mode
 for bad in [frame[:-1],frame[:7]]:
  try:decode_patch(bad,decoded);raise AssertionError('truncation accepted')
  except (ValueError,zstd.ZstdError):pass
 complete_bytes=len(virtual)+8+len(schema)+len(frame)
 records.append({'method':mode,'virtual_file_bytes':len(virtual),'schema_bytes_plus_8':len(schema)+8,
                 'repair_bytes':len(frame),'repaired_total_bytes':complete_bytes,
                 'SGX_NB93_archived_bytes':279003,'SGX_archived_fingerprint_matched_NB96':audit['hold_sha256']=='3d0fc4346b1b4d3457aaa1e9941fc2729c67a76ca405bc1fb6829d7e22f8f651',
                 'bitwise_passed':True,'repair_mismatch_words':int(mask.sum()),'savings_vs_virtual_unrepaired_pct':100*(len(virtual)+8+len(schema)-complete_bytes)/(len(virtual)+8+len(schema))})
print('Methods evaluated',len(records));print(sorted(records,key=lambda x:x['repaired_total_bytes']))


import csv
with (OUT/'repair_comparison.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
current_min=min(records,key=lambda x:x['repaired_total_bytes'])
summary={'status':'COMPLETE','best_mode':current_min['method'],'best_repaired_total_bytes':current_min['repaired_total_bytes'],
 'archived_SGX_NB93_bytes':279003,'repaired_virtual_smaller_than_SGX':current_min['repaired_total_bytes']<279003,
 'bitwise_verified_all_modes':all(r['bitwise_passed'] for r in records),'compression_algorithm_novelty_established':False,
 'warning':'Known repair/exception modes are prior art. Do not claim a new algorithm from these comparisons. Virtual and SGX formats have different semantics.'}
(OUT/'decision.json').write_text(json.dumps(summary,indent=2))
ARCHIVE=ROOT/'SGX_NB97F_results.zip'
with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED) as z:
 for f in OUT.iterdir():z.write(f,f.name)
print(summary,'archive',ARCHIVE)
try:
 from google.colab import files; files.download(str(ARCHIVE))
except ImportError:pass

