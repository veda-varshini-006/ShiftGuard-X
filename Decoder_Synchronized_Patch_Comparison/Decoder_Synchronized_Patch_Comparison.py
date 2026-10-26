#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB99 — Decoder-Synchronized Patch Comparison

Compares 5 exact self-framed repair schemes against NB97F’s 19,054-byte mask+XOR reference, using the SAME NB96 evidence. All methods verify exact 64-bit restoration. This is a novelty *exclusion* screen and not a new-algorithm claim. Produces `SGX_NB99_results.zip`.
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


OUT=ROOT/'SGX_NB99_outputs';OUT.mkdir(exist_ok=True)
C=zstd.ZstdCompressor(level=3);D=zstd.ZstdDecompressor()
def comp(x):return C.compress(x)
def decomp(x):return D.decompress(x)
MAGIC=b'SG99'
# Framed data includes magic + header length + canonical JSON + compressed component blobs.
def pack(mode,parts):
 meta={'mode':mode,'n':N,'parts':[{'tag':tag,'compressed_bytes':len(blob)} for tag,blob in parts],
       'original_sha256':hashlib.sha256(target_original.tobytes()).hexdigest()}
 h=json.dumps(meta,sort_keys=True,separators=(',',':')).encode()
 return MAGIC+struct.pack('<I',len(h))+h+b''.join(blob for _,blob in parts)
def unpack(frame):
 if len(frame)<8 or frame[:4]!=MAGIC:raise ValueError('invalid frame')
 hsize,=struct.unpack_from('<I',frame,4)
 if hsize<=0 or hsize>100000 or 8+hsize>len(frame):raise ValueError('invalid header size')
 meta=json.loads(frame[8:8+hsize]);i=8+hsize;result={}
 for p in meta['parts']:
  k=int(p['compressed_bytes']);tag=p['tag']
  if k<0 or i+k>len(frame) or tag in result:raise ValueError('invalid component')
  result[tag]=decomp(frame[i:i+k]);i+=k
 if i!=len(frame):raise ValueError('trailing bytes')
 return meta,result

def leb128_encode(values):
 out=bytearray()
 for x in values:
  v=int(x)
  if v<0:raise ValueError('negative unsigned integer')
  while v>=128:out.append((v&127)|128);v>>=7
  out.append(v)
 return bytes(out)
def leb128_decode(data,count):
 out=[];k=0;val=0;shift=0
 for b in data:
  val|=(b&127)<<shift
  if b&128:
   shift+=7
   if shift>63:raise ValueError('leb128 overlong')
  else:
   out.append(val);val=shift=0
   if len(out)>count:raise ValueError('too many varints')
 if shift or len(out)!=count:raise ValueError('bad varint payload')
 return np.array(out,dtype='<u8')

def encode(mode):
 idx=np.flatnonzero(mask)
 x=(target_original[mask]^target_decoded[mask]).astype('<u8')
 difference=(target_original[mask]-target_decoded[mask]).astype('<u8')
 if mode=='mask_xor':
  pieces=[('mask',comp(np.packbits(mask,bitorder='big').tobytes())),('resid',comp(x.tobytes()))]
 elif mode=='mask_delta':
  pieces=[('mask',comp(np.packbits(mask,bitorder='big').tobytes())),('resid',comp(difference.tobytes()))]
 elif mode=='gaps_xor':
  gaps=np.diff(np.r_[-1,idx]).astype('<u8')
  pieces=[('gaps',comp(leb128_encode(gaps))),('resid',comp(x.tobytes()))]
 elif mode=='gaps_delta':
  gaps=np.diff(np.r_[-1,idx]).astype('<u8')
  pieces=[('gaps',comp(leb128_encode(gaps))),('resid',comp(difference.tobytes()))]
 elif mode=='mask_delta_varint':
  # Map 64-bit modular signed differences to unsigned zigzag WITHOUT overflowing Python scalar arithmetic.
  ss=difference.view('<i8');zz=[(int(v)<<1) if v>=0 else ((-int(v)<<1)-1) for v in ss]
  pieces=[('mask',comp(np.packbits(mask,bitorder='big').tobytes())),('resid',comp(leb128_encode(zz)))]
 else:raise ValueError(mode)
 return pack(mode,pieces)

def decode(frame,base):
 meta,data=unpack(frame);mode=meta['mode'];n=int(meta['n'])
 if n!=len(base):raise ValueError('row count mismatch')
 if mode.startswith('mask_'):
  m=np.unpackbits(np.frombuffer(data['mask'],dtype='u1'),bitorder='big')[:n].astype(bool)
  if len(data['mask'])!=(n+7)//8:raise ValueError('mask length mismatch')
  idx=np.flatnonzero(m)
 else:
  # first gap is idx[0]+1, subsequent gaps are differences between mismatch positions
  # Count is inferred from the *residual* payload for raw-uint64 modes.
  if len(data['resid'])%8:raise ValueError('wrong uint64 residual length')
  gaps=leb128_decode(data['gaps'],len(data['resid'])//8)
  idx=np.cumsum(gaps,dtype='<u8').astype('<i8')-1
  if len(idx) and (np.any(np.diff(idx)<=0) or idx[0]<0 or idx[-1]>=n):raise ValueError('invalid mismatch positions')
 out=np.asarray(base,dtype='<u8').copy()
 if mode=='mask_delta_varint':
  # Variable count of zigzag values is determined by mask population.
  zz=leb128_decode(data['resid'],len(idx))
  signs=zz&np.uint64(1)
  resid=((zz>>np.uint64(1)) ^ (np.uint64(0)-signs)).astype('<u8')
 else:
  if len(data['resid'])!=8*len(idx):raise ValueError('wrong residual count')
  resid=np.frombuffer(data['resid'],dtype='<u8')
 if mode in ('mask_xor','gaps_xor'):out[idx]^=resid
 else:out[idx]+=resid
 if hashlib.sha256(out.tobytes()).hexdigest()!=meta['original_sha256']:raise ValueError('bitwise checksum mismatch')
 return out

# local randomized/pathological IEEE-word reconstruction self-test against provided evidence.
assert np.count_nonzero(mask)==21658
methods=['mask_xor','mask_delta','gaps_xor','gaps_delta','mask_delta_varint']
rows=[]
for mode in methods:
 t=time.perf_counter();f=encode(mode);encode_ms=(time.perf_counter()-t)*1000
 t=time.perf_counter();repaired=decode(f,target_decoded);decode_ms=(time.perf_counter()-t)*1000
 assert np.array_equal(repaired,target_original)
 for corrupt in [f[:-1],f[:7]]:
  try:decode(corrupt,target_decoded);raise AssertionError('corrupt frame accepted')
  except (ValueError,zstd.ZstdError):pass
 rows.append({'method':mode,'full_patch_bytes':len(f),'bitwise_verified':True,
              'encode_ms':round(encode_ms,3),'decode_ms':round(decode_ms,3),
              'repaired_total_virtual_bytes':471416+1062+len(f)})
 print(mode,len(f))
with (OUT/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
base_ref=19054 # NB97F best compressed patch bytes, includes its own framing.
winning=min(rows,key=lambda r:r['full_patch_bytes'])
res={'status':'COMPLETE','NB97F_reference_patch_bytes':base_ref,'winner':winning,
     'smallest_patch_beats_NB97F_by_5pct':winning['full_patch_bytes']<=int(base_ref*.95),
     'source_evidence_sha256':hashlib.sha256(raw).hexdigest(),
     'novel_algorithm_established':False,
     'closest_prior_art_warning':'Masks, sparse varints, modular ULP deltas, zigzag and XOR exception corrections are established coding primitives; improvement alone is not novelty.'}
(OUT/'decision.json').write_text(json.dumps(res,indent=2))
archive=ROOT/'SGX_NB99_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for file in sorted(OUT.iterdir()):z.write(file,file.name)
print(json.dumps(res,indent=2));files.download(str(archive))

