#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB104 — Matched classical-predictor gate (auto-ZIP)
**Research question:** Is January-trained decoder-visible 64-context prediction better than simple classical sign predictors on **held-out February 2024**? Compare *fully self-describing, bit-exact repair frames*, not just payloads. Measure complete Virtual+repair, and report NO novelty claim.

Upload `SGX_NB102_results.zip` (or `SGX_NB102_results (1).zip`) and `SGX_NB100_results.zip`. This is not an independent-domain test. Standard context modeling is prior art.
"""

import numpy as np, json, csv, zipfile, io, hashlib, struct, sys, subprocess
from pathlib import Path
try:
 import zstandard as zstd
except ImportError:
 subprocess.check_call([sys.executable,'-m','pip','-q','install','zstandard==0.25.0'])
 import zstandard as zstd
try:
 from google.colab import files
except ImportError: files=None
ROOT=Path('/content') if Path('/content').exists() else Path('/mnt/data')
OUT=ROOT/'SGX_NB104_outputs'; OUT.mkdir(exist_ok=True)
C=zstd.ZstdCompressor(level=3);D=zstd.ZstdDecompressor()
def upload_zip(candidates):
 for name in candidates:
  p=ROOT/name
  if p.exists():return zipfile.ZipFile(p)
 if files is None:raise FileNotFoundError('Please place '+str(candidates)+' in '+str(ROOT))
 print('Upload one of:',candidates)
 up=files.upload()
 for name in candidates:
  if name in up:
   (ROOT/name).write_bytes(up[name]);return zipfile.ZipFile(ROOT/name)
 raise FileNotFoundError('Expected '+str(candidates))
def read_npz(candidates,member,decision_member='decision.json'):
 with upload_zip(candidates) as z:
  meta=json.loads(z.read(decision_member))
  with np.load(io.BytesIO(z.read(member))) as a:
   orig=a['original'].astype('<u8',copy=True);dec=a['decoded'].astype('<u8',copy=True)
 return meta,orig,dec
def sha(x):return hashlib.sha256(np.asarray(x,dtype='<u8').tobytes()).hexdigest()
def unit_sign(o,d):
 diff=(o-d).astype('<u8');m=o!=d
 ok=(diff==np.uint64(1)) | (diff==np.uint64(0xffffffffffffffff))
 if np.any(m & ~ok):raise ValueError('Non-unit mismatches; this coder must not silently ignore them')
 return m,(diff[m]==np.uint64(1)).astype(np.uint8)
def features(d):return (((d&np.uint64(15)).astype('u1')) | ((((d>>np.uint64(52))&np.uint64(3)).astype('u1'))<<4)).astype('u1')
def uleb(vals):
 out=bytearray()
 for value in vals:
  x=int(value)
  while x>=128:out.append((x&127)|128);x>>=7
  out.append(x)
 return bytes(out)
def readuleb(b,k):
 arr=[];x=shift=0
 for bb in b:
  x|=(bb&127)<<shift
  if bb&128:shift+=7
  else:arr.append(x);x=shift=0
 if shift or len(arr)!=k:raise ValueError('invalid varints')
 return np.array(arr,dtype='u1')
def pack(mode,o,d,signbits,modelbytes=b''):
 mask,sign=unit_sign(o,d)
 if signbits.shape!=sign.shape:raise ValueError('bad sign shape')
 residual=(sign ^ signbits).astype('u1')
 if mode=='varint':raw=uleb(np.where(sign==1,2,1));key='residual'
 else:raw=np.packbits(residual,bitorder='big').tobytes();key='residual'
 parts={'mask':C.compress(np.packbits(mask,bitorder='big').tobytes()),key:C.compress(raw)}
 if modelbytes:parts['model']=C.compress(modelbytes)
 header={'mode':mode,'n':len(o),'sha256':sha(o),'parts':[(k,len(v)) for k,v in parts.items()]}
 h=json.dumps(header,sort_keys=True,separators=(',',':')).encode()
 return b'SG4X'+struct.pack('<I',len(h))+h+b''.join(parts.values())
def unpack(frame,d):
 assert frame[:4]==b'SG4X'
 nhead=struct.unpack_from('<I',frame,4)[0];meta=json.loads(frame[8:8+nhead]);pos=8+nhead;parts={}
 for k,n in meta['parts']:
  parts[k]=D.decompress(frame[pos:pos+n]);pos+=n
 assert pos==len(frame)
 m=np.unpackbits(np.frombuffer(parts['mask'],dtype='u1'),bitorder='big')[:len(d)].astype(bool)
 k=m.sum();mode=meta['mode']
 if mode=='varint':
  codes=readuleb(parts['residual'],int(k));sign=(codes==2).astype('u1')
 else:
  residual=np.unpackbits(np.frombuffer(parts['residual'],dtype='u1'),bitorder='big')[:k]
  if mode=='zero':pred=np.zeros(k,dtype='u1')
  elif mode=='one':pred=np.ones(k,dtype='u1')
  elif mode=='constant':pred=np.full(k,int(parts['model'][0]),dtype='u1')
  elif mode=='frozen64':
   lut=np.unpackbits(np.frombuffer(parts['model'],dtype='u1'),bitorder='big')[:64]
   pred=lut[features(d)[m]]
  elif mode=='previous':
   first=int(parts['model'][0]);pred=np.empty(k,dtype='u1');prior=first
   for i in range(k):
    pred[i]=prior;prior=int(pred[i]^residual[i])
  else:raise ValueError(mode)
  sign=pred^residual
 out=d.copy();idx=np.flatnonzero(m);out[idx[sign==1]]+=np.uint64(1);out[idx[sign==0]]-=np.uint64(1)
 if sha(out)!=meta['sha256']:raise ValueError('reconstruction checksum failed')
 return out
# Note: decoder uses only decoder-visible values, model bytes, mask and residual. Every frame includes headers.
def export_zip(name):
 dest=ROOT/name; fs=sorted(p for p in OUT.rglob('*') if p.is_file())
 if not fs:raise ValueError('No result files')
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
  for f in fs:z.write(f,arcname=f.relative_to(OUT))
 with zipfile.ZipFile(dest) as z:assert z.testzip() is None
 print('Downloaded',dest, 'files:',[f.name for f in fs]);
 if files:files.download(str(dest))


external,orig,dec=read_npz(['SGX_NB102_results.zip','SGX_NB102_results (1).zip'],'external_bitwise_evidence.npz')
assert orig.shape==dec.shape and orig.shape[0]==9
assert sha(orig)==external['float64_source_bits_sha256'], 'source evidence fingerprint differs'
with upload_zip(['SGX_NB100_results.zip']) as z:
 frozen=json.loads(z.read('frozen_model.json')); jan_meta=json.loads(z.read('decision.json'))
model=np.array(frozen['bits'],dtype='u1');assert model.shape==(64,)
o=orig[-1];d=dec[-1];mask,sign=unit_sign(o,d)
print('February evidence:',len(o),'words, exceptions:',mask.sum())


def prev_predictions(sign):
 pred=np.empty_like(sign)
 if len(sign):pred[0]=0;pred[1:]=sign[:-1]
 return pred
methods={
 'varint':('varint',np.zeros_like(sign),b''),
 'zero':('zero',np.zeros_like(sign),b''),
 'one':('one',np.ones_like(sign),b''),
 'constant':('constant',np.full(len(sign),1,dtype='u1'),b'\x01'),
 'previous':('previous',prev_predictions(sign),b'\x00'),
 'frozen64':('frozen64',model[features(d)[mask]],np.packbits(model,bitorder='big').tobytes()),
}
rows=[]
for label,(mode,pred,mbytes) in methods.items():
 frame=pack(mode,o,d,pred,mbytes)
 assert np.array_equal(unpack(frame,d),o),label
 total=int(external['virtual_total_with_schema_bytes'])+len(frame)
 rows.append({'method':label,'repair_frame_bytes':len(frame),'virtual_plus_repair_bytes':total,'bit_exact':True,
              'full_original_float64_bytes_9_columns':int(orig.nbytes)})
 rows.sort(key=lambda r:r['repair_frame_bytes'])
print(*rows,sep='\n')


best=rows[0];frozen_row=next(r for r in rows if r['method']=='frozen64')
best_classic=min((r for r in rows if r['method']!='frozen64'),key=lambda r:r['repair_frame_bytes'])
decision={'status':'COMPLETE','dataset':'NYC TLC February 2024','model_fitted_on':'January 2024 only',
 'mismatch_count':int(mask.sum()),'nonunit_error_count':0,'same_domain_only':True,
 'virtual_with_schema_bytes':int(external['virtual_total_with_schema_bytes']),
 'frozen64_vs_best_classical_pct':100*(best_classic['repair_frame_bytes']-frozen_row['repair_frame_bytes'])/best_classic['repair_frame_bytes'],
 'best_matched_method':best['method'],'classical_baselines':['varint','zero','one','constant','previous'],
 'external_entropy_coder_benchmarked':False,'published_state_of_art_fully_compared':False,'novelty_established':False,
 'interpretation':'Classic predictors are not exhaustive prior-art baselines; do not claim novelty based on this table.'}
with (OUT/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
print(json.dumps(decision,indent=2));export_zip('SGX_NB104_results.zip')

