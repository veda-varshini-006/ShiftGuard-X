#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB105 — Frozen-model robustness and shift gate (auto-ZIP)
Tests January→February sign-model shift, contiguous segment robustness, and a **fixed chance-control**. Does not assert cross-domain generalization.

Upload `SGX_NB96_results.zip`, `SGX_NB102_results.zip` (or `(1)`), and `SGX_NB100_results.zip`. No training on February. Automatically exports ZIP.
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
OUT=ROOT/'SGX_NB105_outputs'; OUT.mkdir(exist_ok=True)
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


jan,jo,jd=read_npz(['SGX_NB96_results.zip'],'bitwise_evidence.npz',decision_member='audit.json')
feb,fo,fd=read_npz(['SGX_NB102_results.zip','SGX_NB102_results (1).zip'],'external_bitwise_evidence.npz')
assert jo.shape[0]==fo.shape[0]==9
assert sha(fo)==feb['float64_source_bits_sha256']
with upload_zip(['SGX_NB100_results.zip']) as z:
 frozen=json.loads(z.read('frozen_model.json'))
model=np.array(frozen['bits'],dtype='u1')
assert len(model)==64


def evaluate_segment(label,o,d):
 m,s=unit_sign(o,d);f=features(d)[m]
 predictors={
  'frozen64':('frozen64',model[f],np.packbits(model,bitorder='big').tobytes()),
  'always_zero':('zero',np.zeros_like(s),b''),
  'always_one':('one',np.ones_like(s),b''),
  'varint':('varint',np.zeros_like(s),b''),
 }
 results={}
 for key,(mode,pred,mb) in predictors.items():
  blob=pack(mode,o,d,pred,mb)
  assert np.array_equal(unpack(blob,d),o)
  results[key]=len(blob)
 return {'segment':label,'n':len(o),'exceptions':int(m.sum()),'positive_error_fraction':float(s.mean()) if len(s) else None,
         **{k+'_bytes':v for k,v in results.items()},
         'gain_vs_best_simple_pct':100*(min(results[k] for k in results if k!='frozen64')-results['frozen64'])/min(results[k] for k in results if k!='frozen64'),
         'bit_exact':True}
# January evaluation deliberately EXCLUDES the first 27k calibration rows.
rows=[evaluate_segment('January heldout',jo[-1,27000:],jd[-1,27000:])]
for i in range(4):
 a=i*30000;b=(i+1)*30000
 rows.append(evaluate_segment('February quarter '+str(i+1),fo[-1,a:b],fd[-1,a:b]))
print(*rows,sep='\n')


decision={'status':'COMPLETE','frozen_model':'NB100 January-only 64-context predictor',
 'January_rows_used_for_model_fit':27000,'February_model_retraining':False,
 'quarter_count':4,'all_bit_exact':all(r['bit_exact'] for r in rows),
 'february_all_quarters_beat_best_simple':all(r['gain_vs_best_simple_pct']>0 for r in rows[1:]),
 'external_domain_confirmed':False,'novelty_established':False,
 'caveat':'Only within-NYC-Taxi month and quarter distribution shifts. Baselines are simple and not prior-art exhaustive.'}
with (OUT/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
print(json.dumps(decision,indent=2));export_zip('SGX_NB105_results.zip')

