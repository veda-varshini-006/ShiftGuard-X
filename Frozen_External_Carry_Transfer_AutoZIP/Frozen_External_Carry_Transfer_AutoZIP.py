#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# NB113 — Frozen January cross-column rule on February, no retraining

Research hypothesis test. **A size gain does not establish novelty.**
"""

"""
### Upload February evidence (NB102) and the frozen rules ZIP (NB112).
"""

import os,io,json,zipfile,hashlib,csv,sys,subprocess
import numpy as np
from pathlib import Path
from google.colab import files
try:
 import zstandard as zstd
except ImportError:
 subprocess.check_call([sys.executable,'-m','pip','-q','install','zstandard'])
 import zstandard as zstd
C=zstd.ZstdCompressor(level=9);D=zstd.ZstdDecompressor()
def openzip(label):
 print('Upload '+label)
 u=files.upload();name=next(iter(u))
 return zipfile.ZipFile(io.BytesIO(u[name])),hashlib.sha256(u[name]).hexdigest()
def evidence(z,key):
 a=np.load(io.BytesIO(z.read(key)));o=a['original'];d=a['decoded'];assert o.shape==d.shape and o.shape[0]==9 and o.dtype==np.uint64;return o,d
def signed_direction(o,d):
 original=o[8].view(np.float64);decoded=d[8].view(np.float64)
 mismatch=o[8]!=d[8]
 assert np.all(np.isfinite(original)) and np.all(np.isfinite(decoded))
 above=np.nextafter(decoded,np.float64(np.inf));below=np.nextafter(decoded,np.float64(-np.inf))
 valid= np.where(original>decoded,above,below).view(np.uint64)==o[8]
 assert np.all(valid[mismatch]),'Not all mismatches are one-ULP; this model is inapplicable.'
 return mismatch,(original>decoded).astype(np.uint8)
def feature_set(d):
 # Both predictor families depend ONLY on decoded words; never on original target.
 # Baseline: bits from the reconstructed target word. Candidate: bits from other decoded fields.
 t=d[8];out={}
 for k in range(0,16):out[f'total_bit_{k}']=((t>>np.uint64(k))&1).astype(np.uint8)
 for col in [0,1,3,4,5,6,7]:
  for k in [0,1,2,3,4,5,6,7,8,9,10,11]:
   out[f'cross_col_{col}_bit_{k}']=((d[col]>>np.uint64(k))&1).astype(np.uint8)
 return out
def fit_rule(features,mask,direction,start,end,group):
 names=[k for k in features if k.startswith(group)];selected=np.flatnonzero(mask[start:end])+start
 assert len(selected)>100
 best=None
 for name in names:
  x=features[name][selected];y=direction[selected]
  # empirical majority direction for each binary feature state
  lookup=[];errors=0
  for v in [0,1]:
   yy=y[x==v];one=int(yy.sum());zero=len(yy)-one;pred=int(one>zero);lookup.append(pred);errors+=min(one,zero)
  cand=(errors,name,lookup)
  if best is None or cand[:2]<best[:2]:best=cand
 return {'feature':best[1],'lookup':best[2],'calibration_errors':int(best[0]),'calibration_mismatches':int(len(selected))}
def residual_frame(mask,direction,features,rule):
 ix=np.flatnonzero(mask);pred=np.zeros(len(ix),dtype=np.uint8)
 if rule is not None:
  xx=features[rule['feature']][ix];pred=np.where(xx==0,rule['lookup'][0],rule['lookup'][1]).astype(np.uint8)
 residual=np.bitwise_xor(direction[ix],pred)
 maskpack=np.packbits(mask.astype(np.uint8),bitorder='little').tobytes()
 signpack=np.packbits(residual,bitorder='little').tobytes()
 mbytes=C.compress(maskpack);sbytes=C.compress(signpack)
 metadata=json.dumps({'version':1,'n':len(mask),'errors':len(ix),'rule':rule and {'feature':rule['feature'],'lookup':rule['lookup']},'m':len(mbytes),'s':len(sbytes)},sort_keys=True,separators=(',',':')).encode()
 frame=len(metadata).to_bytes(4,'little')+metadata+mbytes+sbytes
 # Decode *only* from frame and decoder-visible original input d, no encoded target leakage.
 return frame
def reconstruct(frame,d):
 ml=int.from_bytes(frame[:4],'little');meta=json.loads(frame[4:4+ml]);payload=frame[4+ml:]
 assert len(payload)==meta['m']+meta['s']
 mask=np.unpackbits(np.frombuffer(D.decompress(payload[:meta['m']]),dtype=np.uint8),bitorder='little')[:meta['n']].astype(bool)
 assert int(mask.sum())==meta['errors']
 residual=np.unpackbits(np.frombuffer(D.decompress(payload[meta['m']:]),dtype=np.uint8),bitorder='little')[:meta['errors']]
 ix=np.flatnonzero(mask);pred=np.zeros(len(ix),dtype=np.uint8)
 if meta['rule'] is not None:
  feats=feature_set(d);r=meta['rule'];x=feats[r['feature']][ix];pred=np.where(x==0,r['lookup'][0],r['lookup'][1]).astype(np.uint8)
 above=np.bitwise_xor(residual,pred).astype(bool)
 result=d.copy();v=d[8].view(np.float64).copy();v[ix]=np.nextafter(v[ix],np.where(above,np.inf,-np.inf));result[8]=v.view(np.uint64)
 return result
def exact_check(frame,d,original):
 return bool(np.array_equal(reconstruct(frame,d),original))
def export(num,out):
 path=Path('/content')/f'SGX_NB{num}_results.zip'
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(out.rglob('*')):
   if f.is_file():z.write(f,f.relative_to(out))
 with zipfile.ZipFile(path) as z:assert z.testzip() is None
 print('Verified output:',path,'bytes:',path.stat().st_size);files.download(str(path))

z102,sha102=openzip('SGX_NB102_results.zip')
z112,sha112=openzip('SGX_NB112_results.zip')
o,d=evidence(z102,'external_bitwise_evidence.npz');model=json.loads(z112.read('frozen_rules.json'))
mask,direction=signed_direction(o,d);features=feature_set(d)
print('External February rows',len(mask),'one-ULP mismatches',int(mask.sum()))


"""
### Apply frozen models to full February; bit-exact verification includes all nine columns.
"""

rows=[];frames={}
for name,rule in [('raw_sign',None),('decoded_total_bit',model['baseline_rule']),('cross_column_bit',model['cross_rule'])]:
 frame=residual_frame(mask,direction,features,rule);frames[name]=frame
 verified=exact_check(frame,d,o);assert verified,f'Bitwise mismatch in {name}'
 rows.append({'method':name,'repair_bytes':len(frame),'bit_exact':verified,'feb_rows':len(mask),'feb_errors':int(mask.sum())})
base_meta=json.loads(z102.read('decision.json'));base=int(base_meta['virtual_total_with_schema_bytes'])
for r in rows:r['virtual_plus_repair_bytes']=base+r['repair_bytes']
print(*rows,sep='\n');print('NB102 recorded base',base)


"""
### Independent frozen transfer decision, byte accounting, automatically download result ZIP.
"""

OUT=Path('/content/SGX_NB113_outputs');OUT.mkdir(exist_ok=True)
with (OUT/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'decision.json').write_text(json.dumps({'status':'COMPLETE','NB102_zip_sha256':sha102,'NB112_zip_sha256':sha112,'frozen_january_rule':True,'all_nine_columns_bit_exact':all(r['bit_exact'] for r in rows),'cross_beats_frozen_total':rows[2]['repair_bytes']<rows[1]['repair_bytes'],'cross_beats_raw':rows[2]['repair_bytes']<rows[0]['repair_bytes'],'february_virtual_base_reported_bytes':base,'novelty_proven':False,'warning':'No new algorithm claim follows from size gains. Entire Virtual format and comparison semantics require audit.'},indent=2))
for name,frame in frames.items():(OUT/f'{name}.repair.bin').write_bytes(frame)
export(113,OUT)

