#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB106 — Frozen context-resolution benchmark

Compares decoder-visible 4-, 16-, 64- and 256-state sign predictors, **fit only on the January calibration prefix**, on unseen February 2024 data. Every model byte, exception mask, frame header and residual is counted. All frames are actually decoded and verified bit-exact. This is classical context modeling, **not** a novelty demonstration. Results are zipped and downloaded automatically.

Required ZIPs: `SGX_NB96_results.zip`, `SGX_NB102_results.zip` (or `(1)`), `SGX_NB100_results.zip`.
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
OUT=ROOT/'SGX_NB106_outputs'; OUT.mkdir(exist_ok=True)
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
with upload_zip(['SGX_NB100_results.zip']) as z: prior=json.loads(z.read('frozen_model.json'))
assert len(jo[-1])==90000 and len(fo[-1])==120000
assert sha(fo)==feb['float64_source_bits_sha256']
jan_o,jan_d=jo[-1,:27000],jd[-1,:27000]
mask_cal,sign_cal=unit_sign(jan_o,jan_d)
def feat(d,k):
 u=np.asarray(d,dtype='<u8')
 if k==4: return (u & np.uint64(3)).astype('u2')
 if k==16:return (u & np.uint64(15)).astype('u2')
 if k==64:return features(u).astype('u2')
 if k==256:return ((u & np.uint64(15)) | (((u>>np.uint64(52)) & np.uint64(15))<<np.uint64(4))).astype('u2')
 raise ValueError('invalid context count')
def train(k):
 ids=feat(jan_d,k)[mask_cal];tot=np.bincount(ids,minlength=k);pos=np.bincount(ids,weights=sign_cal,minlength=k)
 return (pos*2>tot).astype('u1') # deterministic tie-break to zero
models={k:train(k) for k in (4,16,64,256)}
assert np.array_equal(models[64],np.array(prior['bits'],dtype='u1')), 'NB100 frozen-model definition differs; stop and inspect provenance'
print('January model fingerprint verified; calibration events:',int(mask_cal.sum()))


def build_context_frame(o,d,k,lut):
 m,sign=unit_sign(o,d); pred=lut[feat(d,k)[m]]
 parts=[('mask',C.compress(np.packbits(m,bitorder='big').tobytes())),('model',C.compress(np.packbits(lut,bitorder='big').tobytes())),('residual',C.compress(np.packbits(sign^pred,bitorder='big').tobytes()))]
 header={'n':len(o),'k':k,'original_sha256':sha(o),'parts':[(a,len(b)) for a,b in parts],'format':'frozen_context_v1'}
 h=json.dumps(header,sort_keys=True,separators=(',',':')).encode()
 return b'SG6X'+struct.pack('<I',len(h))+h+b''.join(b for _,b in parts)
def decode_context_frame(frame,d):
 assert frame[:4]==b'SG6X';hlen=struct.unpack_from('<I',frame,4)[0];meta=json.loads(frame[8:8+hlen]);p=8+hlen;parts={}
 for name,size in meta['parts']:
  parts[name]=D.decompress(frame[p:p+size]);p+=size
 assert p==len(frame) and len(d)==meta['n']
 k=meta['k'];m=np.unpackbits(np.frombuffer(parts['mask'],dtype='u1'),bitorder='big')[:len(d)].astype(bool)
 lut=np.unpackbits(np.frombuffer(parts['model'],dtype='u1'),bitorder='big')[:k]
 r=np.unpackbits(np.frombuffer(parts['residual'],dtype='u1'),bitorder='big')[:int(m.sum())]
 signs=lut[feat(d,k)[m]]^r;out=d.copy();ix=np.flatnonzero(m)
 out[ix[signs==1]]+=np.uint64(1);out[ix[signs==0]]-=np.uint64(1)
 assert sha(out)==meta['original_sha256']
 return out
rows=[];o=fo[-1];d=fd[-1]
for k,lut in models.items():
 frame=build_context_frame(o,d,k,lut)
 assert np.array_equal(decode_context_frame(frame,d),o)
 rows.append({'method':f'frozen_context_{k}','repair_frame_bytes':len(frame),'virtual_plus_repair_bytes':int(feb['virtual_total_with_schema_bytes'])+len(frame),'bit_exact':True,'model_training':'January 27k calibration'})
for name,pred in [('zero',0),('one',1)]:
 m,s=unit_sign(o,d);frame=pack(name,o,d,np.full(len(s),pred,dtype='u1'))
 assert np.array_equal(unpack(frame,d),o)
 rows.append({'method':name,'repair_frame_bytes':len(frame),'virtual_plus_repair_bytes':int(feb['virtual_total_with_schema_bytes'])+len(frame),'bit_exact':True,'model_training':'none'})
rows.sort(key=lambda r:r['repair_frame_bytes'])
print(*rows,sep='\n')


ref=next(r for r in rows if r['method']=='frozen_context_64');best=rows[0]
decision={'status':'COMPLETE','source':'NYC Taxi February 2024','training':'January prefix (27000), no February refitting','best_model':best['method'],'best_bytes':best['repair_frame_bytes'],'reference_64_bytes':ref['repair_frame_bytes'],'gain_over_64_pct':100*(ref['repair_frame_bytes']-best['repair_frame_bytes'])/ref['repair_frame_bytes'],'bit_exact_all':True,'includes_model_mask_header':True,'novelty_established':False,'limitations':['Same NYC Taxi domain','Context LUT prediction is established coding practice','Not compared with arithmetic coding or published SOTA']}
with (OUT/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
print(json.dumps(decision,indent=2));export_zip('SGX_NB106_results.zip')

