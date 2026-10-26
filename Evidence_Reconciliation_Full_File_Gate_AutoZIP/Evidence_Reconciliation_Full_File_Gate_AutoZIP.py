#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB107 — Reproducibility and full-file gate

Checks the consistency of NB100–NB106 recorded results, source fingerprints, and whether comparisons are repair-stream-only or whole-file. Prevents treating same-domain transfer as independent-domain validation. Emits an auditable ZIP automatically. **Run NB106 first.**

Upload `SGX_NB100_results.zip`, `SGX_NB102_results.zip`, `SGX_NB103_results.zip`, `SGX_NB104_results.zip`, `SGX_NB105_results.zip`, `SGX_NB106_results.zip`. Optional NB93 total is retained only as a historical, different-month comparison and is **not** a February matched baseline.
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
OUT=ROOT/'SGX_NB107_outputs'; OUT.mkdir(exist_ok=True)
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


names={'100':['SGX_NB100_results.zip'],'102':['SGX_NB102_results.zip','SGX_NB102_results (1).zip'],'103':['SGX_NB103_results.zip'],'104':['SGX_NB104_results.zip'],'105':['SGX_NB105_results.zip'],'106':['SGX_NB106_results.zip']}
reports={};zip_hashes={}
for id,candidates in names.items():
 with upload_zip(candidates) as z:
  assert z.testzip() is None
  reports[id]=json.loads(z.read('decision.json'))
  zip_hashes[id]=hashlib.sha256(b''.join(z.read(n) for n in sorted(z.namelist()))).hexdigest()
  assert reports[id]['status']=='COMPLETE',f'NB{id} did not complete'
print('All six archives and decisions verified')


with upload_zip(names['102']) as z:
 with np.load(io.BytesIO(z.read('external_bitwise_evidence.npz'))) as q:
  orig=q['original'].astype('<u8',copy=True);dec=q['decoded'].astype('<u8',copy=True)
assert sha(orig)==reports['102']['float64_source_bits_sha256']
counts=np.count_nonzero(orig!=dec,axis=1).tolist()
assert counts[-1]==reports['102']['mismatches'] and all(x==0 for x in counts[:-1])
unit_sign(orig[-1],dec[-1]);assert len(orig[-1])==120000
assert reports['103']['comparison'][0]['mismatches']==counts[-1]
assert reports['104']['mismatch_count']==counts[-1]
assert reports['100']['source_NB96_sha256']==reports['103']['comparison'][0]['frozen_model_source']
assert reports['105']['all_bit_exact'] and reports['106']['bit_exact_all']
with upload_zip(names['104']) as z: classical=list(csv.DictReader(io.StringIO(z.read('comparison.csv').decode())))
with upload_zip(names['106']) as z: context=list(csv.DictReader(io.StringIO(z.read('comparison.csv').decode())))
for row in classical+context:
 assert row['bit_exact']=='True'
 assert int(row['virtual_plus_repair_bytes'])==int(row['repair_frame_bytes'])+reports['102']['virtual_total_with_schema_bytes']
print('Bitwise source, cross-archive consistency, and whole-file accounting checks passed')


rows=[];base=int(reports['102']['virtual_total_with_schema_bytes'])
for table,name in [(classical,'NB104 matched classical'),(context,'NB106 context-resolution')]:
 for r in table:rows.append({'experiment':name,'method':r['method'],'virtual_base_bytes':base,'repair_bytes':int(r['repair_frame_bytes']),'complete_file_bytes':int(r['virtual_plus_repair_bytes']),'bit_exact':r['bit_exact']})
rows.sort(key=lambda x:x['complete_file_bytes'])
best=rows[0]
decision={'status':'COMPLETE','february_raw_f64_9cols_bytes':int(orig.nbytes),'february_virtual_plus_schema_bytes':base,'best_observed_whole_file_bytes':best['complete_file_bytes'],'best_observed_coder':best['method'],'best_observed_experiment':best['experiment'],'full_file_improvement_over_unrepaired_virtual':False,'repair_is_necessary_for_bit_exact':True,'external_domain_validation':False,'algorithmic_novelty_established':False,'all_recorded_checks_pass':True,'notebook_archives_SHA256':zip_hashes,'caveats':['Virtual baseline without correction does NOT reconstruct bit-exact','NB103 and NB104/106 use different framing, so repair-byte numbers are not directly interchangeable','Whole-file results here apply ONLY to February Virtual data; no matched February SGX NB93 codec test','No published SOTA or arithmetic coding comparison performed']}
with (OUT/'full_file_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
print(json.dumps(decision,indent=2));export_zip('SGX_NB107_results.zip')

