#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB101 — Frozen Decoder Model Stability and Transfer Gate

Stress-test the same model with strict contiguous folds: freeze on the earliest 30% and evaluate the next 35% and final 35% independently. Separately test whether tuning to earlier observations transfers to later ones. Reference frame costs are recalculated on each identical segment using the same NB99-compatible implementation. Failing gates reject the hypothesis; passing does not establish novelty. Output `SGX_NB101_results.zip`.

This is **algorithmic hypothesis screening**, not a novelty claim. A compact conditional-bit predictor resembles established context modeling; benefits must be compared with prior art.
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
try:
 from google.colab import files
except ImportError:
 files=None
p=ROOT/'SGX_NB96_results.zip'
if not p.exists():
 print('Upload SGX_NB96_results.zip:')
 upload=files.upload() if files else {}
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


import json,csv,io,zipfile,struct,hashlib,time
from pathlib import Path
import numpy as np
import zstandard as zstd
C=zstd.ZstdCompressor(level=3); D=zstd.ZstdDecompressor()
def framepack(method, n, parts, expected):
 meta={'mode':method,'n':n,'sha256':hashlib.sha256(expected.tobytes()).hexdigest(),
       'parts':[{'name':tag,'length':len(b)} for tag,b in parts]}
 h=json.dumps(meta,sort_keys=True,separators=(',',':')).encode()
 return b'SG1X'+struct.pack('<I',len(h))+h+b''.join(b for _,b in parts)
def frameunpack(blob):
 if len(blob)<8 or blob[:4]!=b'SG1X':raise ValueError('bad magic')
 k=struct.unpack_from('<I',blob,4)[0]
 if k>100000 or k<1 or 8+k>len(blob):raise ValueError('bad header')
 meta=json.loads(blob[8:8+k]);pos=8+k;parts={}
 for entry in meta['parts']:
  l=int(entry['length']); key=entry['name']
  if l<0 or pos+l>len(blob) or key in parts:raise ValueError('invalid part')
  parts[key]=D.decompress(blob[pos:pos+l]);pos+=l
 if pos!=len(blob):raise ValueError('trailing bytes')
 return meta,parts
def uleb(vals):
 out=bytearray()
 for x in vals:
  a=int(x)
  if a<0:raise ValueError('negative')
  while a>=128:out.append(128|(a&127));a>>=7
  out.append(a)
 return bytes(out)
def readuleb(b,num):
 out=[];x=shift=0
 for octet in b:
  x|=(octet&127)<<shift
  if octet&128:
   shift+=7
   if shift>63:raise ValueError('too long')
  else:
   out.append(x);x=shift=0
   if len(out)>num:raise ValueError('too many')
 if shift or len(out)!=num:raise ValueError('invalid varints')
 return np.asarray(out,dtype='<u8')
def signs(orig,pred):
 a=(np.asarray(orig,dtype='<u8')-np.asarray(pred,dtype='<u8')).astype('<u8')
 m=(np.asarray(orig,dtype='<u8')!=np.asarray(pred,dtype='<u8'))
 # Real NB96 observations all errors +/- 1; reject data otherwise.
 if np.any(m & (a!=np.uint64(1)) & (a!=np.uint64(0xffffffffffffffff))):
  raise ValueError('not one-ULP differences')
 return m,(a[m]==np.uint64(1)).astype(np.uint8)
def basepatch(orig,pred):
 mask,bit=signs(orig,pred);idx=np.flatnonzero(mask)
 zig=np.where(bit==1,2,1).astype('u1')
 return framepack('NB99_mask_delta_varint',len(orig),[
 ('mask',C.compress(np.packbits(mask,bitorder='big').tobytes())),('resid',C.compress(uleb(zig)))],np.asarray(orig,dtype='<u8'))
def conditional_features(values):
 v=np.asarray(values,dtype='<u8')
 return (((v&np.uint64(15)).astype('u1')) | (((v>>np.uint64(52))&np.uint64(3)).astype('u1')<<4)).astype('u1')
def fit(train_orig,train_dec):
 m,s=signs(train_orig,train_dec)
 feat=conditional_features(train_dec)[m];plus=np.zeros(64,dtype=np.int64);minus=np.zeros(64,dtype=np.int64)
 np.add.at(plus,feat[s==1],1);np.add.at(minus,feat[s==0],1)
 # 64 decoder-visible groups; model is 8 bytes after bit packing.
 return (plus>minus).astype('u1')
def predictor_patch(orig,dec,model):
 mask,s=signs(orig,dec)
 feat=conditional_features(dec)[mask];predicted=model[feat]
 wrong=(predicted!=s).astype('u1')
 return framepack('frozen_sign_predictor',len(orig),[
 ('model',C.compress(np.packbits(model,bitorder='big').tobytes())),
 ('mask',C.compress(np.packbits(mask,bitorder='big').tobytes())),
 ('wrong_signs',C.compress(np.packbits(wrong,bitorder='big').tobytes()))],np.asarray(orig,dtype='<u8'))
def unpatch(frame,dec):
 meta,p=frameunpack(frame);v=np.asarray(dec,dtype='<u8').copy();n=len(v)
 if meta['n']!=n:raise ValueError('length mismatch')
 mb=p['mask'];expected=(n+7)//8
 if len(mb)!=expected:raise ValueError('wrong mask length')
 mask=np.unpackbits(np.frombuffer(mb,dtype='u1'),bitorder='big')[:n].astype(bool)
 k=int(mask.sum())
 if meta['mode']=='NB99_mask_delta_varint':
  zz=readuleb(p['resid'],k)
  if np.any((zz!=1)&(zz!=2)):raise ValueError('unexpected nonunit residual')
  plus=(zz==2)
 elif meta['mode']=='frozen_sign_predictor':
  modelraw=p['model'];wrongraw=p['wrong_signs']
  if len(modelraw)!=8 or len(wrongraw)!=(k+7)//8:raise ValueError('invalid predictor payload')
  model=np.unpackbits(np.frombuffer(modelraw,dtype='u1'),bitorder='big')[:64]
  wrong=np.unpackbits(np.frombuffer(wrongraw,dtype='u1'),bitorder='big')[:k]
  plus=(model[conditional_features(v)[mask]] ^ wrong).astype(bool)
 else:raise ValueError('unknown coding mode')
 index=np.flatnonzero(mask);v[index[plus]]+=np.uint64(1);v[index[~plus]]-=np.uint64(1)
 if hashlib.sha256(v.tobytes()).hexdigest()!=meta['sha256']:raise ValueError('bitwise checksum mismatch')
 return v


OUT=ROOT/'SGX_NB101_outputs';OUT.mkdir(exist_ok=True)
x=target_original.copy();y=target_decoded.copy();n=len(x);cut=int(n*.30);middle=int(n*.65)
model=fit(x[:cut],y[:cut]);records=[]
for name,lo,hi in [('validation',cut,middle),('final_transfer',middle,n)]:
 orig=x[lo:hi];dec=y[lo:hi];base=basepatch(orig,dec);candidate=predictor_patch(orig,dec,model)
 assert np.array_equal(unpatch(base,dec),orig)
 assert np.array_equal(unpatch(candidate,dec),orig)
 records.append({'segment':name,'n':len(orig),'baseline_bytes':len(base),'candidate_bytes':len(candidate),
                 'gain_pct':round(100*(len(base)-len(candidate))/len(base),5),
                 'bitwise_verified':True,'passes_5pct':len(candidate)<=.95*len(base)})
with (OUT/'fold_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
status='PASS_INTERNAL_FOLDS_ONLY' if all(r['passes_5pct'] for r in records) else 'REJECT_GENERALIZATION_HYPOTHESIS'
report={'status':'COMPLETE','research_gate':status,'results':records,'source_NB96_sha256':hashlib.sha256(raw).hexdigest(),
        'no_test_time_fitting':True,'novelty_established':False,
        'external_families_still_required':True,
        'caveat':'Same NYC dataset family; internal folds are NOT independent external replication.'}
(OUT/'decision.json').write_text(json.dumps(report,indent=2))

# Verified single-ZIP export — automatically downloads in Google Colab.
from pathlib import Path
import zipfile
archive = ROOT / (OUT.name.replace('_outputs', '_results') + '.zip')
result_files = sorted(p for p in OUT.rglob('*') if p.is_file())
if not result_files:
    raise RuntimeError('No experiment result files were written. Run the experiment cells first.')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for f in result_files:
        z.write(f, arcname=str(f.relative_to(OUT)))
with zipfile.ZipFile(archive) as z:
    failed = z.testzip()
    if failed is not None:
        raise RuntimeError('ZIP integrity check failed: ' + failed)
print(json.dumps(report, indent=2))
print('Results saved:', archive, '| files:', len(result_files), '| bytes:', archive.stat().st_size)
try:
    from google.colab import files as colab_files
    colab_files.download(str(archive))
except ImportError:
    print('Not running in Colab; ZIP saved at:', archive)

