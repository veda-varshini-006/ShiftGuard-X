#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB103 — Frozen January Predictor to February Transfer

Requires `SGX_NB102_results.zip` and `SGX_NB100_results.zip`. Uses **the unchanged 64-bit predictor from NB100**, trained only on January NYC data. Compares its February correction cost against the bit-exact NB100 matched baseline. Fails closed if February contains non-one-ULP errors; does **not** silently drop them. Includes metadata, model, masks, and correction payload. The output auto-downloads as `SGX_NB103_results.zip`. Success would show same-domain temporal transfer, **not algorithmic novelty or independent-domain replication**.
"""

import io,zipfile,json,csv,sys,subprocess,hashlib
from pathlib import Path
import numpy as np
try:import zstandard as zstd
except ImportError:
 subprocess.check_call([sys.executable,'-m','pip','-q','install','zstandard==0.25.0']);import zstandard as zstd
ROOT=Path('/content') if Path('/content').exists() else Path('/mnt/data')
OUT=ROOT/'SGX_NB103_outputs';OUT.mkdir(exist_ok=True)
try:from google.colab import files
except ImportError:files=None

def getzip(name):
 p=ROOT/name
 if not p.exists():
  if files is None:raise FileNotFoundError(name)
  print('Please upload',name)
  uploaded=files.upload()
  if name not in uploaded:raise FileNotFoundError(name)
  p.write_bytes(uploaded[name])
 return zipfile.ZipFile(p)
errors=[];evidence=None; model=None;source_model=None;external_meta=None
try:
 with getzip('SGX_NB102_results.zip') as z:
  external_meta=json.loads(z.read('decision.json'))
  if external_meta['status']!='COMPLETE':raise ValueError('NB102 incomplete')
  with np.load(io.BytesIO(z.read('external_bitwise_evidence.npz'))) as f:
   orig=f['original'].astype('<u8',copy=True);dec=f['decoded'].astype('<u8',copy=True)
  if orig.shape!=dec.shape or orig.shape[0]!=9:raise ValueError('Bad source shapes')
  expected=external_meta['float64_source_bits_sha256']
  if hashlib.sha256(orig.tobytes()).hexdigest()!=expected:raise ValueError('Evidence checksum mismatch')
  evidence=(orig[-1],dec[-1])
 with getzip('SGX_NB100_results.zip') as z:
  d=json.loads(z.read('decision.json'));params=json.loads(z.read('frozen_model.json'))
  if d['status']!='COMPLETE' or len(params['bits'])!=64:raise ValueError('Bad NB100 frozen model')
  model=np.asarray(params['bits'],dtype='u1');source_model=d['source_NB96_sha256']
except Exception as e:errors.append(repr(e))


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


records=[]
if evidence is not None and model is not None and not errors:
 x,y=evidence
 # NB100's signed-one-ULP hypothesis must be true on unseen February before comparing.
 delta=(x-y).astype('<u8');m=x!=y
 unit=np.logical_or(delta==np.uint64(1),delta==np.uint64(0xffffffffffffffff))
 nonunit=int(np.count_nonzero(m & (~unit)))
 if nonunit:errors.append(f'{nonunit} outside-one-ULP errors; NB100 mechanism cannot represent them exactly. No successful transfer claimed.')
 else:
  try:
   baseline=basepatch(x,y);candidate=predictor_patch(x,y,model)
   assert np.array_equal(unpatch(baseline,y),x)
   assert np.array_equal(unpatch(candidate,y),x)
   gain=100*(len(baseline)-len(candidate))/len(baseline)
   records=[{'dataset':'NYC-TLC-2024-02','n':len(x),'mismatches':int(m.sum()),'nonunit_errors':0,
             'frozen_model_source':source_model,'baseline_bytes':len(baseline),
             'candidate_bytes':len(candidate),'gain_pct':gain,'bits_verified':True}]
  except Exception as e:errors.append(repr(e))
state='COMPLETE' if records and not errors else 'INCOMPLETE'
report={'status':state,'data':'External February 2024 NYC taxi','model':'Frozen NB100 trained only on January 2024',
 'comparison':records,'errors':errors,'novelty_established':False,
 'external_domain_confirmed':False,'research_gate':('PASS_MONTH_TRANSFER' if state=='COMPLETE' and records[0]['gain_pct']>=5 else 'STOP_OR_REDESIGN')}
(OUT/'decision.json').write_text(json.dumps(report,indent=2))
with (OUT/'comparison.csv').open('w',newline='') as f:
 fields=list(records[0]) if records else ['status'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records if records else [{'status':'INCOMPLETE'}])
p=ROOT/'SGX_NB103_results.zip'
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
 for f in OUT.iterdir():z.write(f,f.name)
with zipfile.ZipFile(p) as z:assert z.testzip() is None
print(json.dumps(report,indent=2))
if files:files.download(str(p))

