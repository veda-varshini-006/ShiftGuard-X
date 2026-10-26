#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB86 — Precision-Stratification Falsification Gate

**Purpose:** screen one narrow hypothesis for exactly lossless IEEE-754 float64 streams. This is NOT a demonstrated novel algorithm, Q1 submission, or fair ALP benchmark. The data families below have been seen in earlier SGX research; the 25/75 split is within familiar data, **not blind external confirmation**.

**Question:** Can reversible binary16/binary32/binary64 strata with explicit 2-bit membership tags reduce real compressed size versus exactly matching full-file generic and specialized baselines, after all group headers are paid?

**Critical controls:** whole-column exact float16/float32, dictionary on original IEEE bits, shuffle+codec, ordinary generic compression, random permutation of group memberships, and a high-precision-only negative case. NaN payloads and negative zeros must round trip identically. The first quarter is calibration; settings are frozen for the final three quarters. Candidate is chosen only when ≥5% smaller in calibration.

**Predefined decision:** even a successful development run does not prove originality. A later, untouched, independent evaluation on ≥3 new families must show ≥5% aggregate savings, improvement on ≥2 families, no bit errors, and a distinct mechanism versus closest prior art, including Apache Parquet ALP, byte-stream split, dictionary and precision-width coding. Timing and native baselines must follow.
"""

# Standalone Colab or local Python3; no Google Drive and no prior notebooks required.
import os, sys, json, time, hashlib, csv, zipfile, struct, zlib, lzma, platform, warnings
from pathlib import Path
import numpy as np
from sklearn.datasets import load_diabetes, load_wine, load_breast_cancer, load_digits
try:
 import zstandard as zstd
except ImportError:
 zstd = None
 if Path('/content').exists():
  try:
   import subprocess
   subprocess.run([sys.executable,'-m','pip','-q','install','zstandard'],check=True,timeout=90)
   import zstandard as zstd
  except Exception as e: print('Zstandard unavailable; that baseline will be marked unavailable:',repr(e))
ROOT=Path('/content') if Path('/content').exists() else Path('/mnt/data')
OUT=ROOT/'SGX_NB86_outputs';OUT.mkdir(parents=True,exist_ok=True)
print('Zstandard available:',zstd is not None)
def save_csv(name, rows):
 if rows:
  with (OUT/name).open('w', newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def export_results():
 target=ROOT/'SGX_NB86_results.zip'
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(OUT.glob('*')):
   if p.is_file():z.write(p,p.name)
 print('RESULTS ZIP:',target,'bytes:',target.stat().st_size)
 try:
  from google.colab import files
  files.download(str(target))
 except ImportError:pass
 return target


# Lossless, self-describing bit-exact codec. This does NOT establish algorithmic novelty.
MODE_NAMES=['zlib','lzma','shuffle_zlib','dict_zlib','f16_exact','f32_exact','precision_zlib']
if zstd is not None:MODE_NAMES += ['zstd','shuffle_zstd','precision_zstd']
BASELINES=[m for m in MODE_NAMES if not m.startswith('precision_')]
CANDIDATES=[m for m in MODE_NAMES if m.startswith('precision_')]
IDS={m:i+1 for i,m in enumerate(MODE_NAMES)}
REVID={v:k for k,v in IDS.items()}
def raw64(arr):return np.ascontiguousarray(arr,dtype='<f8').tobytes()
def classes(raw):
 v=np.frombuffer(raw,dtype='<f8');original=v.view('<u8')
 with np.errstate(over='ignore',invalid='ignore',under='ignore'):
  h=v.astype('<f2').astype('<f8').view('<u8')==original
  s=v.astype('<f4').astype('<f8').view('<u8')==original
 # Nonfinite data always stays bitwise raw (including distinct NaN payloads).
 finite=np.isfinite(v);tags=np.full(len(v),2,dtype=np.uint8)
 tags[finite&s]=1;tags[finite&h]=0
 return tags
def tagpack(tags):
 out=np.zeros((len(tags)+3)//4,dtype=np.uint8)
 for i in range(4):
  vals=tags[i::4]
  if len(vals): out[:len(vals)] |= vals.astype(np.uint8) << (2*i)
 return out.tobytes()
def tagunpack(b,n):
 data=np.frombuffer(b,dtype=np.uint8);out=np.empty(n,dtype=np.uint8)
 if len(data)!=(n+3)//4:raise ValueError('bad tag length')
 for i in range(4):out[i::4] = (data[:len(out[i::4])]>>(2*i))&3
 if np.any(out>2):raise ValueError('bad tag')
 return out
def compress_bytes(raw,which):
 if which=='zlib':return zlib.compress(raw,6)
 if which=='zstd':return zstd.ZstdCompressor(level=3).compress(raw)
 raise ValueError(which)
def decompress_bytes(data,which):
 if which=='zlib':return zlib.decompress(data)
 if which=='zstd':return zstd.ZstdDecompressor().decompress(data)
 raise ValueError(which)
def prec_encode(raw,kind):
 n=len(raw)//8;v=np.frombuffer(raw,dtype='<f8');tags=classes(raw)
 payloads=[tagpack(tags), v[tags==0].astype('<f2').tobytes(),v[tags==1].astype('<f4').tobytes(),v[tags==2].astype('<f8').tobytes()]
 compressed=[compress_bytes(p,kind) for p in payloads]
 # 24-byte self-describing header includes count and all payload lengths.
 return struct.pack('<4sI4I',b'BPS1',n,*[len(p) for p in compressed])+b''.join(compressed)
def prec_decode(buf,kind):
 if len(buf)<24:raise ValueError('short frame')
 magic,n,*lens=struct.unpack('<4sI4I',buf[:24])
 if magic!=b'BPS1' or sum(lens)!=len(buf)-24:raise ValueError('bad frame')
 offset=24; payloads=[]
 for length in lens:
  payloads.append(decompress_bytes(buf[offset:offset+length],kind));offset+=length
 tags=tagunpack(payloads[0],n)
 for tag,width in ((0,2),(1,4),(2,8)):
  if len(payloads[tag+1])!=int(np.sum(tags==tag))*width:raise ValueError('payload size mismatch')
 # Never parse the raw exception stratum as numerical float; preserve NaN payloads.
 decoded=np.empty(n,dtype='<u8')
 decoded[tags==0]=np.frombuffer(payloads[1],dtype='<f2').astype('<f8').view('<u8')
 decoded[tags==1]=np.frombuffer(payloads[2],dtype='<f4').astype('<f8').view('<u8')
 decoded[tags==2]=np.frombuffer(payloads[3],dtype='<u8')
 return decoded.tobytes()
def byteshuffle(raw):return np.frombuffer(raw,dtype=np.uint8).reshape(-1,8).T.copy().tobytes()
def byteunshuffle(raw):return np.frombuffer(raw,dtype=np.uint8).reshape(8,-1).T.copy().tobytes()
def standard_encode(mode,raw):
 n=len(raw)//8;v=np.frombuffer(raw,dtype='<f8')
 if mode=='zlib':body=zlib.compress(raw,6)
 elif mode=='lzma':body=lzma.compress(raw,preset=1)
 elif mode=='zstd':body=compress_bytes(raw,'zstd')
 elif mode=='shuffle_zlib':body=zlib.compress(byteshuffle(raw),6)
 elif mode=='shuffle_zstd':body=compress_bytes(byteshuffle(raw),'zstd')
 elif mode in ('f16_exact','f32_exact'):
  tag=classes(raw);allowed=0 if mode=='f16_exact' else 1
  if np.any(tag>allowed):raise ValueError('whole-stream exact width not applicable')
  body=v.astype('<f2' if allowed==0 else '<f4').tobytes()
 elif mode=='dict_zlib':
  uniq,inv=np.unique(np.frombuffer(raw,dtype='<u8'),return_inverse=True)
  if len(uniq)>65535:raise ValueError('dictionary too large')
  w=1 if len(uniq)<=256 else 2
  tokens=inv.astype('<u1' if w==1 else '<u2').tobytes()
  body=zlib.compress(struct.pack('<HB',len(uniq),w)+uniq.astype('<u8').tobytes()+tokens,6)
 else:raise ValueError(mode)
 return b'BAS1'+struct.pack('<BI',IDS[mode],n)+body

def standard_decode(blob):
 if len(blob)<9 or blob[:4]!=b'BAS1':raise ValueError('invalid baseline frame')
 idx,n=struct.unpack('<BI',blob[4:9]);mode=REVID.get(idx);body=blob[9:]
 if mode=='zlib':out=zlib.decompress(body)
 elif mode=='lzma':out=lzma.decompress(body)
 elif mode=='zstd':out=decompress_bytes(body,'zstd')
 elif mode in ('shuffle_zlib','shuffle_zstd'):
  x=zlib.decompress(body) if mode=='shuffle_zlib' else decompress_bytes(body,'zstd');out=byteunshuffle(x)
 elif mode in ('f16_exact','f32_exact'):
  width='<f2' if mode=='f16_exact' else '<f4';out=np.frombuffer(body,dtype=width).astype('<f8').tobytes()
 elif mode=='dict_zlib':
  dat=zlib.decompress(body)
  if len(dat)<3:raise ValueError('short dictionary')
  k,w=struct.unpack('<HB',dat[:3]);d=np.frombuffer(dat,dtype='<u8',offset=3,count=k)
  if w not in (1,2):raise ValueError('bad dictionary width')
  toks=np.frombuffer(dat[3+8*k:],dtype='<u1' if w==1 else '<u2')
  if len(toks)!=n or (len(toks) and np.max(toks)>=k):raise ValueError('bad dictionary tokens')
  out=d[toks].astype('<u8').tobytes()
 else:raise ValueError('bad mode')
 if len(out)!=n*8:raise ValueError('wrong decoded length')
 return out

def encode(m,raw):return prec_encode(raw,m.split('_')[-1]) if m.startswith('precision_') else standard_encode(m,raw)
def decode(m,frame):return prec_decode(frame,m.split('_')[-1]) if m.startswith('precision_') else standard_decode(frame)
def measure(m,raw,reps=3):
 readings=[]
 for _ in range(reps):
  t=time.perf_counter_ns();packed=encode(m,raw);enc=(time.perf_counter_ns()-t)/1e6
  t=time.perf_counter_ns();d=decode(m,packed);dec=(time.perf_counter_ns()-t)/1e6
  if d!=raw:raise AssertionError('BITWISE MISMATCH')
  readings.append((len(packed),enc,dec))
 return {'size':readings[0][0],'encode_ms':float(np.median([x[1] for x in readings])),'decode_ms':float(np.median([x[2] for x in readings]))}


import math
from collections import Counter
# Include arbitrary-payload NaNs, -0, infinity, subnormals, dense random normal values.
special=np.array([0,0x8000000000000000,0x7ff8000000000001,0x7ff800000000face,0x7ff0000000000000,0xfff0000000000000,1,0x8000000000000001,0x3ff0000000000000,0x3fd5555555555555],dtype='<u8').tobytes()
edge=[]
for m in MODE_NAMES:
 try:
  v=encode(m,special);ok=decode(m,v)==special
  edge.append({'mode':m,'result':'PASS' if ok else 'FAIL','detail':''})
  if not ok:raise AssertionError(m)
 except ValueError as e:edge.append({'mode':m,'result':'INAPPLICABLE','detail':str(e)})
assert not any(r['result']=='FAIL' for r in edge)
save_csv('edge_cases.csv',edge)
# Fixed development families; held-out means later rows within these KNOWN families, NOT independent validation.
data=[('diabetes',load_diabetes().data),('wine_classification',load_wine().data),('breast_cancer',load_breast_cancer().data),('digits',load_digits().data)]
selected=[]; details=[]; errors=[]; provenance=[]
for family,matrix in data:
 a=np.asarray(matrix,dtype='<f8')
 provenance.append({'family':family,'rows':a.shape[0],'columns':a.shape[1],'sha256_all_float64':hashlib.sha256(a.tobytes()).hexdigest(),'source':'sklearn bundled dataset, real data, seen in prior SGX work'})
 for c in range(a.shape[1]):
  name=f'{family}__c{c:02d}';v=np.ascontiguousarray(a[:,c]);n=len(v)
  cut=max(32,n//4);cal=raw64(v[:cut]);hold=raw64(v[cut:]);
  calibration={};heldout={}
  for m in MODE_NAMES:
   try:
    calibration[m]=measure(m,cal,2)
    heldout[m]=measure(m,hold,2)
    details.append({'stream':name,'family':family,'method':m,'cal_bytes':calibration[m]['size'],'hold_bytes':heldout[m]['size'],'hold_encode_ms':heldout[m]['encode_ms'],'hold_decode_ms':heldout[m]['decode_ms'],'exact':True})
   except ValueError as e:
    errors.append({'stream':name,'method':m,'status':'INAPPLICABLE','reason':str(e)})
   except Exception as e:
    errors.append({'stream':name,'method':m,'status':'ERROR','reason':repr(e)})
  if not any(m in calibration for m in BASELINES):raise RuntimeError('no valid baseline')
  base=min((m for m in BASELINES if m in calibration),key=lambda m:calibration[m]['size'])
  viable=[m for m in CANDIDATES if m in calibration]
  cand=min(viable,key=lambda m:calibration[m]['size']) if viable else None
  # Calibrate policy before held-out, with a fixed 5% minimum margin.
  choose=cand if cand and calibration[cand]['size']<=.95*calibration[base]['size'] else base
  oracle=min((m for m in BASELINES if m in heldout),key=lambda m:heldout[m]['size'])
  bv=heldout[base]['size'];pv=heldout[choose]['size'];cv=heldout[cand]['size'] if cand else None
  selected.append({'stream':name,'family':family,'n_cal':cut,'n_hold':n-cut,'base_cal':base,'candidate_cal':cand or 'NONE','selected':choose,'base_heldout_bytes':bv,'policy_heldout_bytes':pv,'candidate_heldout_bytes':cv,'oracle_baseline_heldout_bytes':heldout[oracle]['size'],'candidate_selected':choose in CANDIDATES,'gain_vs_cal_baseline_pct':100*(bv-pv)/bv,'gain_vs_oracle_baseline_pct':100*(heldout[oracle]['size']-pv)/heldout[oracle]['size'],'hold_sha256':hashlib.sha256(hold).hexdigest()})
save_csv('development_measurements.csv',details);save_csv('frozen_decisions.csv',selected);save_csv('inapplicable_or_errors.csv',errors);save_csv('dataset_provenance.csv',provenance)
# Labeled synthetic positive control and shuffled-order negative control, never pooled with real data.
rng=np.random.default_rng(20261024)
synthetic=[]
a=rng.normal(size=2048).astype('<f2').astype('<f8');b=rng.normal(size=2048).astype('<f4').astype('<f8');c=rng.normal(size=2048).astype('<f8')
for scenario,values in [('precision_strata_contiguous',np.concatenate([a,b,c])),('precision_strata_row_permuted',rng.permutation(np.concatenate([a,b,c]))),('high_precision_only',c),('half_precision_only',a)]:
 raw=raw64(values)
 for m in MODE_NAMES:
  try:
   r=measure(m,raw,2);synthetic.append({'scenario':scenario,'data_kind':'SIMULATED_CONTROL','mode':m,'bytes':r['size'],'bit_exact':True})
  except ValueError:pass
save_csv('synthetic_controls.csv',synthetic)
base_total=sum(r['base_heldout_bytes'] for r in selected);policy_total=sum(r['policy_heldout_bytes'] for r in selected)
candidate_wins=sum(r['candidate_heldout_bytes'] is not None and r['candidate_heldout_bytes']<r['oracle_baseline_heldout_bytes'] for r in selected)
policy_wins=sum(r['policy_heldout_bytes']<r['base_heldout_bytes'] for r in selected)
family_wins=[]
for family in sorted(set(r['family'] for r in selected)):
 rows=[r for r in selected if r['family']==family]
 v=sum(x['base_heldout_bytes'] for x in rows);w=sum(x['policy_heldout_bytes'] for x in rows)
 family_wins.append({'family':family,'baseline_bytes':v,'policy_bytes':w,'gain_percent':100*(v-w)/v})
save_csv('family_summary.csv',family_wins)
# Strict gate: two positive independent families and at least 3% gain; these reused development families cannot satisfy it.
summary={'project':'ShiftGuard-X','notebook':86,'research_scope':'feasibility only, NOT a novelty proof','mechanism':'bit-exact binary-precision stratification of float64 with grouped payloads','real_families':len(data),'real_streams':len(selected),'candidate_selected_count':sum(r['candidate_selected'] for r in selected),'candidate_wins_vs_heldout_baseline_oracle':candidate_wins,'policy_wins_vs_cal_baseline':policy_wins,'calibrated_baseline_bytes':base_total,'policy_bytes':policy_total,'gain_percent':100*(base_total-policy_total)/base_total,'exception_count':sum(e['status']=='ERROR' for e in errors),'zstd_available':zstd is not None,'synthetic_not_used_for_go_decision':True,'preregistered_go_conditions':{'min_real_family_count':3,'min_distinct_unseen_family_wins':2,'aggregate_gain_percent':5,'lossless_mismatches':0,'closest_prior_art_match_required':True},'state':'DEVELOPMENT_DIAGNOSTIC_ONLY','decision':'DO_NOT_CLAIM_ORIGINALITY; REQUIRE_UNSEEN_DATA_AND_CLOSEST_BASELINE' }
(OUT/'decision.json').write_text(json.dumps(summary,indent=2))
(OUT/'environment.json').write_text(json.dumps({'python':sys.version,'numpy':np.__version__,'zstandard':getattr(zstd,'__version__',None),'platform':platform.platform()},indent=2))
print('RESULTS',json.dumps(summary,indent=2));print('FAMILY RESULTS',family_wins)
export_results()

