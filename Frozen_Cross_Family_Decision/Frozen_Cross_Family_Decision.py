#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB51: Frozen Cross Family Decision

**Purpose:** Per-stream 25% calibration followed by 75% held-out testing on scikit-learn public numerical datasets. Applies a 5% frozen threshold and refuses to claim novelty.

**Instructions:** Run **Runtime → Run all** in a fresh Google Colab CPU runtime. The notebook writes `SGX_NB51_results.zip` and attempts to download it. Each notebook is independent. If network access or ALP fails, status and logs are still exported rather than inventing numbers.

**Research integrity:** Bitwise round-trip success is necessary but not sufficient for novelty, competitive compression, or a Q1 publication.
"""

import os,sys,json,csv,io,hashlib,time,zipfile,statistics,zlib,lzma,platform,traceback,subprocess,shutil,struct,math
from pathlib import Path
import numpy as np
ROOT=Path('/content') if Path('/content').exists() else Path('/mnt/data')
OUT=ROOT/'SGX_NB51_outputs';OUT.mkdir(parents=True,exist_ok=True)
def write_csv(name,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def write_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=str))
def export():
 out=ROOT/'SGX_NB51_results.zip'
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(OUT.rglob('*')):
   if f.is_file():z.write(f,f.relative_to(OUT))
 print('RESULTS_ZIP:',out,'BYTES:',out.stat().st_size)
 try:
  from google.colab import files
  files.download(str(out))
 except ImportError:pass
 except Exception as e:print('Automatic download unavailable; download manually:',out,repr(e))
 return out
def run(cmd,timeout=600,cwd=None,env=None):
 t=time.monotonic()
 try:
  p=subprocess.run(cmd,cwd=cwd,env=env,capture_output=True,text=True,timeout=timeout)
  return {'command':cmd,'returncode':p.returncode,'duration_s':round(time.monotonic()-t,3),'stdout_tail':p.stdout[-24000:],'stderr_tail':p.stderr[-12000:]}
 except Exception as e:return {'command':cmd,'returncode':-999,'duration_s':round(time.monotonic()-t,3),'error':repr(e)}
def raw(v):return np.ascontiguousarray(v,dtype='<f8').tobytes()
def sha(b):return hashlib.sha256(b).hexdigest()
write_json('environment.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'smoke':os.environ.get('SGX_LOCAL_SMOKE_ONLY')=='1'})

# Exploratory exact decimal lattice with raw-bit exceptions and compact integer bitpacking.
# Decimal scaling, exception streams and bitpacking all have prior art (including ALP-family ideas).
# No novelty or performance claims are made by this notebook.
SCALES=[1,10,100,1000,10000]
MAG=b'QPK1'
def bitpack(values,width):
  acc=0;pending=0;out=bytearray()
  for v in values:
   v=int(v)
   if v<0 or (width and v >= (1<<width)) or (not width and v):raise ValueError('Out-of-range packed value')
   acc|=v<<pending;pending+=width
   while pending>=8:
    out.append(acc&255);acc>>=8;pending-=8
  if pending:out.append(acc&255)
  return bytes(out)
def bitunpack(buf,width,count):
  acc=0;pending=0;it=iter(buf);out=[]
  for _ in range(count):
   while pending<width:
    try:acc|=next(it)<<pending
    except StopIteration:raise ValueError('Truncated packed values')
    pending+=8
   out.append(acc&((1<<width)-1));acc>>=width;pending-=width
  if any(it):raise ValueError('Trailing nonzero packed bytes')
  return out
def lattice_pack(r,scale):
 if len(r)%8:raise ValueError('misaligned input')
 a=np.frombuffer(r,dtype='<f8'); n=len(a)
 # Word equality rather than numeric equality handles -0.0 and NaN payloads.
 original=np.frombuffer(r,dtype='<u8')
 present=[];integers=[];exceptions=[]
 for i,v in enumerate(a):
  v=float(v);accept=False;k=0
  if math.isfinite(v) and -2147483648 <= v*scale <= 2147483647:
   candidate=int(round(v*scale))
   if -2147483648<=candidate<=2147483647:
    restored=float(candidate)/scale
    if struct.pack('<d',restored)==r[i*8:i*8+8]:accept=True;k=candidate
  if accept:present.append(1);integers.append(k)
  else:present.append(0);exceptions.append(r[i*8:i*8+8])
 mn=min(integers) if integers else 0
 shifted=[v-mn for v in integers]
 width=max(shifted,default=0).bit_length()
 packed=bitpack(shifted,width)
 marker=bitpack(present,1)
 sections=[zlib.compress(marker,6),zlib.compress(packed,6),zlib.compress(b''.join(exceptions),6)]
 # 4 magic, 1 scale index, 4 n, 4 accepted, 8 min, 1 width, 12 compressed section lengths
 header=MAG+struct.pack('<BIIqBIII',SCALES.index(scale),n,len(integers),mn,width,*[len(x) for x in sections])
 return header+b''.join(sections)
def lattice_unpack(blob):
 if blob[:4]!=MAG or len(blob)<34:raise ValueError('Bad header')
 k,n,naccept,mn,width,a,b,c=struct.unpack('<BIIqBIII',blob[4:34]);scale=SCALES[k]
 if n>100_000_000 or naccept>n or width>64:raise ValueError('Invalid parameters')
 if len(blob)!=34+a+b+c:raise ValueError('Length mismatch')
 p=34;marker=zlib.decompress(blob[p:p+a]);p+=a
 packed=zlib.decompress(blob[p:p+b]);p+=b
 rawexceptions=zlib.decompress(blob[p:p+c]);p+=c
 mask=bitunpack(marker,1,n)
 if sum(mask)!=naccept:raise ValueError('Mask count mismatch')
 numbers=bitunpack(packed,width,naccept)
 if len(rawexceptions)!=(n-naccept)*8:raise ValueError('Exception size mismatch')
 it=iter(numbers);e=0;out=bytearray()
 for tag in mask:
  if tag:
   val=(next(it)+mn)/scale
   out.extend(struct.pack('<d',val))
  else:out.extend(rawexceptions[e:e+8]);e+=8
 return bytes(out)
def generic_pack(mode,r):
 if mode=='zlib6':return zlib.compress(r,6)
 if mode=='lzma1':return lzma.compress(r,preset=1)
 if mode=='zstd3':
  import zstandard as zstd
  return zstd.ZstdCompressor(level=3).compress(r)
 raise ValueError(mode)
def generic_unpack(mode,b):
 if mode=='zlib6':return zlib.decompress(b)
 if mode=='lzma1':return lzma.decompress(b)
 if mode=='zstd3':
  import zstandard as zstd
  return zstd.ZstdDecompressor().decompress(b)
 raise ValueError(mode)
def trial(mode,r,reps=3):
 enc=lattice_pack if mode.startswith('lattice') else generic_pack
 sizes=[];et=[];dt=[]
 for _ in range(reps):
  t=time.perf_counter_ns()
  if mode.startswith('lattice'):b=lattice_pack(r,SCALES[int(mode.split('_')[1])])
  else:b=generic_pack(mode,r)
  et.append((time.perf_counter_ns()-t)/1e6)
  t=time.perf_counter_ns()
  decoded=lattice_unpack(b) if mode.startswith('lattice') else generic_unpack(mode,b)
  dt.append((time.perf_counter_ns()-t)/1e6)
  assert decoded==r,('lossless failure',mode)
  sizes.append(len(b))
 return {'bytes':sizes[0],'encode_ms_median':float(statistics.median(et)),'decode_ms_median':float(statistics.median(dt))}
def tests():
 edge_bits=np.array([0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0x7ff8000000001234,0x0000000000000001,0xffffffffffffffff],dtype='<u8')
 examples=[b'',edge_bits.tobytes(),edge_bits.tobytes()*8,raw(np.arange(512,dtype='<f8')/10),raw(np.zeros(200))]
 checks=[]
 for a in examples:
  for scale in SCALES:
   c=lattice_pack(a,scale);d=lattice_unpack(c)
   checks.append({'n':len(a)//8,'scale':scale,'success':d==a})
 assert all(x['success'] for x in checks)
 return checks

from sklearn.datasets import load_diabetes,load_wine,load_breast_cancer,load_digits
source_datasets=[('diabetes',load_diabetes().data),('wine_classification',load_wine().data),('breast_cancer',load_breast_cancer().data),('digits',load_digits().data)]
write_csv('edge_case_tests.csv',tests())
# Freeze per-stream selection on first 25%; evaluate only on chronological remaining 75%.
# Never let a held-out oracle determine the selection.
rows=[];failures=[];DETAIL=[]
base_modes=['zlib6','lzma1']
try:import zstandard;base_modes.append('zstd3')
except ImportError:pass
modes=base_modes+['lattice_'+str(i) for i in range(len(SCALES))]
for name,dataset in source_datasets:
 arr=np.asarray(dataset,dtype='<f8')
 for col in range(arr.shape[1]):
  vals=arr[:,col]; cut=max(24,len(vals)//4)
  if cut>=len(vals):continue
  cal=raw(vals[:cut]); held=raw(vals[cut:]);key=name+'__c'+str(col).zfill(2)
  try:
   calibr={m:trial(m,cal,1)['bytes'] for m in modes}
   ref_mode=min(base_modes,key=lambda m:calibr[m]); cand_mode=min((m for m in modes if m.startswith('lattice')),key=lambda m:calibr[m])
   # Conservative 5% margin on calibration, without using held-out values.
   selected=cand_mode if calibr[cand_mode]<=0.95*calibr[ref_mode] else ref_mode
   measurements={m:trial(m,held,3) for m in modes}
   oracle_base=min(base_modes,key=lambda m:measurements[m]['bytes'])
   for m,met in measurements.items():DETAIL.append({'stream':key,'dataset':name,'mode':m,'n_heldout':len(vals)-cut,'sha256':sha(held),**met})
   result=measurements[selected]['bytes'];bbest=measurements[oracle_base]['bytes']
   rows.append({'stream':key,'dataset':name,'n_cal':cut,'n_hold':len(vals)-cut,'sha256_heldout':sha(held),'cal_selected':selected,'cal_best_generic':ref_mode,'cal_best_candidate':cand_mode,'cal_candidate_bytes':calibr[cand_mode],'cal_best_generic_bytes':calibr[ref_mode],'heldout_selected_bytes':result,'heldout_oracle_generic_bytes':bbest,'gain_vs_heldout_generic_oracle_pct':100*(bbest-result)/bbest,'selected_lattice':selected.startswith('lattice')})
  except Exception as e:failures.append({'stream':key,'error':repr(e)})
write_csv('frozen_policy_results.csv',rows);write_csv('heldout_ablation.csv',DETAIL);write_csv('failures.csv',failures)
chosen=[x for x in rows if x['selected_lattice']]
summary={'status':'EVALUATED' if rows else 'FAILED','total_streams':len(rows),'lattice_selected_streams':len(chosen),'lattice_heldout_wins_vs_oracle':sum(x['gain_vs_heldout_generic_oracle_pct']>0 for x in chosen),'lattice_heldout_losses_vs_oracle':sum(x['gain_vs_heldout_generic_oracle_pct']<0 for x in chosen),'num_errors':len(failures),'comparison_caveat':'The heldout-best generic is a hindsight oracle, not an implementable baseline selector. Evaluate policy-vs-policy for publication.','algorithm_novelty':'NOT_ESTABLISHED','specialized_baseline_comparison':'NOT_AVAILABLE_IN_THIS_NOTEBOOK','q1_readiness':'NOT_ESTABLISHED'}
if chosen:
 summary['candidate_total_heldout_bytes']=sum(x['heldout_selected_bytes'] for x in chosen)
 summary['generic_oracle_total_bytes_for_chosen']=sum(x['heldout_oracle_generic_bytes'] for x in chosen)
summary['decision']='RETHINK_LATTICE' if not chosen or sum(x['gain_vs_heldout_generic_oracle_pct']>0 for x in chosen)<=sum(x['gain_vs_heldout_generic_oracle_pct']<0 for x in chosen) else 'FURTHER_BASELINE_AND_NOVELTY_WORK_REQUIRED'
write_json('research_gate.json',summary)
print('RESEARCH_GATE',json.dumps(summary,indent=2));export()

