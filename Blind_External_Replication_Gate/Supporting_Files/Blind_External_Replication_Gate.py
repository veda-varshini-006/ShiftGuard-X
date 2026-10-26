#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB69 — Blind_External_Replication_Gate

Locked 15% selection policy evaluated on three new UCI families. Source hashes, 25/75 split, exact decode, strong baselines, conservative independent-family bootstrap. No substitution on download failure.

**Run**: Google Colab CPU → Runtime → Run all. Each notebook is fully standalone and writes a results ZIP. A failed download generates an explicit incomplete decision, never a synthetic replacement. **No paper acceptance or novelty is asserted.**
"""

import sys,os,time,csv,json,io,struct,hashlib,statistics,zlib,lzma,platform,urllib.request,importlib.util,math,zipfile,traceback
from pathlib import Path
import numpy as np
NOTEBOOK_NO=69
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/f'SGX_NB{NOTEBOOK_NO}_outputs';OUT.mkdir(exist_ok=True,parents=True)
SMOKE=os.getenv('SGX_LOCAL_SMOKE_ONLY','0')=='1'
def emit_csv(name,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r.keys()))
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def emit_json(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=str))
def raw(v):return np.asarray(v,dtype='<f8').copy(order='C').tobytes()
def sha(x):return hashlib.sha256(x).hexdigest()
def deliver():
 filename=ROOT/f'SGX_NB{NOTEBOOK_NO}_results.zip'
 with zipfile.ZipFile(filename,'w',zipfile.ZIP_DEFLATED) as z:
  for f in OUT.rglob('*'):
   if f.is_file():z.write(f,f.relative_to(OUT))
 print('RESULT ZIP:',filename,'SIZE',filename.stat().st_size)
 try:
  from google.colab import files;files.download(str(filename))
 except ImportError:pass
 except Exception as e:print('Use Files sidebar to download ZIP:',e)
 return filename
emit_json('environment.json',{'notebook':NOTEBOOK_NO,'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'smoke_only':SMOKE,'no_synthetic_substitution_in_production':True})

# Attempt to include Zstandard as a stronger generic baseline. Failure is explicit.
if not importlib.util.find_spec('zstandard') and os.getenv('SGX_SKIP_OPTIONAL_INSTALL')!='1':
 import subprocess
 try:
  p=subprocess.run([sys.executable,'-m','pip','-q','install','zstandard'],capture_output=True,text=True,timeout=120)
  emit_json('zstd_dependency_install.json',{'returncode':p.returncode,'stderr_tail':p.stderr[-1000:]})
 except Exception as exc:emit_json('zstd_dependency_install.json',{'error':repr(exc)})
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

# Stronger complete-file alternatives; outer container header always included for every method.
METHODS=['zlib6','lzma1','zstd3','shuffle_zlib6','xor_zlib6','int_for','int_for_zlib','dict_bitpack','dict_zlib','float32_zlib']+['lattice_'+str(i) for i in range(5)]
METHODS=[m for m in METHODS if m!='zstd3' or importlib.util.find_spec('zstandard')]
NONLATTICE=[m for m in METHODS if not m.startswith('lattice_')]
CANDIDATES=[m for m in METHODS if m.startswith('lattice_')]
MIDS={name:i for i,name in enumerate(METHODS)}
INV={i:name for name,i in MIDS.items()}
FRAME=b'SGX5'
def _shuffle(r):return np.frombuffer(r,dtype=np.uint8).reshape(-1,8).T.copy().tobytes()
def _unshuffle(b):return np.frombuffer(b,dtype=np.uint8).reshape(8,-1).T.copy().tobytes()
def _xor(r):
 a=np.frombuffer(r,dtype='<u8');b=a.copy()
 if len(b)>1:b[1:]=a[1:]^a[:-1]
 return b.astype('<u8').tobytes()
def _unxor(r):
 a=np.frombuffer(r,dtype='<u8').copy();shift=1
 while shift<len(a):a[shift:]^=a[:-shift].copy();shift*=2
 return a.astype('<u8').tobytes()
def _integrals(r):
 a=np.frombuffer(r,dtype='<f8')
 if not np.all(np.isfinite(a)):raise ValueError('has nonfinite values')
 if np.any(a<-(2**31)) or np.any(a>(2**31-1)):raise ValueError('outside int32')
 v=a.astype('<i4')
 if v.astype('<f8').tobytes()!=r:raise ValueError('not bit-exact int32 (including signed zeros)')
 return v.astype(np.int64)
def _int_for_pack(r, zipped):
 a=_integrals(r);mn=int(a.min()) if len(a) else 0
 shift=(a-mn).astype(np.uint64);width=int(shift.max()).bit_length() if len(shift) else 0
 packed=bitpack(shift,width);return struct.pack('<qB',mn,width)+(zlib.compress(packed,6) if zipped else packed)
def _int_for_unpack(b,n,zipped):
 mn,w=struct.unpack('<qB',b[:9]);buf=zlib.decompress(b[9:]) if zipped else b[9:]
 vals=bitunpack(buf,w,n);x=np.array([mn+int(k) for k in vals],dtype=np.int64)
 if np.any(x<-(2**31)) or np.any(x>(2**31-1)):raise ValueError('invalid int')
 return x.astype('<f8').tobytes()
def _dict_pack(r,zipped):
 a=np.frombuffer(r,dtype='<u8');unique,ids=np.unique(a,return_inverse=True)
 if len(unique)>65535:raise ValueError('dictionary > 65535')
 w=max(0,(len(unique)-1).bit_length())
 p=struct.pack('<IB',len(unique),w)+unique.astype('<u8').tobytes()+bitpack(ids,w)
 return zlib.compress(p,6) if zipped else p
def _dict_unpack(b,n,zipped):
 data=zlib.decompress(b) if zipped else b
 k,w=struct.unpack('<IB',data[:5]);d=np.frombuffer(data,dtype='<u8',count=k,offset=5)
 idx=bitunpack(data[5+8*k:],w,n)
 if any(i>=k for i in idx):raise ValueError('bad dict id')
 return d[np.asarray(idx,dtype=np.int64)].astype('<u8').tobytes()
def encode_payload(m,r):
 if m.startswith('lattice_'):return lattice_pack(r,SCALES[int(m.split('_')[1])])
 if m in ('zlib6','lzma1','zstd3'):return generic_pack(m,r)
 if m=='shuffle_zlib6':return zlib.compress(_shuffle(r),6)
 if m=='xor_zlib6':return zlib.compress(_xor(r),6)
 if m=='int_for':return _int_for_pack(r,False)
 if m=='int_for_zlib':return _int_for_pack(r,True)
 if m=='dict_bitpack':return _dict_pack(r,False)
 if m=='dict_zlib':return _dict_pack(r,True)
 if m=='float32_zlib':
  a=np.frombuffer(r,dtype='<f8');b=a.astype('<f4')
  if b.astype('<f8').tobytes()!=r:raise ValueError('not bit-exact float32')
  return zlib.compress(b.tobytes(),6)
 raise ValueError('unknown method')
def decode_payload(m,b,n):
 if m.startswith('lattice_'):return lattice_unpack(b)
 if m in ('zlib6','lzma1','zstd3'):return generic_unpack(m,b)
 if m=='shuffle_zlib6':return _unshuffle(zlib.decompress(b))
 if m=='xor_zlib6':return _unxor(zlib.decompress(b))
 if m=='int_for':return _int_for_unpack(b,n,False)
 if m=='int_for_zlib':return _int_for_unpack(b,n,True)
 if m=='dict_bitpack':return _dict_unpack(b,n,False)
 if m=='dict_zlib':return _dict_unpack(b,n,True)
 if m=='float32_zlib':return np.frombuffer(zlib.decompress(b),dtype='<f4').astype('<f8').tobytes()
 raise ValueError('unknown method')
def encode(m,r):
 if len(r)%8:raise ValueError('unaligned')
 p=encode_payload(m,r)
 return FRAME+struct.pack('<BIQ',MIDS[m],len(r)//8,len(p))+p  # 17 bytes inclusive
# Fully self-describing stream; checked without external method context.
def decode(f):
 if f[:4]!=FRAME or len(f)<17:raise ValueError('frame')
 mid,n,nbytes=struct.unpack('<BIQ',f[4:17]);p=f[17:]
 if mid not in INV or len(p)!=nbytes:raise ValueError('body length / mode')
 b=decode_payload(INV[mid],p,n)
 if len(b)!=8*n:raise ValueError('decoded length')
 return b
def measured(m,r,reps=3):
 size=None;enc=[];dec=[]
 # warmup and take medians of measured real work; no cached encoding or selection time counted.
 for j in range(reps+1):
  st=time.perf_counter_ns();f=encode(m,r);a=(time.perf_counter_ns()-st)/1e6
  st=time.perf_counter_ns();b=decode(f);d=(time.perf_counter_ns()-st)/1e6
  if b!=r:raise AssertionError('BIT MISMATCH '+m)
  if j:size=len(f);enc.append(a);dec.append(d)
 return {'bytes':size,'bpv':8*size/(len(r)//8),'encode_ms_median':statistics.median(enc),'decode_ms_median':statistics.median(dec),'exact':True}
def all_edges():
 bits=np.array([0,0x8000000000000000,0x7ff8000000000001,0x7ff8000000000033,0x7ff0000000000000,0xfff0000000000000,0x0000000000000001,0xffffffffffffffff],dtype='<u8')
 examples={'empty':b'','special':bits.tobytes(),'repeated_special':bits.tobytes()*4,'integers':raw(np.arange(-20,20)),'decimal':raw(np.arange(50)/10)}
 out=[]
 for label,b in examples.items():
  for m in METHODS:
   try:
    c=encode(m,b);ok=decode(c)==b;out.append({'input':label,'mode':m,'status':'PASS' if ok else 'FAIL'})
   except ValueError as e:out.append({'input':label,'mode':m,'status':'INAPPLICABLE','reason':str(e)})
 assert not any(x['status']=='FAIL' for x in out)
 return out


MARGIN=0.15
GENERICS=[m for m in ('zlib6','lzma1','zstd3') if m in METHODS]
def choose(raw_cal):
 # All decisions depend on calibration input only. Charge encode and decode attempts to selection.
 records=[]; t0=time.perf_counter_ns()
 for mode in METHODS:
  st=time.perf_counter_ns()
  try:
   frame=encode(mode,raw_cal)
   if decode(frame)!=raw_cal:raise AssertionError('bitwise mismatch')
   records.append({'mode':mode,'bytes':len(frame),'elapsed_ms':(time.perf_counter_ns()-st)/1e6,'status':'PASS'})
  except (ValueError,OverflowError,FloatingPointError) as ex:
   records.append({'mode':mode,'bytes':None,'elapsed_ms':(time.perf_counter_ns()-st)/1e6,'status':'INAPPLICABLE','reason':str(ex)})
 total=(time.perf_counter_ns()-t0)/1e6
 ok=[r for r in records if r['status']=='PASS']
 if not ok:raise RuntimeError('No usable codec; generic always expected to work')
 non=[r for r in ok if r['mode'] in NONLATTICE]
 gen=[r for r in ok if r['mode'] in GENERICS]
 lat=[r for r in ok if r['mode'] in CANDIDATES]
 best_non=min(non,key=lambda r:r['bytes'])
 best_gen=min(gen,key=lambda r:r['bytes'])
 best_lat=min(lat,key=lambda r:r['bytes']) if lat else None
 picked=best_lat['mode'] if best_lat and best_lat['bytes']<=best_non['bytes']*(1-MARGIN) else best_non['mode']
 selected_ms=sum(r['elapsed_ms'] for r in records if r['status']=='PASS' or r['status']=='INAPPLICABLE')
 base_ms=sum(r['elapsed_ms'] for r in records if r['mode'] in NONLATTICE)
 return {'policy_method':picked,'baseline_method':best_non['mode'],'generic_safe_method':best_gen['mode'],
         'calibration_total_ms':total,'calibration_policy_sum_ms':selected_ms,'calibration_baseline_sum_ms':base_ms,
         'calibration_records':records}
def with_fallback(mode,r,safe_mode):
 st=time.perf_counter_ns();fallback=False;reason=''
 try:
  frame=encode(mode,r);decode_frame=decode(frame)
  if decode_frame!=r:raise AssertionError('bit mismatch')
 except (ValueError,OverflowError,FloatingPointError) as ex:
  fallback=True;reason=repr(ex)
  frame=encode(safe_mode,r)
  if decode(frame)!=r:raise AssertionError('fallback bit mismatch')
 elapsed_ms=(time.perf_counter_ns()-st)/1e6
 # Timing includes failed encoding and successful fallback work.
 t=time.perf_counter_ns();d=decode(frame);decode_ms=(time.perf_counter_ns()-t)/1e6
 assert d==r
 return {'requested':mode,'actual':safe_mode if fallback else mode,'fallback':fallback,
         'reason':reason,'bytes':len(frame),'encode_ms':elapsed_ms,'decode_ms':decode_ms,'bit_exact':True}
def eval_stream(dataset,column,arr,source_sha,subset_note=''):
 a=np.asarray(arr,dtype='<f8')
 if len(a)<80: raise ValueError('fewer than 80 values')
 cut=max(32,len(a)//4);cal=raw(a[:cut]);held=raw(a[cut:]);meta=choose(cal)
 policy=with_fallback(meta['policy_method'],held,meta['generic_safe_method'])
 base=with_fallback(meta['baseline_method'],held,meta['generic_safe_method'])
 # Hindsight: evaluate every non-lattice choice on heldout, purely diagnostic. Never use it for selection.
 sizes={};inapplicable=[]
 for m in METHODS:
  try:
   f=encode(m,held)
   if decode(f)!=held:raise AssertionError('decode mismatch')
   sizes[m]=len(f)
  except (ValueError,OverflowError,FloatingPointError) as ex:inapplicable.append({'mode':m,'reason':str(ex)})
 best_non=min(NONLATTICE,key=lambda m:sizes.get(m,float('inf')))
 best_all=min(METHODS,key=lambda m:sizes.get(m,float('inf')))
 return {'dataset':dataset,'column':column,'n_total':len(a),'n_cal':cut,'n_test':len(a)-cut,
         'input_sha256':source_sha,'heldout_sha256':sha(held),'subset':subset_note,
         'selected_method':meta['policy_method'],'actual_method':policy['actual'],'baseline_method':meta['baseline_method'],
         'policy_bytes':policy['bytes'],'baseline_bytes':base['bytes'],
         'oracle_nonlattice_bytes':sizes[best_non],'oracle_nonlattice_method':best_non,
         'oracle_all_bytes':sizes[best_all],
         'policy_saving_vs_frozen_base_pct':100*(base['bytes']-policy['bytes'])/base['bytes'],
         'policy_saving_vs_oracle_nonlattice_pct':100*(sizes[best_non]-policy['bytes'])/sizes[best_non],
         'policy_encode_ms':policy['encode_ms'],'baseline_encode_ms':base['encode_ms'],
         'policy_decode_ms':policy['decode_ms'],'baseline_decode_ms':base['decode_ms'],
         'policy_selection_ms':meta['calibration_policy_sum_ms'],'baseline_selection_ms':meta['calibration_baseline_sum_ms'],
         'policy_encode_plus_selection_ms':policy['encode_ms']+meta['calibration_policy_sum_ms'],
         'baseline_encode_plus_selection_ms':base['encode_ms']+meta['calibration_baseline_sum_ms'],
         'fallback':policy['fallback'],'fallback_reason':policy['reason'],'baseline_fallback':base['fallback'],
         'inapplicable_methods':';'.join(q['mode'] for q in inapplicable),'bit_exact':True}
def aggregate(rows):
 if not rows:return {}
 a=sum(int(r['policy_bytes']) for r in rows); b=sum(int(r['baseline_bytes']) for r in rows)
 o=sum(int(r['oracle_nonlattice_bytes']) for r in rows)
 return {'streams':len(rows),'policy_bytes':a,'frozen_baseline_bytes':b,'oracle_nonlattice_bytes':o,
         'aggregate_saving_vs_frozen_baseline_percent':100*(b-a)/b,'aggregate_saving_vs_oracle_nonlattice_percent':100*(o-a)/o,
         'wins_vs_frozen_baseline':sum(r['policy_bytes']<r['baseline_bytes'] for r in rows),
         'losses_vs_frozen_baseline':sum(r['policy_bytes']>r['baseline_bytes'] for r in rows),
         'fallbacks':sum(bool(r['fallback']) for r in rows),
         'policy_selection_ms':sum(r['policy_selection_ms'] for r in rows),
         'policy_encode_ms':sum(r['policy_encode_ms'] for r in rows),
         'frozen_baseline_selection_ms':sum(r['baseline_selection_ms'] for r in rows),
         'frozen_baseline_encode_ms':sum(r['baseline_encode_ms'] for r in rows)}
def edge_test():
 b=np.array([0,0x8000000000000000,0x7ff8000000000001,0x7ff0000000000000,
             0xfff0000000000000,0x0000000000000001,0xffffffffffffffff],dtype='<u8').tobytes()
 checks=[]
 for m in METHODS:
  try:
   f=encode(m,b);status='PASS' if decode(f)==b else 'FAIL'
  except (ValueError,OverflowError,FloatingPointError) as ex:status='INAPPLICABLE'
  checks.append({'method':m,'status':status})
 assert not any(k['status']=='FAIL' for k in checks)
 return checks


from collections import defaultdict
import urllib.request
# These three UCI dataset families are not present in prior SGX experiments.
# The first 2048 rows (or fewer) are used without inspecting results; family IDs are pinned here.
SPEC=[
 ('letter_recognition','https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data','letter'),
 ('magic_gamma','https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data','magic'),
 ('protein_CASP','https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv','casp')]
emit_csv('edge_tests.csv',edge_test())
def external_samples():
 if SMOKE:
  # Smoke is opt-in, explicitly synthetic, and not used for any real-data claim.
  rng=np.random.default_rng(20261023)
  for name,_,_ in SPEC:
   a=rng.integers(0,60,size=(180,3)).astype('<f8');content=raw(a.flatten())
   yield name,a,{'dataset':name,'source':'SYNTHETIC_LOCAL_SMOKE_ONLY','sha256_download':sha(content),
                 'raw_rows':180,'rows_used':180,'features':3,'SMOKE':True}
  return
 for name,url,kind in SPEC:
  try:
   source=url;member=''
   request=urllib.request.Request(url,headers={'User-Agent':'SGX-NB69/1.0'})
   try:
    with urllib.request.urlopen(request,timeout=90) as resp:payload=resp.read()
    text=payload.decode('utf-8-sig')
   except Exception as first_error:
    # The UCI archive provides the same dataset via an official static zip endpoint.
    # This fallback applies ONLY to the MAGIC UCI family and is always recorded.
    if kind!='magic': raise first_error
    source='https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip'
    req2=urllib.request.Request(source,headers={'User-Agent':'SGX-NB69/1.0'})
    with urllib.request.urlopen(req2,timeout=90) as resp:payload=resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
     matches=[n for n in z.namelist() if n.lower().endswith('magic04.data')]
     if not matches:raise ValueError('official MAGIC ZIP did not contain magic04.data')
     member=matches[0];text=z.read(member).decode('utf-8-sig')
   r=csv.reader(io.StringIO(text))
   out=[];row_count=0;skips=0
   for row in r:
    if not row:continue
    if kind=='casp' and row_count==0 and row[0].strip().upper()=='RMSD':
     row_count+=1;continue
    row_count+=1
    try:
     if kind=='letter': vals=[float(x) for x in row[1:17]]
     elif kind=='magic': vals=[float(x) for x in row[:10]]
     else: vals=[float(x) for x in row[:10]]
     count_expected=16 if kind=='letter' else 10
     if len(vals)!=count_expected: raise ValueError('column count')
     if len(out)<2048:out.append(vals)
    except (ValueError,IndexError):skips+=1
   if skips>0:raise ValueError(f'{name}: {skips} malformed numerical rows; no silent removal permitted')
   a=np.asarray(out,dtype='<f8')
   if len(a)<80:raise ValueError(f'{name}: too few rows')
   yield name,a,{'dataset':name,'source':source,'source_member':member,'sha256_download':sha(payload),'download_bytes':len(payload),
                 'raw_rows':row_count,'rows_used':len(a),'features':a.shape[1],'SMOKE':False}
  except Exception as ex:
   yield name,None,{'dataset':name,'source':url,'error':repr(ex),'SMOKE':False}
rows=[];provenance=[];errors=[];dataset_success=[]
try:
 for name,a,meta in external_samples():
  provenance.append(meta)
  if a is None:
   errors.append({'dataset':name,'error':meta['error']});dataset_success.append({'dataset':name,'expected_columns':None,'evaluated_columns':0,'completed':False});continue
  nrows=0
  for j in range(a.shape[1]):
   try:
    r=eval_stream(name,f'f{j:02d}',a[:,j].copy(),meta['sha256_download'],
       'FIRST 2048 ROWS OR FEWER in source order; 25% calibration, 75% heldout')
    rows.append(r);nrows+=1
   except Exception as e:
    errors.append({'dataset':name,'column':j,'error':repr(e)})
  dataset_success.append({'dataset':name,'expected_columns':a.shape[1],'evaluated_columns':nrows,
                          'completed':nrows==a.shape[1]})
except Exception as e:errors.append({'dataset':'DATASET_LOADING','error':repr(e)})
emit_csv('external_dataset_provenance.csv',provenance)
emit_csv('external_frozen_results.csv',rows)
emit_csv('errors.csv',errors)
emit_csv('dataset_completion.csv',dataset_success)
family=[]
for name,_,_ in SPEC:
 rr=[r for r in rows if r['dataset']==name]
 if rr:family.append({'dataset':name,**aggregate(rr)})
emit_csv('family_summary.csv',family)
all_stat=aggregate(rows)
# Family bootstrap: only three independent families, so a CI is very uncertain.
if len(family)==3:
 rng=np.random.default_rng(417);bootstrap=[]
 for i in range(2000):
  sam=rng.integers(0,3,size=3)
  p=sum(family[q]['policy_bytes'] for q in sam);b=sum(family[q]['frozen_baseline_bytes'] for q in sam)
  bootstrap.append(100*(b-p)/b)
 all_stat['family_bootstrap_95_percent_CI']=[float(np.percentile(bootstrap,2.5)),float(np.percentile(bootstrap,97.5))]
complete=(len(dataset_success)==3 and all(z['completed'] for z in dataset_success) and not errors and not SMOKE)
all_stat.update({'state':'COMPLETE_REAL_EXTERNAL' if complete else ('SMOKE_ONLY_NOT_RESEARCH' if SMOKE else 'INCOMPLETE_EXTERNAL'),
 'dataset_families_complete':sum(z['completed'] for z in dataset_success),'failed_load_or_stream_errors':len(errors),
 'novelty_claim':False,'matched_ALP_native':False,
 'locked_before_dataset_fetch':{'method':'NB52 exact QPK1 + frozen calibration selector','margin_percent':15,'calibration_fraction':.25,'sample_cap_rows':2048},
 'bootstrap_warning':'Only three family clusters, so the interval is descriptive; stream-level samples are not independent.',
 'publication_gate':'NOT_READY: native ALP matched API and distinct novelty are absent, even if external sizing improves.'})
emit_json('decision.json',all_stat)
print('NB69 decision:',json.dumps(all_stat,indent=2))
deliver()

