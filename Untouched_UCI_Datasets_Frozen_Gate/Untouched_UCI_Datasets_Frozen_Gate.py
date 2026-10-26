#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB54 — Untouched_UCI_Datasets_Frozen_Gate

Use *new real UCI dataset families* Abalone, Airfoil Self-Noise, and Bike Sharing (hourly) with fixed 25% prefix calibration / 75% held-out. Freeze 5% selection margin; compare lattice to an expanded specialized baseline oracle, include full size overhead, verify bitwise decoding and report missing data explicitly.

This is a standalone Google Colab CPU notebook. Execute **Runtime → Run all**, then download its `SGX_NB54_results.zip`. The notebook never substitutes synthetic data if a real dataset cannot be downloaded.
"""

import os,sys,io,csv,json,struct,hashlib,time,zipfile,statistics,traceback,math,zlib,lzma,platform,urllib.request,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/'SGX_NB54_outputs';OUT.mkdir(parents=True,exist_ok=True)
def emit_csv(name, rows):
 if not rows:return
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def emit_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=str))
def deliver():
 file=ROOT/'SGX_NB54_results.zip'
 with zipfile.ZipFile(file,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(OUT.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(OUT))
 print('RESULT ZIP',file,'bytes',file.stat().st_size)
 try:
  from google.colab import files
  files.download(str(file))
 except ImportError:pass
 except Exception as e:print('Download manually if needed:',repr(e))
 return file
def raw(v):return np.ascontiguousarray(v,dtype='<f8').tobytes()
def sha(b):return hashlib.sha256(b).hexdigest()
def env():emit_json('environment.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'zstandard_installed':importlib.util.find_spec('zstandard') is not None,'NB50_reference_reused':True,'source_data_substitution':'NEVER'})


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


# New UCI dataset families not used in prior NB27–NB51 notebook evaluations.
# No synthetic substitute and no silently replacing failed families.
EXTERNAL_SOURCES={
 'abalone':[
  'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',
  'https://archive.ics.uci.edu/static/public/1/abalone.zip'],
 'airfoil':[
  'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat',
  'https://archive.ics.uci.edu/static/public/291/airfoil+self-noise.zip'],
 'bike_hourly':[
  'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip',
  'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip']}
def download_family(family):
 errors=[]
 for link in EXTERNAL_SOURCES[family]:
  try:
   with urllib.request.urlopen(link,timeout=60) as resp:blob=resp.read()
   if blob[:2]==b'PK':
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
     target={'abalone':'abalone.data','airfoil':'airfoil_self_noise.dat','bike_hourly':'hour.csv'}[family]
     names=[name for name in z.namelist() if name.lower().endswith(target.lower())]
     if not names:raise ValueError('target file not found in zip: '+str(z.namelist()[:10]))
     return z.read(names[0]),link,sha(blob)
   if family=='bike_hourly':raise ValueError('expected bike ZIP')
   return blob,link,sha(blob)
  except Exception as e:errors.append(link+': '+repr(e))
 # Allow real user-supplied file when network is blocked. No generated substitute.
 target={'abalone':'abalone.data','airfoil':'airfoil_self_noise.dat','bike_hourly':'hour.csv'}[family]
 path=ROOT/target
 if path.exists():return path.read_bytes(),str(path),sha(path.read_bytes())
 raise RuntimeError('; '.join(errors))
def fresh_streams():
 failures=[];streams=[];status=[]
 for family in EXTERNAL_SOURCES:
  try:
   blob,url,digest=download_family(family)
   if family=='abalone':
    parsed=list(csv.reader(io.StringIO(blob.decode('utf-8-sig'))))
    names=['length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight','rings']
    data=np.array([[float(x) for x in row[1:9]] for row in parsed if len(row)>=9],dtype='<f8')
   elif family=='airfoil':
    names=['frequency','angle','chord','velocity','displacement','sound_pressure']
    data=np.loadtxt(io.BytesIO(blob),dtype='<f8').reshape(-1,6)
   else:
    parsed=list(csv.DictReader(io.StringIO(blob.decode('utf-8-sig'))))
    names=[x for x in parsed[0] if x not in ('dteday',)]
    data=np.array([[float(row[k]) for k in names] for row in parsed],dtype='<f8')
   if len(data)<100:raise ValueError('too few rows')
   # Fixed deterministic prefix cap, stated in protocol; preserve source row order.
   data=data[:4096]
   for j,key in enumerate(names):
    b=raw(data[:,j]);streams.append({'dataset':family,'column':key,'stream':family+'__'+key,'source':url,'dataset_source_sha256':digest,'n':len(b)//8,'bytes':b})
   status.append({'family':family,'loaded':True,'rows_used':len(data),'columns':len(names),'url':url,'source_sha256':digest})
  except Exception as e:
   failures.append({'family':family,'error':repr(e)});status.append({'family':family,'loaded':False,'error':repr(e)})
 return streams,status,failures


env();results=[];errors=[];decision=[];family_status=[]
try:
 emit_csv('edge_cases.csv',all_edges())
 streams,family_status,failures=fresh_streams()
 emit_csv('download_failures.csv',failures)
 emit_csv('dataset_provenance.csv',family_status)
 for item in streams:
  b=item['bytes'];n=item['n'];cut=max(32,n//4)
  if cut>=n:continue
  cal=b[:8*cut];test=b[8*cut:]
  # Pre-specified policy, frozen without looking at held-out data:
  # Find best applicable non-lattice comparator and best lattice config on calibration;
  # choose lattice only if it improves >=5%. Otherwise retain calibration-best baseline.
  cal_rows=[];err_before=len(errors)
  for m in METHODS:
   try:
    t=measured(m,cal,1)
    cal_rows.append((m,t['bytes']))
   except ValueError:pass
   except Exception as e:errors.append({'stream':item['stream'],'stage':'calibration','mode':m,'error':repr(e)})
  generic=[x for x in cal_rows if x[0] in NONLATTICE]
  lattices=[x for x in cal_rows if x[0] in CANDIDATES]
  if not generic or not lattices:errors.append({'stream':item['stream'],'stage':'selection','error':'missing required methods'});continue
  baseline=min(generic,key=lambda x:x[1]);candidate=min(lattices,key=lambda x:x[1]);selected=candidate[0] if candidate[1]<=0.95*baseline[1] else baseline[0]
  # Held-out oracle for reporting only. It is NEVER used to choose deployed method.
  held=[]
  for m in METHODS:
   if not any(k==m for k,_ in cal_rows):continue
   try:
    y=measured(m,test,3);held.append({'stream':item['stream'],'dataset':item['dataset'],'column':item['column'],'mode':m,'n_heldout':n-cut,'sha256_heldout':sha(test),'source':item['source'],'source_dataset_sha256':item['dataset_source_sha256'],**y})
   except ValueError:pass
   except Exception as e:errors.append({'stream':item['stream'],'stage':'heldout','mode':m,'error':repr(e)})
  observed={r['mode']:r for r in held}
  if selected not in observed:errors.append({'stream':item['stream'],'stage':'heldout','error':'selected codec failed'});continue
  best_generic=min((x for x in held if x['mode'] in NONLATTICE),key=lambda x:x['bytes'])
  selected_size=observed[selected]['bytes'];baseline_size=best_generic['bytes']
  decision.append({'stream':item['stream'],'dataset':item['dataset'],'column':item['column'],'n_calibration':cut,'n_heldout':n-cut,'frozen_selection':selected,'frozen_cal_baseline':baseline[0],'cal_baseline_bytes':baseline[1],'cal_best_lattice':candidate[0],'cal_lattice_bytes':candidate[1],'selected_test_bytes':selected_size,'test_oracle_specialized_bytes':baseline_size,'gain_vs_test_oracle_percent':100*(baseline_size-selected_size)/baseline_size,'test_oracle_method':best_generic['mode'],'selected_is_lattice':selected.startswith('lattice_'),'exact':True,'source':item['source'],'sha256_heldout':sha(test)})
  results.extend(held)
 emit_csv('heldout_all_methods.csv',results);emit_csv('frozen_policy_decisions.csv',decision);emit_csv('errors.csv',errors)
 families=sorted({r['dataset'] for r in decision});all_families=len(families)==3
 pos=sum(x['gain_vs_test_oracle_percent']>0 for x in decision);neg=sum(x['gain_vs_test_oracle_percent']<0 for x in decision)
 sizes=(sum(x['selected_test_bytes'] for x in decision),sum(x['test_oracle_specialized_bytes'] for x in decision))
 gain=100*(sizes[1]-sizes[0])/sizes[1] if sizes[1] else None
 gate='INCOMPLETE_DATA' if not all_families else ('REJECT_CURRENT_DIRECTION' if gain is not None and gain<=0 else 'PROMISING_BUT_NOVELTY_AND_SPECIALIZED_BASELINES_MISSING')
 status={'status':'COMPLETE' if all_families else 'INCOMPLETE_DATA','gate':gate,'families':families,'family_status':family_status,'streams_evaluated':len(decision),'policy_lattice_selected':sum(x['selected_is_lattice'] for x in decision),'wins_vs_oracle':pos,'losses_vs_oracle':neg,'ties':len(decision)-pos-neg,'aggregate_gain_vs_oracle_percent':gain,'benchmark_scope':'complete SGX5 frames; same float64 test values; every baseline fitted only on calibration; hindsight best specialized reported as oracle, not a deployable selector','limitations':'Novelty and ALP/Chimp/Gorilla comparisons remain unproven; UCI datasets not necessarily time ordered; up to first 4096 rows; decisions use first 25% calibration; codec-specific selection computation is not included in timing.'}
except Exception as e:status={'status':'ERROR','error':repr(e),'traceback':traceback.format_exc()[-5000:]}
finally:
 emit_json('decision.json',status);print('NB54',json.dumps(status,indent=2,default=str)[:8000]);deliver()

