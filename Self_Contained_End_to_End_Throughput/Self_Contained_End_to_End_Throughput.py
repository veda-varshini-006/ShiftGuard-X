#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB56F — Repaired, self-contained throughput benchmark

The complete executable pipeline runs in **one cell**, preventing the `warm_tests` missing-definition error caused by running a later cell without its definitions. Uses actual dataset URLs; no synthetic substitution. Reports explicit errors if a data source fails.
"""

import os,sys,io,csv,json,struct,hashlib,time,zipfile,statistics,traceback,math,zlib,lzma,platform,urllib.request,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/'SGX_NB56F_outputs';OUT.mkdir(parents=True,exist_ok=True)
def emit_csv(name, rows):
 if not rows:return
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def emit_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=str))
def deliver():
 file=ROOT/'SGX_NB56F_results.zip'
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


# Optimised encoder and decoder: EXACT SAME QPK1 format as NB50, with NumPy vector operations.
def vec_bitpack(x,w):
 arr=np.asarray(x,dtype=np.uint64)
 if w==0 or len(arr)==0:return b''
 if np.any(arr >= (np.uint64(1)<<np.uint64(w))) if w<64 else False:raise ValueError('bit overflow')
 bits=((arr[:,None]>>np.arange(w,dtype=np.uint64))&1).astype(np.uint8).reshape(-1)
 return np.packbits(bits,bitorder='little').tobytes()
def vec_bitunpack(b,w,n):
 if w==0:return np.zeros(n,dtype=np.uint64)
 if len(b)!=(n*w+7)//8:raise ValueError('bit packed size mismatch')
 bits=np.unpackbits(np.frombuffer(b,dtype=np.uint8),bitorder='little')[:n*w].reshape(n,w)
 return (bits.astype(np.uint64)*(np.uint64(1)<<np.arange(w,dtype=np.uint64))).sum(axis=1,dtype=np.uint64)
def lattice_pack_fast(r,scale):
 if len(r)%8:raise ValueError('alignment')
 a=np.frombuffer(r,dtype='<f8');n=len(a)
 with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
  scaled=a*scale
  valid=np.isfinite(a)&np.isfinite(scaled)&(scaled >= -2147483648)&(scaled <= 2147483647)
  rounded=np.rint(np.where(valid,scaled,0)).astype(np.int64)
  restored=(rounded.astype('<f8')/scale).view('<u8')
  valid &= restored==a.view('<u8')
 values=rounded[valid];mn=int(values.min()) if len(values) else 0
 shifted=(values-mn).astype(np.uint64)
 width=int(shifted.max()).bit_length() if len(shifted) else 0
 marked=np.packbits(valid.astype(np.uint8),bitorder='little').tobytes()
 packed=vec_bitpack(shifted,width)
 exc=a.view('<u8')[~valid].astype('<u8').tobytes()
 groups=[zlib.compress(marked,6),zlib.compress(packed,6),zlib.compress(exc,6)]
 return MAG+struct.pack('<BIIqBIII',SCALES.index(scale),n,len(values),mn,width,*[len(g) for g in groups])+b''.join(groups)
def lattice_unpack_fast(blob):
 if blob[:4]!=MAG or len(blob)<34:raise ValueError('magic/header')
 idx,n,nacc,mn,width,a,b,c=struct.unpack('<BIIqBIII',blob[4:34])
 if idx>=len(SCALES) or n>100_000_000 or nacc>n or width>64 or len(blob)!=34+a+b+c:raise ValueError('header invalid')
 off=34;mark=zlib.decompress(blob[off:off+a]);off+=a
 packed=zlib.decompress(blob[off:off+b]);off+=b
 exceptions=zlib.decompress(blob[off:off+c])
 if len(mark)!=(n+7)//8 or len(exceptions)!=(n-nacc)*8:raise ValueError('section size')
 flags=np.unpackbits(np.frombuffer(mark,dtype=np.uint8),bitorder='little')[:n].astype(bool)
 if flags.sum()!=nacc:raise ValueError('accepted count')
 integers=vec_bitunpack(packed,width,nacc)
 decoded=(integers.astype(np.int64)+mn).astype('<f8')/SCALES[idx]
 out=np.empty(n,dtype='<u8');out[flags]=decoded.view('<u8')
 out[~flags]=np.frombuffer(exceptions,dtype='<u8')
 return out.astype('<u8').tobytes()
def time_trial_pack_unpack(r,scale,fast=False,reps=5):
 e=[];d=[];last=None;decompressed=None
 for i in range(reps+1):
  t=time.perf_counter_ns();blob=lattice_pack_fast(r,scale) if fast else lattice_pack(r,scale);encode_ms=(time.perf_counter_ns()-t)/1e6
  t=time.perf_counter_ns();v=lattice_unpack_fast(blob) if fast else lattice_unpack(blob);decode_ms=(time.perf_counter_ns()-t)/1e6
  if v!=r:raise AssertionError('bit-exact failure')
  if i>0:e.append(encode_ms);d.append(decode_ms)
  last=blob
 return {'bytes':len(last),'encode_ms_median':statistics.median(e),'decode_ms_median':statistics.median(d)}



# Use NB53's validated fast implementation for lattice modes, with SGX5 self-describing framing.
reference_encode_payload=encode_payload
reference_decode_payload=decode_payload
def encode_payload_fast(m,r):
 if m.startswith('lattice_'):return lattice_pack_fast(r,SCALES[int(m.split('_')[1])])
 return reference_encode_payload(m,r)
def decode_payload_fast(m,b,n):
 if m.startswith('lattice_'):return lattice_unpack_fast(b)
 return reference_decode_payload(m,b,n)
def encode_fast(m,r):
 if len(r)%8:raise ValueError('unaligned')
 p=encode_payload_fast(m,r)
 return FRAME+struct.pack('<BIQ',MIDS[m],len(r)//8,len(p))+p
def decode_fast(f):
 if f[:4]!=FRAME or len(f)<17:raise ValueError('frame')
 mid,n,npayload=struct.unpack('<BIQ',f[4:17]);p=f[17:]
 if mid not in INV or len(p)!=npayload:raise ValueError('length or mode')
 out=decode_payload_fast(INV[mid],p,n)
 if len(out)!=8*n:raise ValueError('decoded length')
 return out
def verified_size(m,r):
 blob=encode_fast(m,r)
 if decode_fast(blob)!=r:raise AssertionError('BITWISE_MISMATCH '+m)
 return len(blob)
def timed(m,r,reps=3,fast=True):
 fn_encode=encode_fast if fast else encode
 fn_decode=decode_fast if fast else decode
 E=[];T=[];size=0
 for k in range(reps+1):
  t=time.perf_counter_ns();blob=fn_encode(m,r);e=(time.perf_counter_ns()-t)/1e6
  t=time.perf_counter_ns();out=fn_decode(blob);d=(time.perf_counter_ns()-t)/1e6
  if out!=r:raise AssertionError('BIT_MISMATCH '+m)
  if k:
   E.append(e);T.append(d)
  size=len(blob)
 return {'bytes':size,'encode_ms':statistics.median(E),'decode_ms':statistics.median(T),'bits_per_value':8*size/(len(r)//8)}
def warm_tests():
 cases=[b'',raw(np.array([0.,-0.,1.0,-1.0,1.125])),np.array([0,0x8000000000000000,0x7ff8000000000042,0x7ff0000000000000,0xffffffffffffffff],dtype='<u8').tobytes(),raw(np.arange(128)/100)]
 rows=[]
 for i,r in enumerate(cases):
  for m in METHODS:
   try:
    b=encode_fast(m,r);ok=decode_fast(b)==r
    if m.startswith('lattice_'):
     ok=ok and lattice_unpack(b[17:])==r and encode(m,r)==b
    rows.append({'case':i,'mode':m,'status':'PASS' if ok else 'FAIL'})
   except ValueError as e:rows.append({'case':i,'mode':m,'status':'INAPPLICABLE','reason':str(e)})
 if any(x['status']=='FAIL' for x in rows):raise AssertionError('edge mismatch')
 return rows


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


def get_wine():
 urls={'red':'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv','white':'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'}
 for family,url in urls.items():
  # Fresh real downloads, no synthetic fallback. Optional manual CSVs in /content for offline use.
  local=ROOT/('winequality-'+family+'.csv')
  if local.exists(): payload=local.read_bytes();actual_source=str(local)
  else:
   with urllib.request.urlopen(url,timeout=60) as resp:payload=resp.read()
   actual_source=url
  data=list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig')),delimiter=';'))
  if not data:raise ValueError('Wine CSV empty '+family)
  size=(len(data)//1024)*1024
  if size<1024:raise ValueError('Wine stream too short '+family)
  for col in data[0].keys():
   a=np.array([float(row[col]) for row in data[:size]],dtype='<f8')
   yield {'dataset':'wine_'+family,'stream':'wine_'+family+'__'+col.replace(' ','_'),'column':col,'n':len(a),'bytes':raw(a),'source':actual_source,'source_sha256':sha(payload)}



POLICY_VERSION='SGX55_FIXED_2026_10_22'
MARGIN_ORIGINAL=0.05
MARGIN_ROBUST=0.15
FOLD_MARGIN=0.05
MIN_CAL_N=64
# Cal tests use a stable fixed order; any applicable complete SGX5 file is valid.
def choose(bcal):
 n=len(bcal)//8
 t=time.perf_counter_ns()
 trials={};errors={}
 for m in METHODS:
  try:trials[m]=verified_size(m,bcal)
  except ValueError as e:errors[m]=str(e)
  except Exception as e:errors[m]='UNEXPECTED '+repr(e)
 base=min(((m,size) for m,size in trials.items() if m in NONLATTICE),key=lambda x:(x[1],x[0]))
 latt=min(((m,size) for m,size in trials.items() if m in CANDIDATES),key=lambda x:(x[1],x[0]))
 old=latt[0] if latt[1]<=(1-MARGIN_ORIGINAL)*base[1] else base[0]
 robust=base[0]
 fold_sizes=[];fold_status=[]
 if n>=MIN_CAL_N and latt[1]<=(1-MARGIN_ROBUST)*base[1]:
  boundaries=np.linspace(0,n,4,dtype=int)
  safe=True
  for f in range(3):
   subsection=bcal[int(boundaries[f])*8:int(boundaries[f+1])*8]
   a=verified_size(latt[0],subsection)
   # Compare against each calibration-fold's strongest nonlattice alternative.
   b=min(verified_size(m,subsection) for m in NONLATTICE if m in trials)
   works=a<=(1-FOLD_MARGIN)*b
   fold_sizes.append({'fold':f,'lattice_bytes':a,'best_nonlattice_bytes':b})
   fold_status.append(works)
   if not works:safe=False
  if safe:robust=latt[0]
 return {'baseline':base[0],'baseline_cal_bytes':base[1],'lattice':latt[0],'lattice_cal_bytes':latt[1],'original_policy':old,'robust_policy':robust,'fold_checks':fold_status,'fold_size_detail':fold_sizes,'calibration_selection_ms':(time.perf_counter_ns()-t)/1e6,'num_inapplicable':len(errors),'failed_methods':errors}



env();data=[];stream_decisions=[];exceptions=[];provenance=[]
try:
 emit_csv('edge_tests.csv',warm_tests())
 # First three families reused for apples-to-apples with NB55; include Wine Red/White as second familiar family.
 streams,origins,problems=fresh_streams()
 for d in origins:provenance.append(d)
 for item in get_wine():
  streams.append(item|{'dataset_source_sha256':item['source_sha256']})
 for item in streams:
  b=item['bytes'];n=len(b)//8;cut=n//4;cal=b[:cut*8];test=b[cut*8:]
  if cut<MIN_CAL_N:continue
  try:
   ch=choose(cal)
   base=ch['baseline'];orig=ch['original_policy'];robust=ch['robust_policy']
   # Timing self-describing SGX5 frame, real encoder+decoder time. No cached payload or preencoded data.
   for role,m in [('best_calibration_nonlattice',base),('original_policy',orig),('robust_policy',robust)]:
    timing=timed(m,test,4,True)
    data.append({'stream':item['stream'],'family':item['dataset'],'role':role,'mode':m,'implementation':'vectorized_lattice_when_applicable','n_test':n-cut,'sha256_test':sha(test),'calibration_selection_ms':ch['calibration_selection_ms'],**timing})
   if robust.startswith('lattice_'):
    ref=timed(robust,test,3,False);fast=timed(robust,test,4,True)
    assert ref['bytes']==fast['bytes']
    data.append({'stream':item['stream'],'family':item['dataset'],'role':'lattice_reference_ablation','mode':robust,'implementation':'NB50_reference','n_test':n-cut,'sha256_test':sha(test),'calibration_selection_ms':0,**ref})
   stream_decisions.append({'stream':item['stream'],'family':item['dataset'],'frozen_baseline':base,'frozen_robust':robust,'chosen_is_lattice':robust.startswith('lattice_'),'calibration_selection_ms':ch['calibration_selection_ms'],'n_test':n-cut})
  except Exception as e:exceptions.append({'stream':item['stream'],'error':repr(e),'trace':traceback.format_exc()[-1000:]})
 emit_csv('end_to_end_timing.csv',data);emit_csv('selection_cost.csv',stream_decisions);emit_csv('errors.csv',exceptions);emit_csv('dataset_provenance.csv',provenance)
 status={'state':'COMPLETE' if len(stream_decisions)==54 and len(origins)==3 and all(x['loaded'] for x in origins) else 'INCOMPLETE','streams':len(stream_decisions),'families':sorted({x['family'] for x in stream_decisions}),'rows_timed':len(data),'chosen_lattice':sum(x['chosen_is_lattice'] for x in stream_decisions),'timing_scope':'Full SGX5 frame encode/decode, warmup + 4 measured repetitions; includes codec payload and header; calibration selection measured separately; excludes dataset download and file I/O; all CPU Python benchmarks. Native ALP/Chimp/Gorilla NOT benchmarked.','speed_claim':'No native-code speed superiority implied.'}
 # Weighted throughput across streams (bytes of RAW input / total median elapsed time). Only performance of chosen policy and calibration baseline.
 for role in ('best_calibration_nonlattice','robust_policy'):
  v=[x for x in data if x['role']==role]
  if v:
   raw_bytes=sum(x['n_test']*8 for x in v)
   status[role+'_total_encode_ms']=sum(x['encode_ms'] for x in v)
   status[role+'_total_decode_ms']=sum(x['decode_ms'] for x in v)
   status[role+'_aggregate_encoded_bytes']=sum(x['bytes'] for x in v)
   status[role+'_encode_mibps']=raw_bytes/(2**20)/(sum(x['encode_ms'] for x in v)/1000)
   status[role+'_decode_mibps']=raw_bytes/(2**20)/(sum(x['decode_ms'] for x in v)/1000)
except Exception as e:status={'state':'ERROR','error':repr(e),'traceback':traceback.format_exc()[-6000:]}
finally:
 emit_json('decision.json',status);print('NB56 status:',json.dumps(status,indent=2,default=str));deliver()

