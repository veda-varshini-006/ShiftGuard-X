#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB58: Safe_Fallback_Complete_Timing

Fixes the int32 inapplicable-codec failure without modifying existing codec format. A failed selected codec is replaced by a fixed always-applicable zlib fallback. Measures selection, attempted compression, complete file sizes and bit-exact decoding. Not a new coding algorithm.

Run **Runtime → Run all** in a fresh Google Colab CPU session. The final cell writes a results ZIP and attempts a browser download. This work is exploratory and does not establish novelty, journal acceptance, or ALP superiority.
"""

import os,sys,io,csv,json,struct,hashlib,time,zipfile,statistics,traceback,math,zlib,lzma,platform,urllib.request,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/'SGX_NB58_outputs';OUT.mkdir(parents=True,exist_ok=True)
def emit_csv(name, rows):
 if not rows:return
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def emit_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=str))
def deliver():
 file=ROOT/'SGX_NB58_results.zip'
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




# NB58 protocol: fixed same data families, 25% prefix calibration, no hindsight selection.
# Chosen specialized methods may become INAPPLICABLE on holdout: this is a domain change,
# not a corrupted stream. Charge attempted encoding time and switch to always-applicable zlib.
import random
random.seed(20261022)
REPS=5
FALLBACK='zlib6'
def encode_with_safe_fallback(requested,b):
    start=time.perf_counter_ns(); why=''; actual=requested; attempted=False
    try:
        blob=encode_fast(requested,b)
    except ValueError as exc:
        attempted=True;why=str(exc);actual=FALLBACK
        blob=encode_fast(FALLBACK,b)
    encoding_ms=(time.perf_counter_ns()-start)/1e6
    t=time.perf_counter_ns();result=decode_fast(blob);decoding_ms=(time.perf_counter_ns()-t)/1e6
    if result!=b:raise AssertionError('BITWISE ROUNDTRIP FAILURE')
    return {'actual_mode':actual,'requested_mode':requested,'inapplicable_selected':attempted,'reason':why,
            'blob':blob,'encode_ms':encoding_ms,'decode_ms':decoding_ms}
def time_real_policy(requested,b,reps=REPS):
    trials=[]
    for i in range(reps+1):
        r=encode_with_safe_fallback(requested,b)
        if i>0:trials.append(r)
    assert len({(t['actual_mode'],len(t['blob'])) for t in trials})==1
    t=trials[-1]
    return {'selected_mode':requested,'actual_mode':t['actual_mode'],
            'fallback_used':t['inapplicable_selected'],'fallback_reason':t['reason'],
            'encoded_bytes':len(t['blob']),
            'encode_ms':statistics.median(x['encode_ms'] for x in trials),
            'decode_ms':statistics.median(x['decode_ms'] for x in trials)}

# Targeted regression: calibration makes int32 seem valid but held-out data contains -0.0.
cal_test=raw(np.arange(128,dtype=np.float64))
holdout_test=raw(np.array([0.0,-0.0,2.0,3.0]))
a=time_real_policy('int_for',holdout_test,2)
assert a['fallback_used'] and a['actual_mode']==FALLBACK
b=time_real_policy('int_for_zlib',holdout_test,2)
assert b['fallback_used'] and b['actual_mode']==FALLBACK
edge_rows=warm_tests()+[{'case':'heldout_negzero_int_for','mode':'int_for','status':'PASS_FALLBACK'}, {'case':'heldout_negzero_int_for_zlib','mode':'int_for_zlib','status':'PASS_FALLBACK'}]
assert not any(r['status']=='FAIL' for r in edge_rows)

env();emit_csv('edge_tests.csv',edge_rows)
rows=[];selects=[];errors=[];provenance=[];families=[]
SMOKE=os.environ.get('SGX_LOCAL_SMOKE')=='1'
if SMOKE:
    data=np.arange(320,dtype=np.float64)
    data[-7:]=np.array([-0.,1.,2.,3.,4.,5.,6.])
    streams=[{'dataset':'synthetic_test_only','stream':'fixture_float64_320','n':len(data),'bytes':raw(data),'dataset_source_sha256':'SYNTHETIC_TEST_ONLY'}]
else:
    streams,families,issues=fresh_streams();errors.extend(issues)
    provenance.extend(families)
    try:
        for item in get_wine():streams.append(item)
    except Exception as exc:errors.append({'family':'wine','error':repr(exc)})

for item in streams:
    stream=item['stream'];whole=item['bytes'];n=len(whole)//8
    try:
        split=n//4
        if split<MIN_CAL_N:raise ValueError('Insufficient calibration values')
        cal=whole[:split*8];hold=whole[split*8:]
        st=time.perf_counter_ns();selection=choose(cal);selection_ms=(time.perf_counter_ns()-st)/1e6
        bmethod=selection['baseline'];rmethod=selection['robust_policy']
        base=time_real_policy(bmethod,hold)
        candidate=time_real_policy(rmethod,hold)
        generic=time_real_policy('zlib6',hold)
        for role,metric in [('calibration_nonlattice',base),('safe_sgx_policy',candidate),('fixed_zlib',generic)]:
            rows.append({'stream':stream,'family':item['dataset'],'role':role,'n_cal':split,
                         'n_test':n-split,'sha256_heldout':sha(hold),'selection_ms':selection_ms,
                         **metric,'bit_exact':True})
        selects.append({'stream':stream,'family':item['dataset'],'requested_baseline':bmethod,
                        'requested_sgx':rmethod,'actual_sgx':candidate['actual_mode'],
                        'baseline_fallback':base['fallback_used'],
                        'sgx_fallback':candidate['fallback_used'],
                        'selection_ms':selection_ms,
                        'bytes_baseline':base['encoded_bytes'],
                        'bytes_sgx':candidate['encoded_bytes'],
                        'bytes_zlib':generic['encoded_bytes'],
                        'improved_vs_cal_base':candidate['encoded_bytes']<base['encoded_bytes']})
    except Exception as exc:errors.append({'stream':stream,'error':repr(exc),'traceback':traceback.format_exc()[-2200:]})

emit_csv('end_to_end_measurements.csv',rows)
emit_csv('frozen_decisions.csv',selects)
emit_csv('errors.csv',errors)
emit_csv('dataset_provenance.csv',provenance)
roles={}
for role in ('calibration_nonlattice','safe_sgx_policy','fixed_zlib'):
    found=[r for r in rows if r['role']==role]
    if not found:continue
    in_bytes=sum(r['n_test']*8 for r in found)
    encoded=sum(r['encoded_bytes'] for r in found)
    e=sum(r['encode_ms'] for r in found)
    d=sum(r['decode_ms'] for r in found)
    sel=sum(r['selection_ms'] for r in found)
    # For a single-use stream, SGX/full baseline must pay selection once per stream.
    roles[role]={'streams':len(found),'raw_bytes':in_bytes,'encoded_bytes':encoded,
                 'encode_ms_sum':e,'decode_ms_sum':d,'selection_ms_sum':sel,
                 'encode_plus_cal_ms':e+sel if role!='fixed_zlib' else e,
                 'single_use_end_to_end_mibps':(in_bytes/1048576)/((e+sel if role!='fixed_zlib' else e)/1000),
                 'steady_encode_mibps':(in_bytes/1048576)/(e/1000),
                 'steady_decode_mibps':(in_bytes/1048576)/(d/1000)}
complete=(not SMOKE and len(selects)==54 and len(families)==3 and all(f.get('loaded') for f in families)
          and not errors and all(r['bit_exact'] for r in rows))
state='SMOKE_ONLY' if SMOKE else ('COMPLETE' if complete else 'INCOMPLETE')
summary={'state':state,'streams_completed':len(selects),'streams_expected':54,
         'selected_fallbacks_baseline':sum(x['baseline_fallback'] for x in selects),
         'selected_fallbacks_sgx':sum(x['sgx_fallback'] for x in selects),
         'methods':METHODS,'policy_version':POLICY_VERSION,'roles':roles,
         'errors':len(errors),
         'timing_notes':['Same full SGX5 container accounting for each codec',
                         'Warmup plus five measured repeat encodes and decodes; median per stream',
                         'Selection measured separately; single-use throughput includes selection',
                         'Fallback on inapplicable method is charged in measured encode time',
                         'Python/NumPy implementations only; no ALP or native-code speed claim',
                         'Datasets partly used during prior SGX experiments; not untouched research evaluation']}
if 'calibration_nonlattice' in roles and 'safe_sgx_policy' in roles:
    sb=roles['safe_sgx_policy']['encoded_bytes'];bb=roles['calibration_nonlattice']['encoded_bytes']
    summary['savings_vs_calibrated_baseline_percent']=100*(bb-sb)/bb
emit_json('decision.json',summary)
print(json.dumps(summary,indent=2)[:12000]);deliver()

