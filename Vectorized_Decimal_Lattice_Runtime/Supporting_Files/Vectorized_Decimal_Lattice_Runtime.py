#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB53 — Vectorized_Decimal_Lattice_Runtime

Use the original NB50 QPK1 format, accelerate numerical acceptance and bitpacking with NumPy, verify cross-decoder interoperability and special IEEE values, and measure the true encode/decode time on frozen-calibration scale choices.

This is a standalone Google Colab CPU notebook. Execute **Runtime → Run all**, then download its `SGX_NB53_results.zip`. The notebook never substitutes synthetic data if a real dataset cannot be downloaded.
"""

import os,sys,io,csv,json,struct,hashlib,time,zipfile,statistics,traceback,math,zlib,lzma,platform,urllib.request,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/'SGX_NB53_outputs';OUT.mkdir(parents=True,exist_ok=True)
def emit_csv(name, rows):
 if not rows:return
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def emit_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=str))
def deliver():
 file=ROOT/'SGX_NB53_results.zip'
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


env();results=[];edge=[];errors=[]
try:
 testbits=np.array([0,0x8000000000000000,0x7ff8000000000001,0x7ff0000000000000,0xfff0000000000000,0x0000000000000001,0xffffffffffffffff],dtype='<u8').tobytes()
 testcases=[b'',testbits,testbits*50,raw(np.arange(2048)/100),raw(np.arange(-25,25))]
 for b in testcases:
  for scale in SCALES:
   old=lattice_pack(b,scale);new=lattice_pack_fast(b,scale)
   exact=(lattice_unpack_fast(new)==b and lattice_unpack(new)==b and lattice_unpack_fast(old)==b)
   edge.append({'n':len(b)//8,'scale':scale,'exact':exact,'identical_encoded_bytes':old==new,'old_bytes':len(old),'new_bytes':len(new)})
 if not all(x['exact'] for x in edge):raise AssertionError('fast encoder/decoder failed exact test')
 emit_csv('edge_tests.csv',edge)
 for item in get_wine():
  r=item['bytes'];n=item['n'];cut=max(32,n//4)
  cal=r[:8*cut];hold=r[8*cut:]
  choices=[(len(lattice_pack(cal,s)),s) for s in SCALES]
  selected=min(choices)[1]  # choose BEFORE touching held-out, based ONLY on calibration bytes
  ref=time_trial_pack_unpack(hold,selected,False,5)
  fast=time_trial_pack_unpack(hold,selected,True,5)
  # Additional independent bitwise frame compatibility assertions.
  if lattice_unpack_fast(lattice_pack(hold,selected))!=hold or lattice_unpack(lattice_pack_fast(hold,selected))!=hold:raise AssertionError('cross version failure')
  results.append({'stream':item['stream'],'dataset':item['dataset'],'n_calibration':cut,'n_heldout':n-cut,'scale_frozen':selected,'sha256_heldout':sha(hold),'reference_bytes':ref['bytes'],'fast_bytes':fast['bytes'],'size_identical':ref['bytes']==fast['bytes'],'reference_encode_ms':ref['encode_ms_median'],'fast_encode_ms':fast['encode_ms_median'],'encode_speedup_ref_over_fast':ref['encode_ms_median']/fast['encode_ms_median'],'reference_decode_ms':ref['decode_ms_median'],'fast_decode_ms':fast['decode_ms_median'],'decode_speedup_ref_over_fast':ref['decode_ms_median']/fast['decode_ms_median'],'exact':True})
 emit_csv('frozen_scale_runtime.csv',results)
 status={'status':'COMPLETE','streams':len(results),'all_exact':all(x['exact'] for x in results),'mean_encode_speedup_ref_over_fast':statistics.mean(x['encode_speedup_ref_over_fast'] for x in results),'mean_decode_speedup_ref_over_fast':statistics.mean(x['decode_speedup_ref_over_fast'] for x in results),'note':'5 repetitions after warmup; Python local timings not native implementation; same QPK1 format as NB50; speedups may be <1 and are never guaranteed.'}
except Exception as e:status={'status':'ERROR','error':repr(e),'traceback':traceback.format_exc()[-5000:]}
finally:
 emit_json('status.json',status);print('NB53',status);deliver()

