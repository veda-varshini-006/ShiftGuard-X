#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB35 — Context Routing Coding Prototype
Falsifiable explicit-routing context-grouped XOR coder, decoder, ablations and overhead accounting.

CPU Colab-compatible; full container overhead, exact-byte decoding checks, and automatic results ZIP. Algorithms are exploratory; novelty is not presumed.
"""

OUT_NAME='SGX_NB35_outputs'



import os,sys,time,json,csv,io,struct,zlib,lzma,hashlib,platform,zipfile,traceback,math
from pathlib import Path
import numpy as np
from sklearn.datasets import load_diabetes, load_wine, load_breast_cancer, load_digits
try:
 import zstandard as zstd
except ImportError:
 zstd=None
OUT=Path('/content' if Path('/content').exists() else '/mnt/data')/OUT_NAME
OUT.mkdir(parents=True,exist_ok=True)
def write_csv(name,rows):
 if not rows:return
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def export_zip(name):
 target=OUT.parent/name
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(OUT.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(OUT))
 print('RESULTS_ZIP:',target,'SIZE:',target.stat().st_size)
 try:
  from google.colab import files
  files.download(str(target))
 except ImportError:pass
 return target
def datasets():
 sources=[('diabetes',load_diabetes()),('wine_classification',load_wine()),('breast_cancer',load_breast_cancer()),('digits',load_digits())]
 for tag,b in sources:
  a=np.asarray(b.data,dtype=np.float64)
  for j in range(a.shape[1]):
   vals=np.ascontiguousarray(a[:,j]); yield f'{tag}__f{j:02d}',vals,tag
# Byte-exact raw bit reinterpretation, not numerical comparison
MAG=b'SGX1'
def serialize(vals):return np.ascontiguousarray(vals,dtype='<f8').tobytes()
def decode_vals(buf):return np.frombuffer(buf,dtype='<f8').copy()
def shuffle_bytes(buf):return np.frombuffer(buf,dtype=np.uint8).reshape(-1,8).T.copy().tobytes()
def unshuffle_bytes(buf):return np.frombuffer(buf,dtype=np.uint8).reshape(8,-1).T.copy().tobytes()
def xor_diff(buf):
 a=np.frombuffer(buf,dtype='<u8'); b=a.copy()
 if len(b)>1:b[1:]=a[1:]^a[:-1]
 return b.astype('<u8').tobytes()
def xor_restore(buf):
 a=np.frombuffer(buf,dtype='<u8').copy()
 # xor prefix scan inverse, vectorized in log N
 k=1
 while k<len(a):
  a[k:] ^= a[:-k].copy();k*=2
 return a.astype('<u8').tobytes()
def pack(mode,raw):
 if mode=='raw':return raw
 if mode=='zlib':return zlib.compress(raw,6)
 if mode=='lzma':return lzma.compress(raw,preset=1)
 if mode=='zstd':
  if zstd is None:raise RuntimeError('zstandard not installed')
  return zstd.ZstdCompressor(level=3).compress(raw)
 if mode=='shuffle_zlib':return zlib.compress(shuffle_bytes(raw),6)
 if mode=='xor_zlib':return zlib.compress(xor_diff(raw),6)
 if mode=='xor_shuffle_zlib':return zlib.compress(shuffle_bytes(xor_diff(raw)),6)
 if mode=='intpack_zlib':
  v=decode_vals(raw)
  if not np.all(np.isfinite(v)) or not np.all(np.equal(v,np.floor(v))):raise ValueError('not exactly integral')
  # only small exact signed integers permitted: avoid overflow and float rounding
  if len(v) and (np.min(v)<-(2**31) or np.max(v)>2**31-1):raise ValueError('integer outside int32')
  a=v.astype('<i4');return zlib.compress(a.tobytes(),6)
 if mode=='dict_zlib':
  v=np.frombuffer(raw,dtype='<u8')
  uniq,ids=np.unique(v,return_inverse=True)
  if len(uniq)>65535:raise ValueError('dictionary too large')
  token=ids.astype('<u2').tobytes();header=struct.pack('<I',len(uniq))+uniq.astype('<u8').tobytes()
  return zlib.compress(header+token,6)
 raise KeyError(mode)
def unpack(mode,buf,n):
 if mode=='raw':return buf
 if mode=='zlib':return zlib.decompress(buf)
 if mode=='lzma':return lzma.decompress(buf)
 if mode=='zstd':return zstd.ZstdDecompressor().decompress(buf)
 if mode=='shuffle_zlib':return unshuffle_bytes(zlib.decompress(buf))
 if mode=='xor_zlib':return xor_restore(zlib.decompress(buf))
 if mode=='xor_shuffle_zlib':return xor_restore(unshuffle_bytes(zlib.decompress(buf)))
 if mode=='intpack_zlib':return np.frombuffer(zlib.decompress(buf),dtype='<i4').astype('<f8').tobytes()
 if mode=='dict_zlib':
  payload=zlib.decompress(buf); k=struct.unpack('<I',payload[:4])[0]
  d=np.frombuffer(payload,dtype='<u8',count=k,offset=4)
  ids=np.frombuffer(payload,dtype='<u2',offset=4+8*k)
  if len(ids)!=n:raise ValueError('bad dictionary index count')
  return d[ids].astype('<u8').tobytes()
 raise KeyError(mode)
def framed(mode,raw):
 body=pack(mode,raw);mb=mode.encode();return MAG+struct.pack('<QB',len(raw)//8,len(mb))+mb+struct.pack('<Q',len(body))+body
def unframed(blob):
 if blob[:4]!=MAG:raise ValueError('magic')
 n,ml=struct.unpack('<QB',blob[4:13]);mode=blob[13:13+ml].decode();bl=struct.unpack('<Q',blob[13+ml:21+ml])[0];body=blob[21+ml:]
 if len(body)!=bl:raise ValueError('truncated frame')
 raw=unpack(mode,body,n)
 if len(raw)!=n*8:raise ValueError('incorrect length')
 return raw
MODES=['raw','zlib','lzma','shuffle_zlib','xor_zlib','xor_shuffle_zlib','intpack_zlib','dict_zlib']+(['zstd'] if zstd else [])
def measure(mode,raw,reps=3):
 out=[]
 for _ in range(reps):
  t=time.perf_counter_ns(); b=framed(mode,raw);enc=(time.perf_counter_ns()-t)/1e6
  t=time.perf_counter_ns(); v=unframed(b);dec=(time.perf_counter_ns()-t)/1e6
  if v!=raw:raise AssertionError('bitwise decoding mismatch')
  out.append((len(b),enc,dec))
 return {'bytes':out[0][0],'enc_ms':float(np.median([x[1] for x in out])),'dec_ms':float(np.median([x[2] for x in out]))}
def checks():
 patterns=np.array([0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0x0000000000000001,0xffffffffffffffff],dtype='<u8').tobytes()
 result=[]
 for mode in MODES:
  try:
   b=framed(mode,patterns);passed=unframed(b)==patterns;result.append({'mode':mode,'passed':passed,'reason':''})
  except ValueError as e:result.append({'mode':mode,'passed':'not_applicable','reason':str(e)})
  except Exception as e:result.append({'mode':mode,'passed':False,'reason':str(e)})
 write_csv('correctness.csv',result)
 assert not any(x['passed'] is False for x in result)
 print('Special-bit-pattern checks:',result)
def environment():
 (OUT/'environment.json').write_text(json.dumps({'python':sys.version,'numpy':np.__version__,'zstandard':getattr(zstd,'__version__',None),'platform':platform.platform(),'dataset_source':'sklearn bundled public datasets; no synthetic values substituted'},indent=2))



# Experimental encoder: group successive XOR residuals according to previous value's exponent.
# Explicit routing bits enable exact reversal; the routing metadata is fully included in size.
# This is NOT asserted to be new in the literature.
def context_encode(raw,bins=2):
 a=np.frombuffer(raw,dtype='<u8');n=len(a)
 if n==0:return struct.pack('<BII',bins,0,0)
 exps=((a>>np.uint64(52))&np.uint64(2047)).astype(np.uint16)
 # deterministic decoder-accessible context: previous ORIGINAL exponent, made available via routing reconstruction
 # For robust independent decoding, encode context labels explicitly; include full cost.
 residual=np.frombuffer(xor_diff(raw),dtype='<u8')
 if bins==2:labels=(exps<1024).astype(np.uint8)
 elif bins==4:labels=(exps//512).astype(np.uint8)
 else:raise ValueError('bins')
 groups=[]
 for k in range(bins):groups.append(zlib.compress(residual[labels==k].astype('<u8').tobytes(),6))
 route=zlib.compress(labels.tobytes(),6)
 hdr=struct.pack('<BII',bins,n,len(route))+route+struct.pack('<'+'I'*bins,*[len(g) for g in groups]);return hdr+b''.join(groups)
def context_decode(buf):
 bins,n,rlen=struct.unpack('<BII',buf[:9]);p=9
 labels=np.frombuffer(zlib.decompress(buf[p:p+rlen]),dtype=np.uint8);p+=rlen
 if len(labels)!=n or np.any(labels>=bins):raise ValueError('routing')
 lens=struct.unpack('<'+'I'*bins,buf[p:p+bins*4]);p+=bins*4
 dest=np.empty(n,dtype='<u8')
 for k,blen in enumerate(lens):
  group=np.frombuffer(zlib.decompress(buf[p:p+blen]),dtype='<u8');p+=blen
  if len(group)!=int(np.sum(labels==k)):raise ValueError('group length')
  dest[labels==k]=group
 if p!=len(buf):raise ValueError('trailing')
 return xor_restore(dest.tobytes())
def proto_frame(raw,bins=2):return b'SGX2'+struct.pack('<BI',bins,len(raw)//8)+context_encode(raw,bins)
def proto_unframe(buf):
 if buf[:4]!=b'SGX2':raise ValueError('magic')
 n=struct.unpack('<I',buf[5:9])[0];out=context_decode(buf[9:])
 if len(out)!=n*8:raise ValueError('size')
 return out
def proto_measure(raw,bins=2,reps=3):
 rows=[]
 for _ in range(reps):
  t=time.perf_counter_ns();c=proto_frame(raw,bins);e=(time.perf_counter_ns()-t)/1e6
  t=time.perf_counter_ns();d=proto_unframe(c);de=(time.perf_counter_ns()-t)/1e6
  if d!=raw:raise AssertionError('prototype mismatch')
  rows.append((len(c),e,de))
 return {'bytes':rows[0][0],'enc_ms':float(np.median([x[1] for x in rows])),'dec_ms':float(np.median([x[2] for x in rows]))}

checks();environment()
edge=np.array([0,0x8000000000000000,0x7ff8000000000001,0x7ff0000000000000,0xffffffffffffffff],dtype='<u8').tobytes()
for bins in (2,4):assert proto_unframe(proto_frame(edge,bins))==edge
rows=[];errors=[]
for name,v,source in datasets():
 raw=serialize(v)
 for mode in ['zlib','lzma','xor_zlib','shuffle_zlib','xor_shuffle_zlib','prototype_2','prototype_4']:
  try:
   x=proto_measure(raw,int(mode[-1])) if mode.startswith('prototype_') else measure(mode,raw)
   rows.append({'stream':name,'source':source,'n':len(v),'mode':mode,'bpv':8*x['bytes']/len(v),**x})
  except Exception as ex:errors.append({'stream':name,'mode':mode,'error':str(ex)})
write_csv('prototype_ablation.csv',rows);write_csv('errors.csv',errors)
(OUT/'novelty_limits.txt').write_text('Context partitioning, XOR prediction, entropy coding, and routing side information all have extensive prior art. This notebook is a falsifiable prototype ONLY, not an original-algorithm claim. Explicit routing costs included. ALP not run.\n')
print('ROWS',len(rows),'ERRORS',len(errors));export_zip('SGX_NB35_results.zip')

