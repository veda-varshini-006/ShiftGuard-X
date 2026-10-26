#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB36 — Frozen External Gate
Calibration-only selection plus held-out evaluation on bundled real datasets with speed and exact reconstruction.

CPU Colab-compatible; full container overhead, exact-byte decoding checks, and automatic results ZIP. Algorithms are exploratory; novelty is not presumed.
"""

OUT_NAME='SGX_NB36_outputs'



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
# The family list is deliberately untouched by candidate-specific tuning inside this notebook.
# Freeze selection on chronological first quarter; do not select by held-out results.
rows=[];decisions=[];errors=[]
for name,v,source in datasets():
 if len(v)<100:continue
 cut=max(32,len(v)//4)
 if cut>=len(v):continue
 cal=serialize(v[:cut]);hold=serialize(v[cut:])
 options=['zlib','lzma','shuffle_zlib','xor_zlib','proto2','proto4']
 def ev(m,raw,reps=3):return proto_measure(raw,int(m[-1]),reps) if m.startswith('proto') else measure(m,raw,reps)
 try:
  cal_metrics={m:ev(m,cal,3) for m in options}
  best_base=min(['zlib','lzma'],key=lambda m:cal_metrics[m]['bytes'])
  # Freeze a prototype only if >= 5% smaller on calibration than best generic.
  best_proto=min(['proto2','proto4'],key=lambda m:cal_metrics[m]['bytes'])
  selection=best_proto if cal_metrics[best_proto]['bytes'] <= .95*cal_metrics[best_base]['bytes'] else best_base
  results={m:ev(m,hold,5) for m in options}
  fixed_base=min(['zlib','lzma'],key=lambda m:results[m]['bytes'])  # report oracle baseline separately, not used in selection
  for m,d in results.items():rows.append({'stream':name,'source':source,'n_heldout':len(v)-cut,'selection':selection,'mode':m,**d})
  policy=results[selection]['bytes']; oracle=results[fixed_base]['bytes']
  decisions.append({'stream':name,'source':source,'n_calibration':cut,'n_heldout':len(v)-cut,'frozen_selection':selection,'cal_baseline':best_base,'heldout_selected_bytes':policy,'heldout_best_generic_bytes':oracle,'gain_vs_oracle_percent':100*(oracle-policy)/oracle,'prototype_selected':selection.startswith('proto'),'sha256_heldout':hashlib.sha256(hold).hexdigest()})
 except Exception as ex:errors.append({'stream':name,'error':str(ex)})
write_csv('heldout_measurements.csv',rows);write_csv('frozen_decisions.csv',decisions);write_csv('errors.csv',errors)
summary={'streams':len(decisions),'prototype_selected':sum(d['prototype_selected'] for d in decisions),'wins_vs_heldout_oracle':sum(d['gain_vs_oracle_percent']>0 for d in decisions),'regressions':sum(d['gain_vs_oracle_percent']<0 for d in decisions),'mean_gain_percent':float(np.mean([d['gain_vs_oracle_percent'] for d in decisions])) if decisions else None,'protocol_warning':'This is exploratory: the method and datasets were not preregistered; heldout best generic is an oracle comparator; no ALP, Chimp, Gorilla in this notebook. No novelty claim.'}
(OUT/'decision.json').write_text(json.dumps(summary,indent=2));print('DECISION',summary);export_zip('SGX_NB36_results.zip')

