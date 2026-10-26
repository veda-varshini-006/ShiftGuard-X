#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X | NB79 — Novelty_Gap_and_Preregistered_Hypothesis

A prior-art matrix, preregistered exponent-conditioned routing hypothesis, and conditional entropy diagnostics with a permutation control. Diagnostic only; no originality asserted.

**Research integrity:** Distinguish bit-exact correctness from novelty and external validity. Every reported size includes a 13-byte SG79 method/length frame. ALP/Chimp/Elf native-codec execution is not claimed. This notebook is standalone and exports a results ZIP.
"""

import os,sys,json,csv,io,math,struct,time,zipfile,zlib,lzma,hashlib,platform,statistics,urllib.request,traceback,random,subprocess
from pathlib import Path
import numpy as np
try:
 import zstandard as zstd
except ImportError:
 zstd=None
if zstd is None and Path('/content').exists():
 print('Installing zstandard for a stronger generic baseline...')
 try:
  subprocess.run([sys.executable,'-m','pip','install','-q','zstandard>=0.22'],check=True,timeout=180)
  import zstandard as zstd
 except Exception as exc: print('WARNING: zstandard install failed; benchmark marked INCOMPLETE:',repr(exc))
SCALES=[1,10,100,1000,10000]
MAG=b'QPK1'
MECHANISM_VERSION='sgx-exp-routing-v0.1-locked-2026-10-24'

def raw(v):return np.ascontiguousarray(np.asarray(v,dtype='<f8')).tobytes()
def digest(v):return hashlib.sha256(v).hexdigest()
def ensure_out(name):
 root=Path('/content' if Path('/content').exists() else '/mnt/data')
 out=root/(name+'_outputs');out.mkdir(exist_ok=True)
 return root,out
def savecsv(out,name,rows):
 if not rows:return
 with (out/name).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()));writer.writeheader();writer.writerows(rows)
def savejson(out,name,obj):
 (out/name).write_text(json.dumps(obj,indent=2,default=str))
def finish(root,out,name):
 zname=root/(name+'_results.zip')
 with zipfile.ZipFile(zname,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(out.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(out))
 print('RESULT ZIP:',zname,'bytes:',zname.stat().st_size)
 try:
  from google.colab import files; files.download(str(zname))
 except ImportError:pass
 return str(zname)

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

def shuffled(r):
 if len(r)%8:raise ValueError('raw alignment')
 return np.frombuffer(r,dtype=np.uint8).reshape(-1,8).T.copy().tobytes()
def unshuffled(r):
 if len(r)%8:raise ValueError('shuffle alignment')
 return np.frombuffer(r,dtype=np.uint8).reshape(8,-1).T.copy().tobytes()
def xor_forward(r):
 a=np.frombuffer(r,dtype='<u8');out=a.copy()
 if len(out)>1:out[1:]=a[1:]^a[:-1]
 return out.astype('<u8').tobytes()
def xor_backward(r):
 a=np.frombuffer(r,dtype='<u8').copy();k=1
 while k<len(a):a[k:]^=a[:-k].copy();k<<=1
 return a.astype('<u8').tobytes()

def route_labels(exponent,bins,negative_control=False):
 if bins not in (2,4,8):raise ValueError('group count')
 n=len(exponent)
 # use lower exponent bits: mathematically decoder-known from head field
 ids=(exponent.astype(np.uint32)&np.uint32(bins-1)).astype(np.uint8)
 if negative_control:
  # deterministic position-specific mask; decoder derives it without side information
  idx=np.arange(n,dtype=np.uint32)
  noise=((idx*np.uint32(2654435761)) ^ (idx>>np.uint32(9)) ^ np.uint32(0x9E3779B9))
  noise^=noise>>np.uint32(16);noise*=np.uint32(0x85EBCA6B);noise^=noise>>np.uint32(13)
  ids^=(noise&np.uint32(bins-1)).astype(np.uint8)
 return ids

def _mantissa_lanes(a):
 low=(a&np.uint64((1<<52)-1)).astype('<u8')
 return np.frombuffer(low.tobytes(),dtype=np.uint8).reshape(-1,8)[:,:7].copy()

def _pack_mant_lanes(lanes):
 # lane shuffle within bucket; decoder knows bucket sizes from the previously decoded exponent header
 return lanes.T.copy().tobytes()
def _unpack_mant_lanes(payload,n):
 if len(payload)!=7*n:raise ValueError('lane length')
 return np.frombuffer(payload,dtype=np.uint8).reshape(7,n).T.copy()

def routed_payload_encode(r,bins=1,negative_control=False):
 a=np.frombuffer(r,dtype='<u8');n=len(a)
 head=(a>>np.uint64(52)).astype('<u2') # sign + exponent, exactly 12 significant bits
 expo=(head&np.uint16(2047)).astype('<u2')
 labels=np.zeros(n,dtype=np.uint8) if bins==1 else route_labels(expo,bins,negative_control)
 lanes=_mantissa_lanes(a)
 head_zip=zlib.compress(head.tobytes(),6)
 parts=[zlib.compress(_pack_mant_lanes(lanes[labels==i]),6) for i in range(bins)]
 flags=1 if negative_control else 0
 prefix=struct.pack('<4sIBBI',b'ERM1',n,bins,flags,len(head_zip))
 return prefix+struct.pack('<'+'I'*bins,*[len(p) for p in parts])+head_zip+b''.join(parts)

def routed_payload_decode(payload):
 if len(payload)<14:raise ValueError('bad payload length')
 magic,n,bins,flags,headlen=struct.unpack('<4sIBBI',payload[:14]);pos=14
 if magic!=b'ERM1' or bins not in (1,2,4,8) or flags not in (0,1) or (bins==1 and flags):raise ValueError('header')
 if n>50_000_000:raise ValueError('unreasonable n')
 if len(payload)<pos+4*bins:raise ValueError('truncated segment lengths')
 lengths=struct.unpack('<'+'I'*bins,payload[pos:pos+4*bins]);pos+=4*bins
 if pos+headlen>len(payload):raise ValueError('truncated header stream')
 head_raw=zlib.decompress(payload[pos:pos+headlen]);pos+=headlen
 head=np.frombuffer(head_raw,dtype='<u2')
 if len(head)!=n or np.any(head>4095):raise ValueError('corrupt head')
 expo=head&np.uint16(2047)
 labels=np.zeros(n,dtype=np.uint8) if bins==1 else route_labels(expo,bins,flags==1)
 lanes=np.empty((n,7),dtype=np.uint8)
 for i,length in enumerate(lengths):
  if pos+length>len(payload):raise ValueError('truncated lane stream')
  dec=zlib.decompress(payload[pos:pos+length]);pos+=length
  mask=(labels==i);lanes[mask]=_unpack_mant_lanes(dec,int(mask.sum()))
 if pos!=len(payload):raise ValueError('trailing bytes')
 block=np.zeros((n,8),dtype=np.uint8);block[:,:7]=lanes
 # top nibble was provably zero: mantissa is 52 bits, so 8th byte remains zero
 mant=np.frombuffer(block.copy().tobytes(),dtype='<u8')
 reconstructed=(head.astype('<u8')<<np.uint64(52))|mant
 return reconstructed.astype('<u8').tobytes()

METHODS=['raw','zlib6','lzma1','shuffle_zlib6','xor_zlib6','float32_exact_zlib','head_mant_split','route2','route4','route8','scrambled2','scrambled4','lattice10','lattice100','lattice1000','lattice10000','dict_zlib','int_for_zlib']
if zstd is not None:METHODS.insert(2,'zstd3')
M2ID={m:i for i,m in enumerate(METHODS)};ID2M={i:m for m,i in M2ID.items()}
BASELINES=[m for m in METHODS if not (m.startswith('route') or m.startswith('scrambled'))]
CANDIDATES=['route2','route4','route8'];CONTROLS=['head_mant_split','scrambled2','scrambled4']

def enc_payload(method,r):
 if method=='raw':return r
 if method=='float32_exact_zlib':
  a=np.frombuffer(r,dtype='<f8');b=a.astype('<f4')
  if b.astype('<f8').tobytes()!=r:raise ValueError('not bit-exact float32')
  return zlib.compress(b.tobytes(),6)
 if method=='zlib6':return zlib.compress(r,6)
 if method=='lzma1':return lzma.compress(r,preset=1)
 if method=='zstd3':return zstd.ZstdCompressor(level=3).compress(r)
 if method=='shuffle_zlib6':return zlib.compress(shuffled(r),6)
 if method=='xor_zlib6':return zlib.compress(xor_forward(r),6)
 if method=='head_mant_split':return routed_payload_encode(r,1,False)
 if method.startswith('route'):return routed_payload_encode(r,int(method[-1]),False)
 if method.startswith('scrambled'):return routed_payload_encode(r,int(method[-1]),True)
 if method.startswith('lattice'):return lattice_pack(r,int(method[len('lattice'):]))
 if method=='dict_zlib':return _dict_pack(r,True)
 if method=='int_for_zlib':return _int_for_pack(r,True)
 raise KeyError(method)
def dec_payload(method,payload,n):
 if method=='raw':return payload
 if method=='float32_exact_zlib':return np.frombuffer(zlib.decompress(payload),dtype='<f4').astype('<f8').tobytes()
 if method=='zlib6':return zlib.decompress(payload)
 if method=='lzma1':return lzma.decompress(payload)
 if method=='zstd3':return zstd.ZstdDecompressor().decompress(payload)
 if method=='shuffle_zlib6':return unshuffled(zlib.decompress(payload))
 if method=='xor_zlib6':return xor_backward(zlib.decompress(payload))
 if method=='head_mant_split' or method.startswith('route') or method.startswith('scrambled'):return routed_payload_decode(payload)
 if method.startswith('lattice'):return lattice_unpack(payload)
 if method=='dict_zlib':return _dict_unpack(payload,n,True)
 if method=='int_for_zlib':return _int_for_unpack(payload,n,True)
 raise KeyError(method)

def encode(method,r):
 if len(r)%8:raise ValueError('unaligned')
 payload=enc_payload(method,r)
 return b'SG79'+struct.pack('<IBI',len(r)//8,M2ID[method],len(payload))+payload # 13-byte frame, all methods

def decode(frame):
 if len(frame)<13 or frame[:4]!=b'SG79':raise ValueError('bad frame')
 n,mid,clen=struct.unpack('<IBI',frame[4:13]);payload=frame[13:]
 if mid not in ID2M or len(payload)!=clen:raise ValueError('bad frame metadata')
 data=dec_payload(ID2M[mid],payload,n)
 if len(data)!=8*n:raise ValueError('bad raw length')
 return data

def timed(method,r,reps=2):
 x=[]
 for rep in range(reps):
  t=time.perf_counter_ns();frame=encode(method,r);et=(time.perf_counter_ns()-t)/1e6
  t=time.perf_counter_ns();recovered=decode(frame);dt=(time.perf_counter_ns()-t)/1e6
  if recovered!=r:raise AssertionError('not bit-identical: '+method)
  x.append((len(frame),et,dt))
 return {'bytes':x[0][0],'encode_ms':statistics.median(a[1] for a in x),'decode_ms':statistics.median(a[2] for a in x)}

def edge_tests(out):
 bits=np.array([0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0x7ff8000000000022,1,0xffffffffffffffff],dtype='<u8').tobytes()
 samples=[b'',bits,bits*11,np.array([0,0x8000000000000000]*25,dtype='<u8').tobytes()]
 rows=[]
 for m in METHODS:
  for sample in samples:
   try:
    frame=encode(m,sample);ok=decode(frame)==sample
    rows.append({'method':m,'n':len(sample)//8,'status':'PASS' if ok else 'FAIL','reason':''})
    if not ok:raise AssertionError('edge mismatch')
   except (ValueError,OverflowError) as e:
    if m in ['dict_zlib','int_for_zlib','float32_exact_zlib'] or m.startswith('lattice'):
     rows.append({'method':m,'n':len(sample)//8,'status':'NOT_APPLICABLE','reason':str(e)[:140]})
    else:raise
 savecsv(out,'edge_cases.csv',rows)
 return rows

def calibration_select(arr,margin=.05,cap=2048):
 cut=max(24,len(arr)//4)
 r=raw(arr[:cut]);metrics={};errors=[];base_trial_ms=0.0
 t0=time.perf_counter_ns()
 for m in METHODS:
  ttrial=time.perf_counter_ns()
  try:metrics[m]=timed(m,r,1)['bytes']
  except Exception as e:errors.append({'method':m,'error':repr(e)})
  finally:
   if m in BASELINES:base_trial_ms+=(time.perf_counter_ns()-ttrial)/1e6
 sel_ms=(time.perf_counter_ns()-t0)/1e6
 bestbase=min((m for m in BASELINES if m in metrics),key=lambda m:metrics[m])
 valid=[m for m in CANDIDATES if m in metrics]
 bestcandidate=min(valid,key=lambda m:metrics[m]) if valid else None
 choice=bestcandidate if bestcandidate and metrics[bestcandidate]<=metrics[bestbase]*(1-margin) else bestbase
 return {'selection':choice,'baseline':bestbase,'candidate':bestcandidate,'calibration_bytes':metrics,'selection_ms':sel_ms,'baseline_selection_ms':base_trial_ms,'cut':cut,'errors':errors}

def dataset_development():
 from sklearn.datasets import load_wine,load_breast_cancer,load_diabetes,load_digits
 for fam,ds,count in [('Wine_classification',load_wine(),10),('Breast_cancer',load_breast_cancer(),10),('Diabetes',load_diabetes(),10),('Digits',load_digits(),8)]:
  a=np.asarray(ds.data,dtype='<f8')[:2048,:count]
  for i in range(a.shape[1]):yield fam,f'column_{i:02d}',a[:,i].copy(),{'source':'scikit-learn built-in data; real, previously used in SGX'}


root,out=ensure_out('SGX_NB79')
from sklearn.datasets import load_wine,load_breast_cancer,load_diabetes,load_digits
literature=[
 {'work':'ALP','year':2024,'mechanism':'Exact decimal to integer with exceptions; front-bit compression and adaptive choice','closest_overlap':'high','source':'https://doi.org/10.1145/3626717'},
 {'work':'Apache Parquet ALP specification','year':2026,'mechanism':'Normative independently decodable ALP page/vector layout','closest_overlap':'high','source':'https://github.com/apache/parquet-format/blob/master/AlpEncoding.md'},
 {'work':'Apache Arrow Rust 60.0','year':2026,'mechanism':'Released ALP encoder/decoder support for Parquet','closest_overlap':'high','source':'https://arrow.apache.org/blog/2026/09/29/arrow-rs-60.0.0/'},
 {'work':'Chimp / Chimp128','year':2022,'mechanism':'XOR residuals and stored leading/trailing-zero structure','closest_overlap':'medium','source':'https://vldb.org/pvldb/vol15/p3058-liakos.pdf'},
 {'work':'Elf-star / SElf-star','year':2025,'mechanism':'Adaptive exponent/significand/XOR metadata coding','closest_overlap':'high','source':'https://doi.org/10.1109/JIOT.2025.3554999'},
 {'work':'Sprintz','year':2018,'mechanism':'Lossless multivariate prediction and bit packing','closest_overlap':'low-medium','source':'https://arxiv.org/abs/1803.01869'},
 {'work':'Byteshuffle / Bitshuffle','year':2014,'mechanism':'Typed-lane transpose followed by compression','closest_overlap':'high','source':'https://github.com/kiyo-masui/bitshuffle'}]
savecsv(out,'closest_prior_art.csv',literature)
hypothesis={'name':'Decoder-synchronous exponent-conditioned mantissa routing (DSEMR)','novelty_claim':'NONE: combination may have been previously published','target_mechanism':'Compress sign+exponent first, then partition exact mantissa bytes by bucket computed from the decoded exponent, with no separate route map. Preserve original order through deterministic inverse grouping.','distinguishing_claim_to_test':'Material benefit over unconditioned head/mantissa split and matched-position scrambled-label negative control, AFTER group-stream metadata; and benefit over strong specialized baselines and fixed decimal-lattice codec.','causal_criteria':['candidate is smaller than independent header+mantissa split','candidate improves over scrambled-label control on a majority of eligible streams','calibration-frozen candidate improves size >= 5% against strongest deployed baseline on multiple unseen dataset families','all special-value bit patterns reconstructed exactly','speed including calibration is competitive enough for explicitly defined usage'],'kill_criteria':['no candidate selections on development families','no gains over independent split / negative control','new external tests fail to show gains','methods indistinguishable from published prior art after full-text review'],'pre_registered_parameters':{'bins':[2,4,8],'compression':'zlib level 6 for all routed groups','calibration_fraction':0.25,'margin_to_activate':0.05,'external_families':['UCI Occupancy Detection','UCI Appliances Energy Prediction','UCI Concrete Slump Test'],'source_data_limit_rows':4096},'publication_gate':'Reject originality assertion without exhaustive full-text prior-art search; reject Q1 readiness without ALP/Chimp/Elf comparable native/portable baselines and confidence intervals.'}
savejson(out,'preregistered_hypothesis.json',hypothesis)
# Count Jensen-Shannon-like/entropy proxy for low mantissa byte conditional on decoded exponent group
rows=[]
def entropy(counts):
 p=counts[counts>0].astype(float)/counts.sum()
 return float(-(p*np.log2(p)).sum()) if len(p) else 0.0
for family,name,values,provenance in dataset_development():
 if len(values)<25:continue
 bits=np.frombuffer(raw(values),dtype='<u8');byte=(bits&np.uint64(255)).astype(np.uint8)
 exponent=((bits>>np.uint64(52))&np.uint64(2047)).astype(np.uint16)
 bins=4;labs=route_labels(exponent,bins)
 marginal=entropy(np.bincount(byte,minlength=256))
 cond=sum((labs==j).mean()*entropy(np.bincount(byte[labs==j],minlength=256)) for j in range(bins) if np.any(labs==j))
 rng=np.random.default_rng(20261024);neg=rng.permutation(labs)
 negent=sum((neg==j).mean()*entropy(np.bincount(byte[neg==j],minlength=256)) for j in range(bins) if np.any(neg==j))
 rows.append({'family':family,'stream':name,'n':len(values),'H_low_mantissa_byte':marginal,'H_cond_exponent_bucket':cond,'H_cond_permuted_bucket':negent,'estimated_conditional_gain_bits_per_value':marginal-cond,'negative_control_gain_bits_per_value':marginal-negent})
savecsv(out,'conditional_entropy_diagnostics.csv',rows)
state={'status':'LITERATURE_AND_HYPOTHESIS_REGISTERED','streams':len(rows),'mean_conditional_gain_bits_per_value':float(np.mean([r['estimated_conditional_gain_bits_per_value'] for r in rows])),'notice':'Finite-sample entropy estimates are biased; positive entropy difference by itself is NOT a compression result or originality evidence. NB80 will measure actual framed bytes.'}
savejson(out,'decision.json',state);print(state)
finish(root,out,'SGX_NB79')

