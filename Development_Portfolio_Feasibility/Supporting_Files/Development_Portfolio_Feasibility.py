#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX_NB83 — Development_Portfolio_Feasibility

Runs exact decoder-verified streaming frame compression under Hedge, follow-the-leader, probe/commit, fixed Zlib and hindsight oracle on previously used development families. Accounts for full-information trial cost.
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



# SGX selector feasibility protocol. Source codecs are the *existing* NB80 family.
# Adaptation rule is standard Hedge with full-information feedback; NOT a new theorem.
from sklearn.datasets import load_diabetes,load_wine,load_breast_cancer,load_digits
BLOCK_N=64
EXPERTS=['raw','zlib6','lzma1','shuffle_zlib6','dict_zlib','int_for_zlib','lattice100']
if zstd is not None:EXPERTS.append('zstd3')
ETA_POLICY='sqrt(8*log(K)/T)'; RNG_SEED=20261024
GATE_GAIN=0.02

def block_frames(arr,block_n=BLOCK_N):
 r=raw(arr); chunk=8*block_n
 blocks=[r[i:i+chunk] for i in range(0,(len(r)//chunk)*chunk,chunk)]
 return blocks, r[len(blocks)*chunk:],r

def frame_options(block):
 # Each method is a complete SG79 codec frame. Invalid methods revert to raw.
 # Size cap enforces losses in [0,1] exactly for all methods, without clipping.
 raw_frame=encode('raw',block); cap=len(raw_frame)+4
 opts={};spent_ms={}; errors={}
 for m in EXPERTS:
  t=time.perf_counter_ns()
  try:
   f=encode(m,block)
   if decode(f)!=block:raise AssertionError('codec round-trip failed')
   if len(f)>len(raw_frame):f=raw_frame
  except (ValueError,OverflowError) as e:
   f=raw_frame;errors[m]='INAPPLICABLE:'+repr(e)
  except Exception as e:
   # Unexpected failures MUST propagate. Never turn a bug into a silent fallback.
   raise RuntimeError(f'Codec unexpected failure: {m}') from e
  spent_ms[m]=(time.perf_counter_ns()-t)/1e6
  opts[m]=f
  assert 0<=len(f)+4<=cap
 return opts,spent_ms,errors,cap

def pack_stream(frames,n_values,block_n=BLOCK_N):
 buf=bytearray(b'SGP1'+struct.pack('<II',n_values,block_n))
 for f in frames:buf.extend(struct.pack('<I',len(f)));buf.extend(f)
 return bytes(buf)

def unpack_stream(blob):
 if blob[:4]!=b'SGP1' or len(blob)<12:raise ValueError('bad wrapper')
 n,bn=struct.unpack('<II',blob[4:12]);p=12;result=bytearray()
 while p<len(blob):
  if p+4>len(blob):raise ValueError('bad frame length')
  length=struct.unpack('<I',blob[p:p+4])[0];p+=4
  if p+length>len(blob):raise ValueError('truncated stream')
  result.extend(decode(blob[p:p+length]));p+=length
 if len(result)!=n*8:raise ValueError('number of decoded bytes differs from header')
 return bytes(result)

def stable_softmax(w):
 a=np.asarray(w,dtype=np.float64); e=np.exp(a-a.max());return e/e.sum()

def policy_experiment(arr,seed=RNG_SEED,block_n=BLOCK_N,permute=False):
 blocks,tail,original=block_frames(arr,block_n)
 if permute and blocks:
  order=np.random.default_rng(1989123).permutation(len(blocks))
  blocks=[blocks[int(i)] for i in order]
  original=b''.join(blocks)+tail
 T=len(blocks);K=len(EXPERTS)
 if T<2:raise ValueError('fewer than two complete blocks')
 eta=math.sqrt(8*math.log(K)/T)
 logs=np.zeros(K,dtype=np.float64);cum=np.zeros(K,dtype=np.float64)
 frames_hedge=[];frames_probe=[];frames_ftl=[];frames_zlib=[];frames_oracle=[]
 expected_sum=0.;fixed_best_norm=np.zeros(K);realized_norm=0.
 online_rows=[];total_feedback_ms=0.;total_probe_ms=0.;total_ftl_ms=0.;total_zlib_ms=0.
 cal=min(max(1,T//4),T-1)
 cal_sizes=np.zeros(K,dtype=np.float64);probe_chosen=None
 rng=np.random.default_rng(seed)
 for t,block in enumerate(blocks):
  distribution=stable_softmax(logs)
  action=int(rng.choice(K,p=distribution))
  ftl=int(np.argmin(cum))
  opts,ms,errs,cap=frame_options(block)
  total_feedback_ms+=sum(ms.values())
  length=np.array([len(opts[m])+4 for m in EXPERTS],dtype=np.float64)
  normal=length/cap
  assert np.all(normal>=0) and np.all(normal<=1)
  # This expected loss uses the pre-update distribution; no future leakage.
  expected_sum+=float(distribution@normal)
  fixed_best_norm+=normal
  realized_norm+=normal[action]
  frames_hedge.append(opts[EXPERTS[action]])
  frames_ftl.append(opts[EXPERTS[ftl]])
  frames_zlib.append(opts['zlib6'])
  frames_oracle.append(opts[EXPERTS[int(np.argmin(length))]])
  total_ftl_ms+=ms[EXPERTS[ftl]];total_zlib_ms+=ms['zlib6']
  if t<cal:
   cal_sizes+=length
   frames_probe.append(opts['zlib6'])
   total_probe_ms+=sum(ms.values())
   if t==cal-1:probe_chosen=int(np.argmin(cal_sizes))
  else:
   frames_probe.append(opts[EXPERTS[probe_chosen]])
   total_probe_ms+=ms[EXPERTS[probe_chosen]]
  online_rows.append({'block':t,'seed':seed,'selected':EXPERTS[action],
    'FTL_selected':EXPERTS[ftl],'probe_commit_selected':'zlib6' if t<cal else EXPERTS[probe_chosen],
    'selected_frame_plus_length_bytes':int(length[action]),
    'expected_normalized_loss':float(distribution@normal),
    'realized_normalized_loss':float(normal[action]),
    'raw_cap_bytes':cap,'fallback_count':len(errs)})
  cum+=normal
  logs-=eta*normal
 # Every policy also serializes the residual tail in a self-describing raw frame.
 tail_frame=encode('raw',tail) if tail else None
 results={};full_costs={'hedge_full_feedback':total_feedback_ms,'FTL_oracle_feedback':total_feedback_ms,'probe_commit':total_probe_ms,'fixed_zlib6':total_zlib_ms}
 for k,frames in [('hedge_full_feedback',frames_hedge),('FTL_oracle_feedback',frames_ftl),('probe_commit',frames_probe),('fixed_zlib6',frames_zlib),('oracle_per_block',frames_oracle)]:
  pkg=pack_stream(frames+([tail_frame] if tail_frame is not None else []),len(original)//8,block_n)
  if unpack_stream(pkg)!=original:raise AssertionError('wrapper output is not bit-exact')
  results[k]=len(pkg)
 # Offline best FIXED expert must use full packaged sizes (same global header and tail).
 # The normalization bound is for full blocks only; tail is identical for every policy.
 bound=math.log(K)/eta+eta*T/8
 expected_regret=expected_sum-float(min(fixed_best_norm))
 if expected_regret > bound+1e-8:raise AssertionError('Hedge expected regret inequality violated (implementation bug)')
 cert={'T':T,'K':K,'eta':eta,'theorem_expected_normalized_regret_upper':bound,
   'observed_expected_normalized_regret':expected_regret,
   'observed_realized_normalized_regret':realized_norm-float(min(fixed_best_norm)),
   'bound_vacuous':bound>=T,'full_feedback_trial_ms':total_feedback_ms,
   'all_methods_evaluated_each_block':True,'mode_set':EXPERTS,
   'assumptions':'fixed-size full blocks, complete frame+4-byte length costs capped by raw, nonanticipatory weights, losses divided by 8*block_n+17',
   'not_covered':'selector CPU time, decoder time, cost of candidate library availability, novelty'}
 return {'sizes':results,'times_ms':full_costs,'certificate':cert,'block_trace':online_rows,
  'input_sha256':digest(original),'block_size':block_n,'raw_bytes':len(original),
  'complete_blocks':T,'tail_values':len(tail)//8,'probe_choice':EXPERTS[probe_chosen]}

def safe_exec(name,streams,permute_controls=False):
 root,out=ensure_out(name)
 decisions=[];traces=[];errors=[];controls=[]
 for fam,column,vals,meta in streams:
  try:
   vals=np.asarray(vals,dtype='<f8');r=policy_experiment(vals)
   v=r['sizes'];c=r['certificate'];best=v['oracle_per_block'];fixed=v['fixed_zlib6']
   decisions.append({'family':fam,'column':str(column),'n_values':len(vals),
    'input_sha256':r['input_sha256'],'n_blocks':r['complete_blocks'],
    'raw_input_bytes':r['raw_bytes'],'selected_hedge_bytes':v['hedge_full_feedback'],
    'ftl_bytes':v['FTL_oracle_feedback'],'probe_commit_bytes':v['probe_commit'],
    'fixed_zlib_bytes':fixed,'oracle_per_block_bytes':best,
    'hedge_savings_vs_fixed_zlib_pct':100*(fixed-v['hedge_full_feedback'])/fixed,
    'probe_savings_vs_fixed_zlib_pct':100*(fixed-v['probe_commit'])/fixed,
    'hedge_trial_ms':r['times_ms']['hedge_full_feedback'],
    'probe_trial_ms_estimated':r['times_ms']['probe_commit'],
    'fixed_zlib_trial_ms':r['times_ms']['fixed_zlib6'],
    'regret_bound':c['theorem_expected_normalized_regret_upper'],
    'expected_regret':c['observed_expected_normalized_regret'],
    'realized_regret':c['observed_realized_normalized_regret'],
    'regret_bound_vacuous':c['bound_vacuous'],'prior_family_used_before':meta.get('previously_used',False)})
   traces.extend(dict(family=fam,column=str(column),**entry) for entry in r['block_trace'])
   if permute_controls:
    shuffled=policy_experiment(vals,permute=True)
    controls.append({'family':fam,'column':str(column),'aligned_hedge_bytes':v['hedge_full_feedback'],
      'permuted_hedge_bytes':shuffled['sizes']['hedge_full_feedback'],
      'permuted_sha256':shuffled['input_sha256'],'control':'permute complete blocks; preserves each block, changes order'})
  except Exception as exc:
   errors.append({'family':fam,'column':str(column),'error':repr(exc)})
 savecsv(out,'policy_results.csv',decisions);savecsv(out,'block_trace.csv',traces)
 savecsv(out,'errors.csv',errors);savecsv(out,'permutation_controls.csv',controls)
 return root,out,decisions,errors,controls



from sklearn.datasets import load_diabetes,load_wine,load_breast_cancer,load_digits
streams=[]
for fam,loader,ncols in [('Diabetes',load_diabetes,4),('Wine_classification',load_wine,4),('Breast_cancer',load_breast_cancer,4),('Digits',load_digits,4)]:
 d=loader();a=np.asarray(d.data,dtype='<f8')
 for j in range(min(ncols,a.shape[1])):
  streams.append((fam,f'column_{j:02d}',a[:,j],{'previously_used':True}))
root,out,decisions,errors,controls=safe_exec('SGX_NB83',streams,permute_controls=True)
meta={'status':'COMPLETE' if not errors and zstd is not None else 'INCOMPLETE',
 'streams':len(decisions),'errors':len(errors),'zstandard_available':zstd is not None,
 'hedge_wins_vs_fixed_zlib':sum(r['hedge_savings_vs_fixed_zlib_pct']>0 for r in decisions),
 'probe_wins_vs_fixed_zlib':sum(r['probe_savings_vs_fixed_zlib_pct']>0 for r in decisions),
 'framed_hedge_bytes':sum(r['selected_hedge_bytes'] for r in decisions),
 'framed_zlib_bytes':sum(r['fixed_zlib_bytes'] for r in decisions),
 'average_trial_time_ratio':float(np.mean([r['hedge_trial_ms']/max(1e-9,r['fixed_zlib_trial_ms']) for r in decisions])) if decisions else None,
 'Q1_ready':False,'mechanism_originality':'NOT_ESTABLISHED; Hedge and code-selection well known'}
savejson(out,'feasibility_gate.json',meta);print(json.dumps(meta,indent=2))
finish(root,out,'SGX_NB83')

