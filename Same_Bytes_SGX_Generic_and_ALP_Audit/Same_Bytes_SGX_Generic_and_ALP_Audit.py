#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X — NB71: Same_Bytes_SGX_Generic_and_ALP_Audit

Imports NB70 raw input ZIP; runs exactly the NB68 QPK1 codec and strong generic and specialized methods; freezes an internal held-out selector. Native ALP bits/value is separated because ALP's official benchmark does not export a portable complete stream.

Run the whole notebook in Google Colab CPU. It exports an auditable ZIP, even if an external step fails. A successful official ALP native benchmark does **not** establish matched portable file-size accounting or originality.
"""


NOTEBOOK_NO=71

import sys, os, csv, io, json, hashlib, time, struct, zipfile, traceback, platform, statistics, subprocess, shutil, urllib.request
from pathlib import Path
import numpy as np
ROOT=Path("/content" if Path("/content").is_dir() else "/mnt/data")
OUT=ROOT/f"SGX_NB{NOTEBOOK_NO}_outputs"
OUT.mkdir(parents=True,exist_ok=True)
SMOKE=os.environ.get("SGX_LOCAL_SMOKE_ONLY","0")=="1"
def shab(b): return hashlib.sha256(b).hexdigest()
def jwrite(fn,obj): (OUT/fn).write_text(json.dumps(obj,indent=2,default=str))
def cwrite(fn,rows):
    if not rows: return
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/fn).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=columns,lineterminator="\n")
        w.writeheader(); w.writerows(rows)
def finish():
    p=ROOT/f"SGX_NB{NOTEBOOK_NO}_results.zip"
    with zipfile.ZipFile(p,"w",zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():z.write(f,f.relative_to(OUT))
    print("RESULT ZIP",p,"SIZE",p.stat().st_size)
    try:
        from google.colab import files
        files.download(str(p))
    except ImportError: pass
    except Exception as exc: print("Download from Files sidebar:",exc)
    return p
def command(cmd, label, timeout=180, cwd=None, env=None):
    print("RUN",label,cmd)
    st=time.perf_counter()
    try:
        p=subprocess.run(cmd,cwd=cwd,env=env,capture_output=True,text=True,timeout=timeout)
        result={"label":label,"command":cmd,"returncode":p.returncode,"elapsed_s":round(time.perf_counter()-st,2)}
        (OUT/f"{label}.log").write_text(p.stdout+"\n-----STDERR-----\n"+p.stderr)
        print(label,"exit=",p.returncode, "last output:",(p.stdout+p.stderr)[-450:])
        return result
    except Exception as e:
        result={"label":label,"command":cmd,"error":repr(e),"elapsed_s":round(time.perf_counter()-st,2)}
        (OUT/f"{label}.log").write_text(traceback.format_exc())
        print(label,"exception",repr(e))
        return result

import math,zlib,lzma,importlib.util
if not importlib.util.find_spec("zstandard") and not SMOKE:
    command([sys.executable,"-m","pip","-q","install","zstandard"],"optional_zstandard",timeout=180)
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

# NB71 requires the actual NB70 result archive, not independently reconstructed values.
p=ROOT/"SGX_NB70_results.zip"
if not p.exists() and not SMOKE:
    try:
        from google.colab import files
        print("Upload SGX_NB70_results.zip from NB70 now")
        files.upload()
    except Exception as e:print("Manual upload required",e)
state={"notebook":71,"NB70_archive_found":p.exists(),"alp_native_available":False,
       "complete_file_alp_comparison":False,"decision":"INCOMPLETE"}
rows=[];policy=[];alp_reference=[];errors=[]
try:
    if not p.exists():raise RuntimeError("NB70 ZIP not available; upload it to /content")
    with zipfile.ZipFile(p) as z:
        names=set(z.namelist())
        manifest=list(csv.DictReader(io.StringIO(z.read("input_manifest.csv").decode())))
        decision=json.loads(z.read("decision.json"))
        state["NB70_decision"]=decision.get("decision")
        state["source_type"]=decision.get("mode")
        if "official_alp_native_results.csv" in names:
            alp_rows=list(csv.DictReader(io.StringIO(z.read("official_alp_native_results.csv").decode("utf-8-sig"))))
            state["alp_native_available"]=bool(decision.get("official_native_bpv_available"))
        else:alp_rows=[]
        for m in manifest:
            stream=m["column_name"]
            arc=f"inputs/{stream}.bin.f64"
            if arc not in names: raise RuntimeError(f"Missing source bytes {arc}")
            r=z.read(arc)
            if shab(r)!=m["sha256"] or len(r)!=int(m["raw_bytes"]):
                raise RuntimeError("RAW SOURCE IDENTITY MISMATCH "+stream)
            n=len(r)//8
            # Same raw bytes for ALP, generic, and lattice. Compute sizes only when codec is reversible.
            full={}
            for mode in METHODS:
                try:
                    d=measured(mode,r,reps=2 if SMOKE else 3)
                    full[mode]=d
                    rows.append({"stream":stream,"source":m["source"],"n":n,"split":"full",
                                 "mode":mode,"sha256":shab(r),**d})
                except (ValueError, OverflowError,ArithmeticError) as e:
                    errors.append({"stream":stream,"split":"full","mode":mode,"status":"INAPPLICABLE","reason":repr(e)})
                except Exception as e:
                    errors.append({"stream":stream,"split":"full","mode":mode,"status":"ERROR","reason":repr(e)})
            alp_match=next((a for a in alp_rows if a.get("column",a.get("column_name",""))==stream),None)
            if alp_match is None:
                # Column label usually in `column`; do not assume index-only equivalence.
                alp_reference.append({"stream":stream,"n":n,"raw_sha256":shab(r),"alp_matched":False})
            else:
                bpv_text=alp_match.get("size","")
                try:bpv=float(bpv_text)
                except (ValueError,TypeError):bpv=None
                alp_reference.append({"stream":stream,"n":n,"raw_sha256":shab(r),
                                       "alp_matched":True,"alp_native_bpv":bpv,
                                       "alp_native_size_source":"upstream benchmark size field",
                                       "alp_total_file_bytes":"NOT_AVAILABLE",
                                       "warning":"Native BPV may use rowgroup/vector padding and excludes application container; not a same-accounting comparison"})
            cut=max(256,n//4)
            cal=r[:cut*8];hold=r[cut*8:];shaheld=shab(hold)
            if not hold:continue
            metric_cal={};selection_elapsed=0
            for mode in METHODS:
                t=time.perf_counter_ns()
                try:metric_cal[mode]=measured(mode,cal,reps=1)
                except (ValueError,OverflowError,ArithmeticError):pass
                finally:selection_elapsed+=(time.perf_counter_ns()-t)/1e6
            valid_base={k:v for k,v in metric_cal.items() if k in NONLATTICE}
            valid_lattice={k:v for k,v in metric_cal.items() if k in CANDIDATES}
            if not valid_base:raise RuntimeError("No calibration nonlattice mode for "+stream)
            base=min(valid_base,key=lambda k:valid_base[k]["bytes"])
            sgx=base
            if valid_lattice:
                candidate=min(valid_lattice,key=lambda k:valid_lattice[k]["bytes"])
                if valid_lattice[candidate]["bytes"]<=0.85*valid_base[base]["bytes"]:
                    sgx=candidate
            def held(mode):
                try:return mode,measured(mode,hold,reps=2 if SMOKE else 3)
                except (ValueError,OverflowError,ArithmeticError):
                    fallback=measured("zlib6",hold,reps=2 if SMOKE else 3)
                    return "zlib6",fallback
            actual_base,bm=held(base)
            actual_sgx,sm=held(sgx)
            policy.append({"stream":stream,"source":m["source"],"n_cal":cut,"n_heldout":n-cut,
                  "raw_sha256_full":shab(r),"raw_sha256_heldout":shaheld,
                  "baseline_cal":base,"sgx_cal":sgx,"baseline_actual":actual_base,"sgx_actual":actual_sgx,
                  "baseline_heldout_bytes":bm["bytes"],"sgx_heldout_bytes":sm["bytes"],
                  "savings_vs_calibrated_baseline_pct":100*(bm["bytes"]-sm["bytes"])/bm["bytes"],
                  "baseline_encode_ms":bm["encode_ms_median"],"sgx_encode_ms":sm["encode_ms_median"],
                  "sgx_decode_ms":sm["decode_ms_median"],"selection_time_ms_all_trials":selection_elapsed,
                  "bit_exact":True,"selection_margin":0.15})
    state["streams_verified"]=len(manifest)
    state["full_codec_measurements"]=len(rows)
    state["frozen_policy_streams"]=len(policy)
    state["official_native_bpv_count"]=sum(a.get("alp_native_bpv") is not None for a in alp_reference)
    state["decision"]="NATIVE_ALP_BPV_AND_MATCHED_CODEC_FRAMES_AVAILABLE" if (
        state["alp_native_available"] and state["official_native_bpv_count"]==len(manifest)) else "PARTIAL_COMPARISON_ONLY"
    state["complete_file_alp_comparison"]=False
except Exception as e:
    state["error"]=repr(e);(OUT/"error_traceback.txt").write_text(traceback.format_exc())
cwrite("full_same_input_generic_and_sgx.csv",rows)
cwrite("frozen_policy_heldout.csv",policy)
cwrite("native_alp_reference_unmatched_accounting.csv",alp_reference)
cwrite("inapplicable_or_errors.csv",errors)
jwrite("decision.json",state)
print("NB71 STATUS:",state)
finish()

