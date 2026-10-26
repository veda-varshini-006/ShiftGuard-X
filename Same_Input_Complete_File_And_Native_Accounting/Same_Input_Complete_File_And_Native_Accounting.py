#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB77 — Same_Input_Complete_File_And_Native_Accounting

Upload SGX_NB76_results.zip when prompted. Uses exactly its four verified input byte streams, fully self-describing SGX/generic frames with bit-exact decoding, calibration-frozen held-out policies, and separately reported native ALP BPV.

**Evidence discipline:** a native ALP bits/value number is not a portable encoded file; missing evidence remains missing. The notebook saves a result ZIP, including error status if execution is incomplete.
"""


NOTEBOOK_NO=77

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



# NB77: strictly requires NB76 original input bytes and manifest.
source_zip=ROOT/"SGX_NB76_results.zip"
if not source_zip.is_file() and not SMOKE:
    try:
        from google.colab import files
        print("Upload SGX_NB76_results.zip here (from NB76).")
        files.upload()
    except Exception as exc:
        print("Manual upload via Files sidebar is also supported:",exc)
state={"notebook":77,"NB76_archive_found":source_zip.is_file(),
       "official_alp_native_bpv_available":False,"complete_alp_portable_size_available":False,
       "same_input_verified":False,"all_decodes_exact":False,
       "fair_alp_vs_sgx_full_file_comparison":False,"decision":"INCOMPLETE"}
allmeasure=[];policy=[];alp_native=[];comparison=[];errors=[];manifest_out=[]
try:
    if not source_zip.is_file():raise RuntimeError("SGX_NB76_results.zip missing in Colab /content")
    with zipfile.ZipFile(source_zip) as z:
        names=set(z.namelist());manifest=list(csv.DictReader(io.StringIO(z.read("input_manifest.csv").decode())))
        s76=json.loads(z.read("decision.json"))
        state["input_provenance"]=s76.get("mode","UNKNOWN")
        state["official_alp_native_bpv_available"]=bool(s76.get("official_native_bpv_available"))
        state["portable_alp_roundtrip_reported_by_NB76"]=bool(s76.get("portable_alp_roundtrip_verified",False))
        assert len(manifest)>0
        native=(list(csv.DictReader(io.StringIO(z.read("official_alp_native_results.csv").decode())))
                if "official_alp_native_results.csv" in names else [])
        good=0
        for m in manifest:
            name=m["column_name"];key=f"inputs/{name}.bin.f64"
            if key not in names:raise RuntimeError("Missing source input "+key)
            raw=z.read(key);digest=shab(raw)
            assert digest==m["sha256"],"Source SHA256 mismatch for "+name
            assert len(raw)==int(m["raw_bytes"])==int(m["n_values"])*8
            n=len(raw)//8;good+=1
            manifest_out.append({"stream":name,"n_full":n,"source":m["source"],"input_sha256":digest,
                                 "identical_full_input_confirmed":True})
            bench=next((r for r in native if r.get("column",r.get("column_name",""))==name),None)
            alp_native.append({"stream":name,"input_sha256":digest,"alp_row_present":bool(bench),
                "alp_bpv_native":bench.get("size","") if bench else "",
                "alp_cycles_per_value_encode_native":bench.get("compression_speed(cycles_per_value)","") if bench else "",
                "alp_cycles_per_value_decode_native":bench.get("decompression_speed(cycles_per_value)","") if bench else "",
                "alp_portable_file_bytes":"NOT_AVAILABLE",
                "size_regime":"ALP_INTERNAL_NATIVE_BPV_ONLY; not directly comparable to full framed SGX"})
            full={}
            for meth in METHODS:
                try:
                    metrics=measured(meth,raw,reps=2 if SMOKE else 3)
                    full[meth]=metrics
                    allmeasure.append({"stream":name,"n_full":n,"split":"full",
                        "input_sha256":digest,"method":meth,**metrics})
                except (ValueError,OverflowError,ArithmeticError) as exc:
                    errors.append({"stream":name,"method":meth,"status":"INAPPLICABLE","error":repr(exc)})
                except Exception as exc:
                    errors.append({"stream":name,"method":meth,"status":"ERROR","error":repr(exc)})
            if not any(x.startswith("lattice_") for x in full):
                errors.append({"stream":name,"status":"NO_LATTICE_METHOD_VALID"})
            prior=[(k,v) for k,v in full.items() if k in NONLATTICE]
            cand=[(k,v) for k,v in full.items() if k in CANDIDATES]
            baseline=min(prior,key=lambda kv:kv[1]["bytes"]) if prior else None
            lattice=min(cand,key=lambda kv:kv[1]["bytes"]) if cand else None
            comparison.append({"stream":name,"input_sha256":digest,"full_input_n":n,
                 "best_nonlattice_oracle_bytes":baseline[1]["bytes"] if baseline else None,
                 "best_nonlattice_oracle_method":baseline[0] if baseline else "",
                 "best_lattice_oracle_bytes":lattice[1]["bytes"] if lattice else None,
                 "best_lattice_oracle_method":lattice[0] if lattice else "",
                 "alp_native_bpv_unmatched_accounting":bench.get("size","") if bench else "",
                 "alp_portable_file_bytes":None,"fair_alp_size_winner": "NOT_DETERMINABLE"})
            # Fairly compare deployed policies on only the SAME held-out portion,
            # calibrated independently; NEVER conflate with upstream full-input BPV.
            cut=max(256,n//4);cal=raw[:8*cut];test=raw[8*cut:]
            cal_metrics={};trial_ms=0.0
            for meth in METHODS:
                st=time.perf_counter_ns()
                try:cal_metrics[meth]=measured(meth,cal,1)
                except (ValueError,OverflowError,ArithmeticError):pass
                except Exception as exc:errors.append({"stream":name,"method":meth,"status":"CAL_ERROR","error":repr(exc)})
                finally:trial_ms+=(time.perf_counter_ns()-st)/1e6
            bases=[x for x in cal_metrics if x in NONLATTICE]
            cands=[x for x in cal_metrics if x in CANDIDATES]
            if not bases:raise RuntimeError("No usable baseline on "+name)
            base=min(bases,key=lambda x:cal_metrics[x]["bytes"])
            selected=base
            if cands:
                winner=min(cands,key=lambda x:cal_metrics[x]["bytes"])
                if cal_metrics[winner]["bytes"] <= 0.85*cal_metrics[base]["bytes"]:selected=winner
            def run_safe(method):
                start=time.perf_counter_ns()
                try:
                    ms=measured(method,test,2 if SMOKE else 3)
                    return method,ms,False,(time.perf_counter_ns()-start)/1e6
                except (ValueError,OverflowError,ArithmeticError):
                    failed_ms=(time.perf_counter_ns()-start)/1e6
                    ms=measured("zlib6",test,2 if SMOKE else 3)
                    return "zlib6",ms,True,failed_ms
            base_actual,bm,bfail,bfailed_ms=run_safe(base)
            sel_actual,sm,sfail,sfailed_ms=run_safe(selected)
            policy.append({"stream":name,"n_calibration":cut,"n_heldout":n-cut,
                "sha256_heldout":shab(test),"baseline_selected_cal":base,"sgx_selected_cal":selected,
                "baseline_actual_heldout":base_actual,"sgx_actual_heldout":sel_actual,
                "baseline_size_bytes":bm["bytes"],"sgx_size_bytes":sm["bytes"],
                "saved_pct_vs_calibrated_baseline":100*(bm["bytes"]-sm["bytes"])/bm["bytes"],
                "baseline_encode_ms":bm["encode_ms_median"]+bfailed_ms if bfail else bm["encode_ms_median"],
                "sgx_encode_ms":sm["encode_ms_median"]+sfailed_ms if sfail else sm["encode_ms_median"],
                "baseline_decode_ms":bm["decode_ms_median"],"sgx_decode_ms":sm["decode_ms_median"],
                "calibration_selection_ms_all_trials":trial_ms,
                "baseline_fell_back":bfail,"sgx_fell_back":sfail,"exact":True})
        state["same_input_verified"]=good==len(manifest)
        state["streams_verified"]=good
        state["bit_exact_measurement_rows"]=len(allmeasure)
        state["all_decodes_exact"]=all(r["exact"] for r in allmeasure) and all(r["exact"] for r in policy)
        state["heldout_rows"]=len(policy)
        state["portable_alp_complete_file_available"]=False
        state["fair_alp_vs_sgx_full_file_comparison"]=False
        state["decision"]=("MATCHED_SOURCE_NATIVE_ALP_VS_FULL_FRAMES_SEPARATE_ACCOUNTING" if
             state["same_input_verified"] and state["official_alp_native_bpv_available"] and
             state["all_decodes_exact"] else "INCOMPLETE")
except Exception as exc:
    state["error"]=repr(exc);(OUT/"error_traceback.log").write_text(traceback.format_exc())
cwrite("verified_inputs.csv",manifest_out)
cwrite("all_full_file_codecs.csv",allmeasure)
cwrite("heldout_frozen_policy.csv",policy)
cwrite("native_alp_not_file_comparable.csv",alp_native)
cwrite("same_input_different_accounting.csv",comparison)
cwrite("inapplicable_or_errors.csv",errors)
jwrite("decision.json",state)
print("NB77",state)
finish()

