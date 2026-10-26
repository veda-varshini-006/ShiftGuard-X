#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB76 — Official_ALP_Portable_Serialization_Gate

Run the pinned upstream ALP binary benchmark on the previously proven UCI inputs, audit encoder/decoder sources, enumerate CTest tests, and require independent full-file roundtrip and metadata evidence. Does NOT mislabel the upstream native BPV as a portable file.

**Evidence discipline:** a native ALP bits/value number is not a portable encoded file; missing evidence remains missing. The notebook saves a result ZIP, including error status if execution is incomplete.
"""


NOTEBOOK_NO=76

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

COMMIT="31ca0ed11c93c99d3f5b5c30e01a3e1c3832d3ce"
REPO=ROOT/"SGX_ALP_NB76"
(DATA:=OUT/"inputs").mkdir(exist_ok=True)
manifest=[]
status={"notebook":76,"pin":COMMIT,"mode":"LOCAL_SYNTHETIC_SMOKE" if SMOKE else "COLAB_REAL_DATA",
        "build":"NOT_RUN","bench":"NOT_RUN","official_native_bpv_available":False,
        "matched_full_file_size_available":False,"input_streams":0}
try:
    if SMOKE:
        sources={"wine_red":[[float((j*i)%23)/10 for i in range(1024)] for j in (1,2)],
                 "wine_white":[[float((j*i)%29)/100 for i in range(2048)] for j in (1,2)]}
        columns={"wine_red":["toy_1","toy_2"],"wine_white":["toy_1","toy_2"]}
    else:
        sources={};columns={}
        urls={
          "wine_red":"https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
          "wine_white":"https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        }
        for family,url in urls.items():
            rawfile=urllib.request.urlopen(url,timeout=90).read()
            (OUT/(family+"_source.csv")).write_bytes(rawfile)
            rd=csv.DictReader(io.StringIO(rawfile.decode("utf-8-sig")), delimiter=';')
            records=list(rd)
            names=list(records[0].keys())
            # Wine red has 1599 rows and white 4898. ALP uses 1024-value vectors.
            n=1024 if family=="wine_red" else 4096
            assert len(records)>=n,(family,len(records))
            pick=["fixed acidity","total sulfur dioxide"]
            sources[family]=[[float(records[i][col]) for i in range(n)] for col in pick]
            columns[family]=pick
            status[family+"_source_sha256"]=shab(rawfile)
    for family,values in sources.items():
        for name,seq in zip(columns[family],values):
            raw=np.asarray(seq,dtype="<f8").tobytes()
            base=f"{family}__{name.replace(' ','_')}"
            dest=DATA/f"{base}.bin.f64"
            dest.write_bytes(raw)
            manifest.append({"id":len(manifest),"column_name":base,"data_type":"double","path":str(dest.resolve()),
              "file_type":"binary","n_values":len(seq),"raw_bytes":len(raw),
              "sha256":shab(raw),"source_family":family,"source":"SYNTHETIC_SMOKE" if SMOKE else "UCI_WINE_QUALITY"})
    cwrite("input_manifest.csv",manifest)
    status["input_streams"]=len(manifest)
    if not SMOKE:
        apt=command(["apt-get","update","-qq"],"apt_update",timeout=360)
        install=command(["apt-get","install","-y","-qq","clang","clang-tidy","cmake","ninja-build","git","build-essential"],
                        "apt_install",timeout=540)
        if install.get("returncode")!=0: raise RuntimeError("Clang / clang-tidy dependency install failed")
        clone=command(["git","clone","https://github.com/cwida/ALP.git",str(REPO)],"git_clone",timeout=300)
        if clone.get("returncode")!=0 and not (REPO/".git").exists():raise RuntimeError("ALP clone failed")
        fetch=command(["git","fetch","--depth","1","origin",COMMIT],"git_fetch_pin",cwd=REPO,timeout=240)
        checkout=command(["git","checkout","--force",COMMIT],"git_checkout",cwd=REPO,timeout=90)
        if checkout.get("returncode")!=0:raise RuntimeError("Cannot checkout pinned official ALP version")
        rev=command(["git","rev-parse","HEAD"],"git_commit",cwd=REPO,timeout=30)
        actual=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if actual!=COMMIT:raise RuntimeError(f"Wrong commit: {actual}")
        config=REPO/"benchmarks"/"your_own_dataset.csv"
        with config.open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f,lineterminator="\n")
            w.writerow(["id","column_name","data_type","path","file_type"])
            for m in manifest:w.writerow([m[x] for x in ["id","column_name","data_type","path","file_type"]])
        # Guard against the exact previous issue: parser does not strip tokens.
        cfg=config.read_bytes()
        assert cfg.count(b"\n")==len(manifest)+1
        assert b"\r" not in cfg and b",binary\n" in cfg
        (OUT/"alp_exact_config.csv").write_bytes(cfg)
        status["alp_config_sha256"]=shab(cfg)
        clang=shutil.which("clang");clangxx=shutil.which("clang++");tidy=shutil.which("clang-tidy")
        if not all([clang,clangxx,tidy]): raise RuntimeError("clang or clang-tidy unavailable")
        env=os.environ.copy();env.update(CC=clang,CXX=clangxx)
        cm=command(["cmake","-S",str(REPO),"-B",str(REPO/"build"),
                     "-DCMAKE_BUILD_TYPE=Release","-DALP_BUILD_BENCHMARKING=ON",
                     "-DALP_ENABLE_VERBOSE_OUTPUT=ON",f"-DCMAKE_C_COMPILER={clang}",
                     f"-DCMAKE_CXX_COMPILER={clangxx}"],"cmake_configure",timeout=300,env=env)
        if cm.get("returncode")!=0:raise RuntimeError("ALP CMake configuration failed")
        build=command(["cmake","--build",str(REPO/"build"),"--target","bench_your_dataset","-j","2"],
                      "build_benchmark",timeout=1200,env=env)
        status["build"]="PASS" if build.get("returncode")==0 else "FAIL"
        if build.get("returncode")!=0: raise RuntimeError("ALP benchmark target compilation failed")
        exe=REPO/"build"/"benchmarks"/"bench_your_dataset"
        if not exe.exists():raise RuntimeError(f"ALP executable not found at {exe}")
        # Inspect the pinned repository for a portable self-contained ALP stream API.
        # String-search hits are leads for review, NOT evidence of serializer support.
        candidates=[]
        needles=("serialize", "deserialize", "compress", "decompress", "encode", "decode", "pack", "unpack")
        for parent in (REPO/"include", REPO/"src", REPO/"benchmarks"):
            if not parent.exists():continue
            for path in parent.rglob("*"):
                if not path.is_file() or path.suffix not in (".hpp",".h",".cpp",".cc",".cxx"):continue
                rel=str(path.relative_to(REPO))
                if len(candidates)>=150:break
                try:content=path.read_text(errors="replace")
                except Exception:continue
                if "alp" not in rel.lower() and not any(x in rel.lower() for x in ("benchmark", "compress", "encode", "bitpack")):continue
                matches={word:content.lower().count(word) for word in needles if word in content.lower()}
                if matches:candidates.append({"path":rel,"sha256":shab(path.read_bytes()),"size":path.stat().st_size,"token_hits":matches})
        jwrite("alp_codec_interface_audit.json",{
            "repo_commit":actual,
            "candidate_code_files":candidates,
            "portable_self_describing_encoder_implemented":False,
            "portable_self_describing_decoder_verified":False,
            "interpretation":"Upstream benchmark is a native BPV metric, not a verified self-contained serialized byte stream. Token hits do not prove an API exists. Exact codec integration needs an independently roundtripping ALP stream and all metadata bytes.",
            "reference":"https://github.com/cwida/ALP/blob/"+actual+"/PRIMITIVES.md"})
        status["portable_alp_roundtrip"]="NOT_IMPLEMENTED"
        status["portable_alp_full_file_bytes"]="NOT_AVAILABLE"
        # No synthetic result or backfilling permitted.
        bench=command([str(exe),"--gtest_filter=ALPBench.bench_alp_on_your_dataset"],
                      "run_official_alp",timeout=900,cwd=REPO/"build")
        status["bench"]="PASS" if bench.get("returncode")==0 else "FAIL"
        res=REPO/"benchmarks"/"your_own_dataset_result.csv"
        if res.exists():
            content=res.read_bytes()
            (OUT/"official_alp_native_results.csv").write_bytes(content)
            with io.StringIO(content.decode("utf-8-sig",errors="replace")) as f:
                parsed=list(csv.DictReader(f))
            status["native_result_rows"]=len(parsed)
            status["native_result_header"]=list(parsed[0]) if parsed else []
            status["official_native_bpv_available"]=bool(bench.get("returncode")==0 and len(parsed)>=len(manifest))
            status["matched_full_file_size_available"]=False
        else:status["native_result_rows"]=0
except Exception as exc:
    status["error"]=repr(exc);(OUT/"failure_traceback.txt").write_text(traceback.format_exc())
status["decision"]="ALP_NATIVE_BENCHMARK_OBTAINED_PORTABLE_PENDING" if status["official_native_bpv_available"] else "ALP_BENCHMARK_INCOMPLETE"
if SMOKE:
    jwrite("alp_codec_interface_audit.json",{"mode":"LOCAL_SYNTHETIC_SMOKE","portable_self_describing_encoder_implemented":False,"portable_self_describing_decoder_verified":False,"note":"No official ALP source analysis in smoke mode"})

# NB76: inspect the actual pinned upstream implementation. These findings do NOT
# certify portable serialization. A verified ALP file must reconstruct by itself.
review={"pin":COMMIT,"upstream_repo":"https://github.com/cwida/ALP",
        "source_checkout_verified":False,"candidate_interfaces":[],
        "native_unit_tests_executed":False,"portable_file_encoder_executed":False,
        "independent_portable_file_decoder_executed":False,
        "complete_alp_file_bytes_recorded":False,"portable_status":"NOT_PROVEN",
        "required_before_fair_size_comparison":[
            "A complete ALP-encoded file including vector selection, exponents, packed data, exceptions, counts and headers",
            "Independent decode from only those file bytes to exactly identical IEEE-754 source bytes",
            "Measured actual file length, including all metadata and dependencies",
            "Same byte range and matched measurement scope for competing codec" ]}
try:
    if REPO.exists() and (REPO/"include"/"alp"/"encoder.hpp").exists():
        review["source_checkout_verified"]=subprocess.run(
            ["git","-C",str(REPO),"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()==COMMIT
        for f in ["include/alp/encoder.hpp","include/alp/decoder.hpp",
                  "include/alp/rd.hpp","benchmarks/benchmark.cpp",
                  "benchmarks/include/benchmark.hpp"]:
            pth=REPO/f
            if not pth.is_file():continue
            lines=pth.read_text(errors="replace").splitlines()
            markers=[(i+1,line.strip()[:280]) for i,line in enumerate(lines)
                     if any(token in line.lower() for token in
                       ("serialize","deserialize","void encode(","void decode(","vectorized_encode","vectorized_decode","exception"))]
            review["candidate_interfaces"].append({"file":f,"sha256":shab(pth.read_bytes()),
                  "line_count":len(lines),"candidate_line_samples":markers[:36]})
        jwrite("source_portability_checklist.json",review)
        # list only; CTest or lower-level encode/decode tests are NOT a portable file proof
        if (REPO/"build").exists():
            tests=command(["ctest","--test-dir",str(REPO/"build"),"-N"],"available_ctest_tests",timeout=45)
            review["native_test_listing_rc"]=tests.get("returncode")
except Exception as e:
    review["source_audit_error"]=repr(e)
    (OUT/"source_audit_traceback.log").write_text(traceback.format_exc())
review["audit_interpretation"]=(
    "The upstream encoder/decoder primitives and native BPV do not themselves "
    "establish a complete serialized ALP format. No invented overhead, file "
    "size, or independent ALP decoder is substituted.")
jwrite("source_portability_checklist.json",review)
status["portable_alp_roundtrip_verified"]=False
status["portable_alp_complete_file_available"]=False
status["source_audit_completed"]=review["source_checkout_verified"]

jwrite("decision.json",status)
print("NB76",status["decision"],status)
finish()

