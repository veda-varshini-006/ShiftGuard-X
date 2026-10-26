#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X — NB72: Publication_Feasibility_Evidence_Gate

Combines NB70 and NB71 archives, audits matched input identity and measurement accounting, then makes a conservative science/publication decision based on explicit evidence criteria.

Run the whole notebook in Google Colab CPU. It exports an auditable ZIP, even if an external step fails. A successful official ALP native benchmark does **not** establish matched portable file-size accounting or originality.
"""


NOTEBOOK_NO=72

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

# Requires NB70 and NB71 ZIPs; uploads are requested only when missing.
required=[ROOT/"SGX_NB70_results.zip",ROOT/"SGX_NB71_results.zip"]
missing=[x.name for x in required if not x.exists()]
if missing and not SMOKE:
    try:
        from google.colab import files
        print("Upload these files:",missing);files.upload()
    except Exception as exc:print("Please upload via Colab Files sidebar:",repr(exc))
notices=[];sources={}
for p in required:
    if not p.exists():
        notices.append(f"MISSING: {p.name}")
        continue
    try:
        with zipfile.ZipFile(p) as z:
            sources[p.name]={f.filename:f.file_size for f in z.infolist()}
            for name in ["decision.json","frozen_policy_heldout.csv","native_alp_reference_unmatched_accounting.csv"]:
                if name in z.namelist():
                    (OUT/(p.stem+"_"+name)).write_bytes(z.read(name))
    except Exception as exc:notices.append(f"Could not read {p.name}: {exc}")
def readjson(p):
    return json.loads(p.read_text()) if p.exists() else {}
def readcsv(p):
    if not p.exists():return []
    with p.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
a=readjson(OUT/"SGX_NB70_results_decision.json")
b=readjson(OUT/"SGX_NB71_results_decision.json")
rows=readcsv(OUT/"SGX_NB71_results_frozen_policy_heldout.csv")
reference=readcsv(OUT/"SGX_NB71_results_native_alp_reference_unmatched_accounting.csv")
def numeric(row,key,default=0.0):
    try:return float(row[key])
    except Exception:return default
den=sum(numeric(r,"baseline_heldout_bytes") for r in rows)
num=sum(numeric(r,"sgx_heldout_bytes") for r in rows)
savings=(100*(den-num)/den) if den else None
native_alp=bool(a.get("official_native_bpv_available") and b.get("alp_native_available"))
full_file_alp=bool(b.get("complete_file_alp_comparison")) # NB71 intentionally False
matched_native_count=sum((r.get("alp_matched") in (True,"True","true","1")) for r in reference)
print("ALP native benchmark proven?",native_alp)
print("End-to-end ALP same-accounting comparison?",full_file_alp)
print("SGX savings on these matched generic inputs only:",savings)
criteria=[
 {"criterion":"NB70 verified pinned ALP benchmark on explicit same raw input","passed":native_alp,
  "evidence":"official native ALP rows and pinned commit, not full-file output"},
 {"criterion":"SGX and generic methods bit-exact on identical source bytes","passed":bool(rows) and all(r.get("bit_exact") in ("True","true","1") for r in rows),
  "evidence":"NB71 archive and input hashes"},
 {"criterion":"ALP and SGX measured with matched complete-file accounting","passed":full_file_alp,
  "evidence":"UNPROVEN: ALP reported native BPV, no independent portable complete file"},
 {"criterion":"Favorable cross-family independent native-codec throughput","passed":False,
  "evidence":"Native ALP/Chimp/Gorilla and SGX not benchmarked with equivalent timing and selection cost"},
 {"criterion":"Distinct algorithm mechanism supported by literature and ablations","passed":False,
  "evidence":"Decimal scaling, packing, exceptions and codec selection overlap known ALP / PseudoDecimals techniques"},
 {"criterion":"Publication-ready dataset breadth/statistical evidence","passed":False,
  "evidence":"NB70 uses limited wine columns; previous NB69 includes only 3 dataset families and small subsets"},
]
cwrite("gate_criteria.csv",criteria)
related=[
 {"work":"ALP: Adaptive Lossless floating-Point Compression","url":"https://github.com/cwida/ALP",
  "overlap":"decimal-to-integer, exceptions, bit-packing and adaptive coding"},
 {"work":"Gorilla: A Fast, Scalable, In-Memory Time Series Database","url":"https://www.vldb.org/pvldb/vol8/p1816-teller.pdf",
  "overlap":"XOR-based floating-point compression"},
 {"work":"Chimp: Efficient Lossless Floating Point Compression for Time Series Databases","url":"https://github.com/panagiotisl/chimp",
  "overlap":"XOR-based numerical compression"},
 {"work":"Sprintz: Time Series Compression for the Internet of Things","url":"https://arxiv.org/abs/1808.02515",
  "overlap":"multivariate prediction/residual coding"}
]
cwrite("prior_art_reference_links.csv",related)
gate={
 "notebook":72,"nb70_status":a.get("decision","MISSING"),
 "nb71_status":b.get("decision","MISSING"),
 "official_alp_native_bpv_available":native_alp,
 "same_input_native_alp_rows":matched_native_count,
 "matched_complete_file_alp_comparison":full_file_alp,
 "heldout_streams":len(rows),
 "sgx_policy_bytes":num if rows else None,
 "baseline_policy_bytes":den if rows else None,
 "sgx_savings_pct_vs_calibrated_baseline":savings,
 "passed_gate_criteria":sum(bool(r["passed"]) for r in criteria),
 "total_gate_criteria":len(criteria),
 "known_novelty":False,
 "Q1_ready":False,
 "recommendation":"CONTINUE_METHOD_COMPARISON_OR_REDESIGN_RESEARCH_QUESTION" if rows and native_alp else "COLLECT_MISSING_COMPARABLE_EVIDENCE",
 "prohibition":"Do not claim SGX beats ALP from native BPV vs complete framed sizes.",
 "missing_or_unreadable_archives":notices
}
jwrite("publication_feasibility_gate.json",gate)
print(json.dumps(gate,indent=2))
finish()

