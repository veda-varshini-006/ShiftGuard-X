#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB75 — Conservative_Publication_Decision_Gate

Upload NB73 and NB74 result ZIPs when prompted. Checks source SHA256 identity and decoding claims before reporting any result; requires actual portable ALP evidence for a fair ALP-vs-SGX winner, and never asserts Q1 readiness.

**Research-safety rule:** only matched input bytes and matching codec accounting can justify a size/throughput win. Lossless byte equality is checked for each implemented codec. Never infer novelty from compression ratios alone.
"""


NOTEBOOK_NO=75

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


import math
from collections import defaultdict

expected={"NB73":"SGX_NB73_results.zip","NB74":"SGX_NB74_results.zip"}
missing=[str(ROOT/f) for f in expected.values() if not (ROOT/f).exists()]
if missing and not SMOKE:
    try:
        from google.colab import files
        print("Please upload the two previous result ZIPs:",list(expected.values()))
        files.upload()
    except ImportError:pass
    except Exception as ex:print("Upload issue:",repr(ex))

read_status={};raw_manifest=[];alp_native=[];nb74_full=[];nb74_hold=[];nb74_account=[]
for label,fname in expected.items():
    p=ROOT/fname
    if not p.exists():
        read_status[label]={"exists":False,"error":"ZIP missing from current runtime"};continue
    try:
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None,"corrupt ZIP"
            present=set(z.namelist())
            decision=json.loads(z.read("decision.json")) if "decision.json" in present else {}
            read_status[label]={"exists":True,"decision":decision}
            if label=="NB73":
                if "input_manifest.csv" in present:
                    raw_manifest=list(csv.DictReader(io.StringIO(z.read("input_manifest.csv").decode())))
                if "official_alp_native_results.csv" in present:
                    alp_native=list(csv.DictReader(io.StringIO(z.read("official_alp_native_results.csv").decode())))
                for m in raw_manifest:
                    arc="inputs/"+m["column_name"]+".bin.f64"
                    if arc not in present or shab(z.read(arc))!=m["sha256"]:
                        raise ValueError("Input ZIP hash mismatch: "+arc)
            else:
                def read_csv(name):
                    return list(csv.DictReader(io.StringIO(z.read(name).decode()))) if name in present else []
                nb74_full=read_csv("full_same_input_generic_and_sgx.csv")
                nb74_hold=read_csv("frozen_policy_heldout.csv")
                nb74_account=read_csv("same_full_input_separate_accounting.csv")
    except Exception as exc:read_status[label]={"exists":False,"error":repr(exc)}

sha_set={m.get("sha256") for m in raw_manifest}
consistent=bool(raw_manifest and nb74_full and all(r.get("sha256") in sha_set for r in nb74_full) and
                set(m["column_name"] for m in raw_manifest)==set(r["stream"] for r in nb74_account))
roundtrip=bool(nb74_full and all(str(r.get("exact")).lower()=="true" for r in nb74_full))
official=bool(len(alp_native)>=len(raw_manifest)>0 and read_status.get("NB73",{}).get("decision",{}).get("official_native_bpv_available"))
complete_alp=False  # Only change after verified real independently-decoding ALP portable stream evidence.

baseline=sum(int(r['baseline_heldout_bytes']) for r in nb74_hold) if nb74_hold else None
policy=sum(int(r['sgx_heldout_bytes']) for r in nb74_hold) if nb74_hold else None
saving=100*(baseline-policy)/baseline if baseline else None

criteria=[
 {'criterion':'Official ALP native benchmark obtained','passed':official,'basis':'Successful pinned upstream ALP BPV rows'},
 {'criterion':'Raw input hashes reconciled','passed':consistent,'basis':'NB73 ZIP SHA256 inputs agree with NB74 full-measurement rows'},
 {'criterion':'Full-stream non-ALP byte-exactness','passed':roundtrip,'basis':'Every NB74 full-stream method decoded exactly'},
 {'criterion':'Full-file portable ALP encode/decode verified','passed':complete_alp,'basis':'No proven independent ALP portable stream with header and exception metadata'},
 {'criterion':'Equal-accounting ALP versus SGX file sizes','passed':False,'basis':'ALP native bits/value is not complete-file bytes'},
 {'criterion':'Equal-scope end-to-end throughput ALP vs SGX','passed':False,'basis':'ALP cycles/value and Python framed timings are not matched end-to-end measurements'},
 {'criterion':'New algorithm distinct from ALP and prior art','passed':False,'basis':'Decimal scaling, exceptions, packing, adaptive choice are established'},
 {'criterion':'External statistical support for Q1 submission','passed':False,'basis':'Only four matched Wine streams and no proven novelty or matched native competitors'}]
cwrite('scientific_criteria.csv',criteria)
summary={'notebook':75,'read_status':read_status,'all_required_inputs_present':all(x.get('exists') for x in read_status.values()),
'official_native_alp':official,'identical_input_verified':consistent,'bit_exact_verified':roundtrip,
'portable_alp_roundtrip':complete_alp,'equal_accounting_alp_vs_sgx':False,
'heldout_baseline_bytes':baseline,'heldout_sgx_bytes':policy,'heldout_savings_vs_calibrated_baseline_pct':saving,
'native_vs_portable_sizes_directly_comparable':False,'Q1_ready':False,'scientific_novelty_claim_supported':False,
'decision':'RETIRE_AS_NOVEL_CODEC__POTENTIAL_COMPARATIVE_STUDY' if consistent and roundtrip and official else 'EVIDENCE_INCOMPLETE',
'required_next_evidence':[
 'Verified independently decodable official ALP stream with complete size including exceptions and metadata, OR explicit scope-limited native BPV research question',
 'Matched ALP versus SGX encoding and decoding timing with identical included costs',
 'Formal distinction from ALP, PseudoDecimals, FastPFOR and similar prior art',
 'More independent dataset families and confidence intervals without using a stream-level oracle baseline']}
jwrite('publication_decision.json',summary)
print('RESEARCH GATE:',summary['decision'],'policy heldout savings:',saving)
finish()

