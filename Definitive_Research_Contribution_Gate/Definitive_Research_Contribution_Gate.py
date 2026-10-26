#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB78 — Definitive_Research_Contribution_Gate

Upload SGX_NB76_results.zip and SGX_NB77_results.zip when prompted. Assesses source identity, correctness, portability, matched sizes and originality without treating failed or missing comparisons as evidence.

**Evidence discipline:** a native ALP bits/value number is not a portable encoded file; missing evidence remains missing. The notebook saves a result ZIP, including error status if execution is incomplete.
"""


import io,zipfile,json,csv,sys,platform,traceback
from pathlib import Path
ROOT=Path('/content' if Path('/content').is_dir() else '/mnt/data')
OUT=ROOT/'SGX_NB78_outputs';OUT.mkdir(parents=True,exist_ok=True)
def read_zip(fn):
    p=ROOT/fn
    if not p.is_file():return None,{"found":False,"reason":"File is not present in runtime"}
    try:
        with zipfile.ZipFile(p) as z:
            d=json.loads(z.read('decision.json'))
            rows={name:list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig')))) for name in z.namelist() if name.endswith('.csv')}
            return (d,rows),{"found":True,"decision":d.get('decision'),"size_bytes":p.stat().st_size}
    except Exception as exc:return None,{"found":False,"error":repr(exc)}
for required in ('SGX_NB76_results.zip','SGX_NB77_results.zip'):
    if not (ROOT/required).is_file():
        try:
            from google.colab import files
            print('Please upload',required)
            files.upload()
        except Exception as e:print('No interactive upload available:',e)
a,sa=read_zip('SGX_NB76_results.zip')
b,sb=read_zip('SGX_NB77_results.zip')
a76=a[0] if a else {}; b77=b[0] if b else {}
ra=a[1] if a else {};rb=b[1] if b else {}
rows=rb.get('heldout_frozen_policy.csv',[])
criteria=[
 ('official_ALP_benchmark_executed',bool(a76.get('official_native_bpv_available')),'Upstream pinned official ALP native benchmark'),
 ('source_input_identical',bool(b77.get('same_input_verified')),'SHA256 bytes verified for ALP reference and competing codecs'),
 ('all_SGX_generic_decodes_bit_exact',bool(b77.get('all_decodes_exact')),'All measured applicable SGX and baseline frames'),
 ('ALP_portable_independent_roundtrip',bool(a76.get('portable_alp_roundtrip_verified')),'Requires independently decoded complete ALP file'),
 ('portable_ALP_file_metadata_counted',bool(a76.get('portable_alp_complete_file_available')),'All ALP metadata and exceptions must count'),
 ('fair_ALP_vs_SGX_size_comparison',bool(b77.get('fair_alp_vs_sgx_full_file_comparison')),'Equivalent complete-file accounting and exactly same input'),
 ('fair_ALP_vs_SGX_native_runtime',False,'No same-language matched end-to-end native pipeline timings in these notebooks'),
 ('original_algorithmic_mechanism',False,'Decimal scaling and exception coding overlap strongly with ALP / prior art'),
 ('independent_family_level_uncertainty',False,'Only four Wine Quality streams in current batch')]
bs=sum(int(float(r['baseline_size_bytes'])) for r in rows)
ss=sum(int(float(r['sgx_size_bytes'])) for r in rows)
report={"notebook":78,"input_archives":{"NB76":sa,"NB77":sb},
        "official_native_ALP_proven":criteria[0][1],
        "equal_accounting_ALP_vs_SGX_proven":criteria[5][1],
        "heldout_policy_streams":len(rows),
        "baseline_bytes":bs,"sgx_bytes":ss,
        "heldout_savings_vs_calibrated_nonalp_baseline_pct":100*(bs-ss)/bs if bs else None,
        "warn":"These heldout byte counts must never be compared directly against ALP native full-stream BPV.",
        "research_contribution_novelty_supported":False,
        "Q1_submission_ready":False,
        "decision":"ALGORITHM_REDESIGN_OR_DIFFERENT_PAPER_QUESTION" if all(x[1] for x in criteria[:3]) else "REQUIRED_BENCHMARK_EVIDENCE_MISSING",
        "next_actions":[
            "Stop presenting decimal-lattice transforms as new without a literature-distinct mechanism.",
            "If ALP head-to-head remains essential, implement and externally validate a complete upstream-ALP serialized file, or explicitly use native BPV in a scope-limited study.",
            "Develop a hypothesis-driven scientific contribution; evaluate against native specialized codecs on new families before paper submission."]}
with (OUT/'publication_decision.json').open('w') as f:json.dump(report,f,indent=2)
with (OUT/'scientific_criteria.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['criterion','passed','evidence_requirement']);w.writeheader()
    for key,ok,explanation in criteria:w.writerow({'criterion':key,'passed':ok,'evidence_requirement':explanation})
print(json.dumps(report,indent=2))
p=ROOT/'SGX_NB78_results.zip'
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
    for file in OUT.iterdir():
        if file.is_file():z.write(file,file.name)
print('Results ZIP:',p)
try:
    from google.colab import files
    files.download(str(p))
except Exception:pass

