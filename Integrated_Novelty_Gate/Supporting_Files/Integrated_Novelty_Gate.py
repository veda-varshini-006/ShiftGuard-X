#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X NB57F — Integrated novelty gate

This is a standalone cell. Upload `SGX_NB55_results.zip` and `SGX_NB56F_results.zip` when prompted. If NB56F did not finish, the gate will remain incomplete. Literature overlap is evidence against treating decimal-lattice operations as inherently novel.
"""


import csv,json,zipfile,io,statistics,hashlib,sys,platform,traceback
from pathlib import Path
ROOT=Path('/content' if Path('/content').exists() else '/mnt/data')
OUT=ROOT/'SGX_NB57F_outputs';OUT.mkdir(parents=True,exist_ok=True)
def emit_csv(name,rows):
 if rows:
  keys=list(dict.fromkeys(k for row in rows for k in row))
  with (OUT/name).open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def emit_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=str))
def delivery():
 dest=ROOT/'SGX_NB57F_results.zip'
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
  for p in OUT.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(OUT))
 print('RESULTS ZIP',dest)
 try:
  from google.colab import files;files.download(str(dest))
 except Exception:pass
def safe_read_zip(path,jsonfilename):
 with zipfile.ZipFile(path) as z:
  names=z.namelist()
  if jsonfilename not in names:raise FileNotFoundError(jsonfilename+' not in '+str(names[:20]))
  return json.loads(z.read(jsonfilename))



# Request the exact NB55 and repaired NB56F result archives in Colab if they are absent.
needed=('SGX_NB55_results.zip','SGX_NB56F_results.zip')
missing=[x for x in needed if not (ROOT/x).exists()]
if missing:
 print('Required uploads:', missing)
 try:
  from google.colab import files
  upload=files.upload()
  # files.upload writes into current working directory. Put a copy in ROOT when needed.
  import shutil
  for name in needed:
   src=Path(name)
   if src.exists() and not (ROOT/name).exists(): shutil.copy2(src,ROOT/name)
 except ImportError:
  print('Colab upload is unavailable. Place files in',ROOT,'and rerun.')

# Related work references are recorded with precise links; this is a PRIOR ART SCREEN,
# not a claim that every paper was comprehensively reviewed.
PAPERS=[
 {'work':'ALP','year':2024,'source':'ACM SIGMOD 2024','url':'https://duckdb.org/library/alp/','mechanism':'Decimal-origin reconstruction, vector-wide factors, exceptions, FOR bitpacking, sampled adaptation','overlap':'CRITICAL','implication':'Our decimal scale + exception + integer packing is substantially covered by ALP.'},
 {'work':'ALP in Apache Parquet','year':2026,'source':'Apache Parquet format specification','url':'https://parquet.apache.org/blog/2026/09/22/alp-adaptive-lossless-floating-point-encoding-in-apache-parquet/','mechanism':'Standardized decimal floating-point encoding with FOR and exception representation','overlap':'CRITICAL','implication':'A common deployment use case is already standardized; SGX needs differentiating capabilities.'},
 {'work':'PseudoDecimals','year':2024,'source':'Explained in ALP and CMU advanced database systems course','url':'https://15721.courses.cs.cmu.edu/spring2024/notes/03-data2.pdf','mechanism':'Round-trip-tested decimal coefficients, exponents and exceptions','overlap':'CRITICAL','implication':'Exact decimal scaling itself is established prior art.'},
 {'work':'SIMD-FastPFOR','year':2015,'source':'Software: Practice and Experience','url':'https://doi.org/10.1002/spe.2203','mechanism':'Frame-of-reference integer bit packing and exception organization','overlap':'HIGH','implication':'Integer packing, vectorization and exception streams are not independently new.'},
 {'work':'Chimp','year':2022,'source':'PVLDB 15(11)','url':'https://vldb.org/pvldb/vol15/p3058-liakos.pdf','mechanism':'XOR-based floating-point time series representation','overlap':'MEDIUM','implication':'Established alternative specialized baseline missing from current benchmark.'},
 {'work':'Elf / Elf+','year':2023,'source':'PVLDB / arXiv','url':'https://arxiv.org/abs/2306.16053','mechanism':'Lossless erasure/restore of floating-point mantissa bits with XOR coding','overlap':'MEDIUM','implication':'Different coding family; important comparison and possible novelty collision for bit manipulations.'},
 {'work':'Elf* / SElf*','year':2025,'source':'IEEE Internet of Things Journal','url':'https://ieeexplore.ieee.org/document/10949160/','mechanism':'Adaptive encoding rules for significands and leading/trailing zero lengths','overlap':'MEDIUM','implication':'Adaptive formatting/zero-length coding has recent strong prior art.'},
]
COMPONENTS=[
 {'SGX_component':'Exact float64-to-scaled-integer test','prior_art':'PseudoDecimals, ALP','novelty_evidence':'NO'},
 {'SGX_component':'Bit-exact exception mask and raw exception values','prior_art':'PseudoDecimals, ALP, FastPFOR','novelty_evidence':'NO'},
 {'SGX_component':'Frame-of-reference integer packing','prior_art':'ALP, SIMD-FastPFOR','novelty_evidence':'NO'},
 {'SGX_component':'zlib for masks, integers and exceptions','prior_art':'General-purpose entropy coding','novelty_evidence':'NO'},
 {'SGX_component':'Five decimal scale trials','prior_art':'PseudoDecimals and ALP factor search','novelty_evidence':'NO'},
 {'SGX_component':'Frozen calibration gating','prior_art':'ALP sampling/adaptive selection; general model selection','novelty_evidence':'NOT ESTABLISHED'},
 {'SGX_component':'NumPy vectorized implementation','prior_art':'ALP vectorized execution','novelty_evidence':'NO'},
]
emit_csv('related_work_matrix.csv',PAPERS)
emit_csv('method_component_novelty.csv',COMPONENTS)
inputs={};missing=[];readerrors=[]
for nb in (55,56):
 p=ROOT/('SGX_NB56F_results.zip' if nb==56 else 'SGX_NB55_results.zip')
 if not p.exists():
  missing.append(p.name);inputs[str(nb)]={'status':'MISSING'};continue
 try:inputs[str(nb)]=safe_read_zip(p,'decision.json')
 except Exception as e:readerrors.append({'input':p.name,'error':repr(e)});inputs[str(nb)]={'status':'READ_ERROR'}
# Evaluation criteria cannot claim Q1 readiness from size gain alone.
n55=inputs.get('55',{});n56=inputs.get('56',{})
full=(n55.get('state')=='COMPLETE' and n56.get('state')=='COMPLETE')
positive=(n55.get('robust_gain_vs_oracle_percent') is not None and n55['robust_gain_vs_oracle_percent']>0)
regressions=(n55.get('worst_robust_regression_pct') is not None and n55['worst_robust_regression_pct']>20)
old=(n55.get('total_bytes') or {}).get('old')
rob=(n55.get('total_bytes') or {}).get('robust')
size_vs_old=(100*(old-rob)/old if old and rob else None)
criteria=[
 {'requirement':'Lossless decoding on real and adversarial inputs','status':'PASS_ON_TESTED' if full else 'INCOMPLETE','note':'The notebooks test exact byte reconstruction, not a formal proof.'},
 {'requirement':'External held-out size advantage versus specialized baselines','status':'POSITIVE' if full and positive else 'NOT_PROVEN','note':'UCI families were already explored in NB54; not new independent confirmation.'},
 {'requirement':'Severe per-stream regression avoidance','status':'FAIL' if full and regressions else ('NO_GT20PCT_REGRESSION_OBSERVED' if full else 'INCOMPLETE'),'note':'No future-data guarantee even if passes tested set.'},
 {'requirement':'End-to-end CPU implementation throughput evidence','status':'PYTHON_ONLY' if full else 'INCOMPLETE','note':'Includes frame and selection accounting, but no native specialized codec runs.'},
 {'requirement':'Official ALP/Chimp/Gorilla matched performance','status':'MISSING','note':'Critical for a competitive claim.'},
 {'requirement':'Genuinely new algorithmic mechanism versus ALP/PDE','status':'NOT_ESTABLISHED','note':'Current operations strongly overlap with published work.'},
 {'requirement':'Q1 journal paper readiness','status':'NO','note':'No objective acceptance guarantee; missing novelty and codec comparisons.'}
]
emit_csv('research_gate_criteria.csv',criteria)
status={'state':'INCOMPLETE_INPUTS' if not full else 'COMPLETE_REVIEW','input_status':{k:v.get('state',v.get('status')) for k,v in inputs.items()},'missing_zips':missing,'read_errors':readerrors,'method_novelty':'NOT_ESTABLISHED','strong_ALP_overlap':True,'matched_ALP_Chimp_Gorilla':'MISSING','aggregate_robust_vs_original_policy_pct':size_vs_old,'decision':'REDESIGN_NEW_MECHANISM' if full else 'WAIT_FOR_NB55_NB56_RESULT_ARCHIVES','qualification':'This decision is a preliminary literature gap analysis and must be extended through systematic research review; no patent or novelty clearance is claimed. 2026-10-22 research snapshot.'}
emit_json('novelty_decision.json',status)
print('NB57',json.dumps(status,indent=2))
delivery()

