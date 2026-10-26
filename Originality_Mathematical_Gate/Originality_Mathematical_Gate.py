#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# ShiftGuard-X — NB130: Originality & Mathematical Design Gate

**Status:** Research design, not a novel algorithm, and **not another full compressor**. Runs without dataset downloads or Colab-only dependencies. This notebook applies a falsifiable theoretical filter to the proposed *context-conditioned lossless numerical coding* hypothesis, documents known prior art and produces an auditable research decision file.

**Motivation from NB129:** Exponent partitioning failed, while strong standalone baselines (Pcodec and LZMA) make naive predictive transformations inadequate. We will **not** claim superiority from generated models or toy data.

**Current candidate:** A *decoder-synchronized, conditional latent model* that uses previously decoded context to predict conditional probability distributions over exact residual symbols. Any new mechanism must exceed prior context modeling and adaptive entropy coding; changing context features alone is unlikely to be publishable.

**Go/no-go:** Require (1) an explicit mathematically distinct operation; (2) a plausible net-bit advantage after model overhead; (3) a full literature audit covering near-equivalents; and (4) practical benchmarks on real held-out sources. This notebook can rule out weak candidates but **cannot certify novelty**.
"""

"""
## 1. Confirm known prior art

The following references are **starting points**, not an exhaustive patent/literature clearance.

- Afroozeh, Kuffó & Boncz, *ALP: Adaptive Lossless Floating-Point Compression*, SIGMOD 2024: https://duckdb.org/library/alp/
- Pcodec maintained reference: https://github.com/pcodec/pcodec and format specification https://github.com/pcodec/pcodec/blob/main/docs/format.md (modes, deltas, dictionary, bins, tANS)
- Elf*, *Adaptive Encoding Strategies for Lossless Floating-Point Compression*, IEEE IoT J 2025: https://ieeexplore.ieee.org/document/10949160/
- AFC, *An adaptive lossless floating-point compression algorithm in time series database*, Information Sciences 2024: https://www.sciencedirect.com/science/article/pii/S0020025523014329

**Novelty risks:** Conditional entropy coding, context models, adaptive selection, two-part models, and exact residual coding have long histories. A mere hybrid of these components is **not a novelty claim**.
"""

import json, math, hashlib, platform, zipfile, csv, io
from pathlib import Path
from collections import Counter
OUT=Path('/content/SGX_NB130_results') if Path('/content').exists() else Path('SGX_NB130_results')
OUT.mkdir(exist_ok=True)
PRIOR_ART=[
 {'method':'ALP','prior_feature':'Adaptive decimal/integer and front-bit representations','url':'https://duckdb.org/library/alp/'},
 {'method':'Pcodec','prior_feature':'Modes, delta, dictionary, bin+offset/tANS latent coding','url':'https://github.com/pcodec/pcodec/blob/main/docs/format.md'},
 {'method':'Elf*','prior_feature':'Adaptive bit-count and XOR structure coding','url':'https://ieeexplore.ieee.org/document/10949160/'},
 {'method':'AFC','prior_feature':'Data-pattern-based adaptive compressor selection','url':'https://www.sciencedirect.com/science/article/pii/S0020025523014329'}]
with (OUT/'prior_art_matrix.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=PRIOR_ART[0].keys());w.writeheader();w.writerows(PRIOR_ART)
print('Recorded',len(PRIOR_ART),'close precedents')


"""
## 2. Information-theoretic feasibility test

For a discrete residual variable `R` and decoder-available context `C`, the ideal savings from conditioning are bounded by the mutual information `I(R;C) = H(R) − H(R|C)` **under the specified source model**. Metadata, estimation error, and runtime costs reduce actual savings. Conditional entropy by itself is not a novelty or codec performance metric.

Here we compute exact *empirical* joint-frequency entropy on small, explicitly synthetic examples; results are illustration only. In particular, learning a model from the held-out target is not decoder-synchronized coding.
"""

def entropy(xs):
 counts=Counter(xs); n=len(xs)
 return -sum((v/n)*math.log2(v/n) for v in counts.values()) if n else 0.0

def conditional_entropy(residual,context):
 assert len(residual)==len(context)
 n=len(residual)
 by={}
 for r,c in zip(residual,context):by.setdefault(c,[]).append(r)
 return sum(len(v)/n*entropy(v) for v in by.values()) if n else 0.0

# Carefully labeled synthetic sanity controls, NOT compression results.
examples={
 'independent_CONTROL': ([0,0,1,1]*16,[0,1,0,1]*16),
 'deterministic_context_CONTROL':([0,1]*32,[0,1]*32),
 'mostly_correlated_CONTROL':([0,1,0,1,0,1,0,0]*8,[0,1,0,1,0,1,0,1]*8)
}
E=[]
for name,(r,c) in examples.items():
 h=entropy(r);hc=conditional_entropy(r,c);i=h-hc
 assert -1e-10<=i<=h+1e-10
 E.append(dict(control=name,empirical_H=h,empirical_H_given_C=hc,empirical_mutual_information_bits_per_symbol=i,model_overhead_bits='not measured',real_data=False))
for row in E:print(row)
assert abs(E[0]['empirical_mutual_information_bits_per_symbol'])<1e-12
assert abs(E[1]['empirical_H_given_C'])<1e-12


"""
## 3. Mathematics: A necessary but insufficient bit-cost condition

For `N` encoded symbols, an idealized upper limit on conditional-model advantage is `N × I(R; C)` bits compared with **ideal marginal entropy coding**, before paying model metadata and approximation overhead. If context has to be transmitted, its cost must also be charged. The actual advantage relative to modern compressors must be measured from complete bitstreams, not this bound.

The following sensitivity table illustrates how a nontrivial context association can still be erased by model overhead. These are **assumptions**, not observed gains.
"""

S=[]
for n in (256,4096,65536):
 for mi in (0.005,0.02,0.1,0.4):
  for overhead in (64,1024,8192):
   available=n*mi-overhead
   S.append(dict(n_symbols=n,assumed_information_bits_per_symbol=mi,assumed_model_overhead_bits=overhead,ideal_net_bit_budget=available,necessary_gate_positive=available>0))
with (OUT/'theoretical_sensitivity.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=S[0].keys());w.writeheader();w.writerows(S)
print('Example: 4096 values, 0.02 hypothetical bits per symbol, 1024 overhead bits ->',4096*.02-1024,'ideal net bits')


"""
## 4. Pre-registered novelty audit

**Candidate A: Plain decoder-synchronized context-conditioned entropy coding.** This is a well-established family of techniques; reject it as an original contribution without a genuinely new mathematical coding construct.

**Candidate B: Learned residual-mask/topology templates.** Substantial overlap risk with XOR compressors, sparse coding and context prediction; reject until a specific non-equivalent coding representation and its rigorous comparison is found.

**Candidate C: Representation-specific predictive causal model with formal complexity guarantee.** This remains a *research question*, not a mechanism. Advancing requires a theorem or genuinely new representation with clear assumptions, plus a prior-art survey in source papers. No superiority should be claimed.

**Research decision:** Do not build NB131 as a compressor based on any of A–C yet. The next legitimate action is a source-grounded mathematical proposal + external independent review of originality, followed by targeted held-out benchmarking.
"""

status=[
 {'candidate':'A context-conditioned entropy coding','originality_risk':'high','specific_new_operation':False,'recommendation':'REJECT AS NOVELTY'},
 {'candidate':'B residual topology templates','originality_risk':'high','specific_new_operation':False,'recommendation':'HOLD / REVIEW PRIOR ART'},
 {'candidate':'C causal numerical representation with proof','originality_risk':'unknown','specific_new_operation':False,'recommendation':'SPECIFY FORMALLY BEFORE CODE'}]
with (OUT/'novelty_gate.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=status[0].keys());w.writeheader();w.writerows(status)
decision={'notebook':'NB130','phase':'originality and mathematical design','experimental_real_data_codec_results':False,'novelty_established':False,'new_implementable_algorithm_specified':False,'go_to_NB131_codec':False,'reason':'No candidate currently passes technical-distinctness and evidence gates','required_next_evidence':['Define one new operation and a bit-exact inverse','Find and contrast nearest prior methods','Give bound or formal analysis with explicit assumptions','Measure complete streams against Pcodec/LZMA/ALP on independent data']}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
(OUT/'README.md').write_text('ShiftGuard-X NB130 originality gate. All entropy examples and overhead sensitivity are SYNTHETIC/HYPOTHETICAL, not codec benchmark gains. Prior-art matrix is preliminary. Novelty unestablished.\n')
archive=OUT.parent/'SGX_NB130_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for file in OUT.iterdir():z.write(file,arcname=file.name)
print(json.dumps(decision,indent=2));print('Created',archive)
try:
 from google.colab import files
 files.download(str(archive))
except ImportError:pass

