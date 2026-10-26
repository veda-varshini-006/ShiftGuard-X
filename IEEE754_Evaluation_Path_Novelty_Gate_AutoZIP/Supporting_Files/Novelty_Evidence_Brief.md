# ShiftGuard-X: NB117 novelty evidence brief

Date: 25 October 2026 | Status: NOT NOVELTY-PROVEN

## Evidence and research rationale

NB116 showed that the previously proposed two-parent reversible transforms failed to outperform strong basic baselines on real UCI Bike and Beijing data. Continuing to change the number of integer-domain parents would not address the fundamental novelty problem.

Relevant prior research:
- ALP (SIGMOD 2024): https://duckdb.org/library/alp/
- Corra (2024): https://arxiv.org/abs/2403.17229
- C3 correlation-aware framework: https://github.com/cwida/C3
- Isenburg, Lindstrom & Snoeyink (2005), exact predictive floating-point residual compression: https://www.sciencedirect.com/science/article/pii/S0010448504002544
- Fout & Ma (2012), adaptive switched prediction for lossless floating-point data: https://pubmed.ncbi.nlm.nih.gov/26357137/
- PVLDB 2025 comparative evaluation: https://www.vldb.org/pvldb/vol18/p4396-hishida.pdf

## Narrow new hypothesis (not a novelty claim)

For decoder-known IEEE-754 binary64 operands a,b,c, consider distinct candidate evaluations:

    p0 = round64(round64(a+b)+c)
    p1 = round64(a+round64(b+c))

For target word Y, the encoder emits a selector 0 if Y is bit-identical to p0, 1 if it is bit-identical to p1 (and not p0), otherwise a raw 64-bit escape. These candidates, the selectors, side columns, and payloads define a complete bit-exact code. The scheme is portable only with a normative binary64 operation and rounding specification; ordinary NumPy evaluation under different hardware/fast-math assumptions is not a complete portability guarantee. The novelty claim is not proven because multi-predictor selection and residual escapes have prior art.

## NB117 fully decoded files, bytes include four columns and format headers

| Source | Whole Zstandard | Raw target + side | One-path XOR + side | Dual-path + side | Evaluation path disagreements |
|---|---:|---:|---:|---:|---:|
| Real UCI Bike identity (17,379 rows) | 87,793 | 87,578 | **55,425** | 55,431 | **0** |
| Engineered roundpath mixture (90,000 rows) | 2,114,408 | 2,114,420 | 1,737,620 | **1,549,970** | 90,000 |
| IEEE special values (8 rows) | **352** | 365 | 372 | 386 | 0 |

All 12 complete containers passed bit-exact round-trip comparison on the local machine. The stored real UCI Bike numerical columns are a previously validated parsed IEEE-754 matrix, *not* the original source CSV bytes. The third source operand on Bike is a zero column that makes the count identity a three-term expression.

The dual-path method loses to the one-path method on real Bike data by 6 bytes, exactly because the two evaluated expressions are identical for these counts. The synthetic mixture is constructed directly from two candidate outputs and thus provides no independent evidence of a real-world gain.

### Decision

**REJECT novelty claim and withhold algorithm-improvement claim.** Continue this hypothesis only if independent real arithmetic-derived fields exhibit reproducible evaluation-path diversity and dual-path selection wins against appropriate multi-predictor and complete-file baselines. A distinct theoretical or algorithmic property is still required.

The current proof-by-cases of bit-exactness is useful implementation correctness, **not itself new**. Do not promise journal acceptance based on these results.

Notebook: SGX_NB117_IEEE754_Evaluation_Path_Novelty_Gate_NO_UPLOAD_AutoZIP.ipynb
Local results: SGX_NB117_results.zip
