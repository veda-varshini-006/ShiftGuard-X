# ShiftGuard-X NB87 — Scientific Contribution Screening Plan

## Research status

**NB86 was rejected:** in the uploaded Colab rerun, the new binary-precision stratification codec was selected on 0/117 familiar development streams. The project has not yet established a novel, competitive lossless compressor or a Q1-ready manuscript.

## The specific question for NB87

Do **real chronologically ordered numerical sensor streams** have temporal structure that is not well captured by a strong general-purpose or byte-shuffled compressor, and can this structure be isolated from marginal compressibility with a row-order permutation control?

NB87 is an *opportunity gate*, not a proposed original coding mechanism. Its XOR preprocessing and codec selection are known techniques. A positive result could narrow the workload on which to seek a genuinely distinct decoder-synchronous encoding invariant. A negative result should end this path rather than trigger another broad search across arbitrary codecs.

## Data and controls

- UCI EEG Eye State (264): sequential 14-channel EEG. Use up to 8 numerical channels.
- UCI Beijing Multi-Site Air Quality (501): hourly sensors, use 2 stations, up to 7 columns each.
- UCI Electrical Grid Stability Simulated Data (471): simulated non-temporal grid configuration; it is an **explicit negative-control workload** and cannot count as a third real temporal family.

Every downloaded source is hashed. The notebook records missing downloads and **never substitutes synthetic values** when running real-data mode. A separate `SGX87_SMOKE=1` mode is exclusively for workflow tests.

## Prespecified decision

A workload advances to mechanism research only if **both** real chronological families have >=5% aggregate net byte saving from a calibration-selected temporal codec versus a calibration-selected strong ordinary codec, plus >=2 percentage points larger advantage in original than shuffled time order. Any worst-stream regression below -10%, source failure, codec unavailability or bitwise mismatch blocks advancement. The simulated grid control must not display the same apparent temporal signature.

**Do not mistake a pass for originality.** No original method is implemented in NB87. The closest prior-art tests would still be Gorilla, Chimp, FPC, Elf/SElf, and ALP. A Q1 submission additionally requires a novel mechanism, full comparable file sizes, native runtime, multiple fresh dataset families and substantive statistical support.

## What happens after NB87

- If the gate **passes**: define one specific decoder-available temporal state variable; derive a total-bit cost inequality; implement a small reversible mechanism with a strong nearest-prior-art baseline on the exact same signals.
- If the gate **fails**: stop hunting marginal gains in generic sensor float64 streams; consider focusing on an application-constrained representation problem or a rigorous reproducibility/benchmark methodology paper **only if** it has an independently novel question.

## Verified references

- Afroozeh et al., ALP (SIGMOD 2024): https://ir.cwi.nl/pub/33334
- Elf* (IEEE Internet of Things Journal, 2025): https://ieeexplore.ieee.org/document/10949160/
- UCI EEG: https://archive.ics.uci.edu/dataset/264/eeg+eye+state
- UCI Beijing: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
- UCI simulated grid: https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data
