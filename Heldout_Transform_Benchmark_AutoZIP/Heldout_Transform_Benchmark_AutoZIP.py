#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# SGX NB116 — Held-out fair codec benchmark
**Goal:** Compare full serialized frame sizes on identical 60k held-out rows with frozen calibration policies, including metadata and Zstd; compare a native LZMA block and a raw+Zstd file. If triangular does not beat strong baselines, reject this compression hypothesis. This is **not** a benchmark against all published algorithms (zfp/Corra remain external dependencies).
"""

# !pip -q install zstandard numpy pandas


OUTPUT='SGX_NB116_outputs'



import os, io, json, zipfile, struct, hashlib, lzma
from pathlib import Path
import numpy as np, pandas as pd, zstandard as zstd
from google.colab import files
ROOT=Path('/content'); OUT=ROOT/OUTPUT; OUT.mkdir(exist_ok=True)
print('Upload SGX_NB96_results.zip (original, decoded bit patterns)')
upload=files.upload()
match=[v for k,v in upload.items() if k.startswith('SGX_NB96') and k.endswith('.zip')]
assert len(match)==1, 'Upload exactly one SGX_NB96_results.zip'
with zipfile.ZipFile(io.BytesIO(match[0])) as z:
    assert z.testzip() is None
    evidence=np.load(io.BytesIO(z.read('bitwise_evidence.npz')))
    original=evidence['original'].copy(); decoded=evidence['decoded'].copy()
assert original.dtype==np.uint64 and original.ndim==2 and original.shape[0]>=2
assert original.shape==decoded.shape
# Original uint64 words are ground truth for exact numerical encoding.
words=original
C,N=words.shape
cut=min(30000,N//3)
train=words[:,:cut].copy(); test=words[:,cut:].copy()
assert test.shape[1]>0
source_sha=hashlib.sha256(original.tobytes()).hexdigest()
print('data shape',words.shape,'train',train.shape,'test',test.shape,'sha256',source_sha)

def shuffle(b):
    a=np.frombuffer(b,dtype=np.uint8)
    assert len(a)%8==0
    return a.reshape(-1,8).T.copy().tobytes()
def unshuffle(b):
    a=np.frombuffer(b,dtype=np.uint8)
    assert len(a)%8==0
    return a.reshape(8,-1).T.copy().tobytes()
def u64bytes(v): return np.asarray(v,dtype='<u8').tobytes()
def frombytes(b): return np.frombuffer(b,dtype='<u8').copy()
# These are established predictive transforms. They are NOT, by themselves, novel.
def transform(v, ref, mode):
    v=np.asarray(v,dtype=np.uint64)
    if mode=='raw': return v.copy()
    if mode=='time_xor':
        out=v.copy();out[1:]=v[1:]^v[:-1];return out
    if mode=='time_sub':
        out=v.copy();out[1:]=v[1:]-v[:-1];return out
    if mode=='cross_xor':return v^ref
    if mode=='cross_sub':return v-ref
    raise ValueError(mode)
def inverse(t,ref,mode):
    t=np.asarray(t,dtype=np.uint64)
    if mode=='raw':return t.copy()
    if mode=='time_xor':
        out=np.bitwise_xor.accumulate(t);return out
    if mode=='time_sub':
        out=np.add.accumulate(t,dtype=np.uint64);return out
    if mode=='cross_xor':return t^ref
    if mode=='cross_sub':return t+ref
    raise ValueError(mode)
MODES=['raw','time_xor','time_sub','cross_xor','cross_sub']
COMPRESSOR=zstd.ZstdCompressor(level=6)
DECOMPRESSOR=zstd.ZstdDecompressor()
def compress_payload(v, shuffled):
    raw=u64bytes(v)
    return COMPRESSOR.compress(shuffle(raw) if shuffled else raw)
def decompress_payload(p, shuffled):
    raw=DECOMPRESSOR.decompress(p)
    return frombytes(unshuffle(raw) if shuffled else raw)
def trial_size(v,ref,mode,shuf): return len(compress_payload(transform(v,ref,mode),shuf))
# Serialized frame contains a magic marker, full JSON header, and every payload.
def encode_frame(arr,policies):
    payloads=[];fields=[]
    for c,(mode,shuf) in enumerate(policies):
        ref=arr[c-1] if c else None
        assert c>0 or mode not in ('cross_xor','cross_sub')
        p=compress_payload(transform(arr[c],ref,mode),shuf)
        payloads.append(p);fields.append({'mode':mode,'shuffle':bool(shuf),'length':len(p)})
    header=json.dumps({'nrows':int(arr.shape[1]),'ncols':int(arr.shape[0]),'fields':fields},separators=(',',':'),sort_keys=True).encode()
    return b'SGX1'+struct.pack('<I',len(header))+header+b''.join(payloads)
def decode_frame(frame):
    assert frame[:4]==b'SGX1';m=struct.unpack('<I',frame[4:8])[0]
    meta=json.loads(frame[8:8+m]);offset=8+m;ans=[]
    for field in meta['fields']:
        p=frame[offset:offset+field['length']];offset+=field['length']
        transformed=decompress_payload(p,field['shuffle'])
        ref=ans[-1] if ans else None
        ans.append(inverse(transformed,ref,field['mode']))
    assert offset==len(frame)
    out=np.stack(ans)
    assert out.shape==(meta['ncols'],meta['nrows'])
    return out
# Calibrate ONLY with training rows; encode/decode held-out without retuning.
def calibrate(arr,allowed):
    policies=[];rows=[]
    for c in range(arr.shape[0]):
        choices=[(trial_size(arr[c],arr[c-1] if c else None,mode,shuf),mode,shuf) for mode in allowed if c or not mode.startswith('cross_') for shuf in [False,True]]
        size,mode,shuf=min(choices)
        policies.append((mode,shuf));rows.append({'column':c,'calibration_bytes':size,'mode':mode,'shuffle':shuf})
    return policies,rows
# Special word safety: all finite, infinities, both signed zero, arbitrary NaN payload bits.
special=np.array([[0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000123,0x7ff0000000000001,0xffffffffffffffff],
                  [1,2,3,4,5,6,7]],dtype=np.uint64)
for mode in MODES:
    for j in range(2):
        if j==0 and mode.startswith('cross_'):continue
        ref=special[j-1] if j else None
        assert np.array_equal(inverse(transform(special[j],ref,mode),ref,mode),special[j])
frame=encode_frame(special,[('raw',True),('cross_sub',False)])
assert np.array_equal(decode_frame(frame),special)
print('Edge-case exact-word round trip passed')



policies={
  'raw_zstd':calibrate(train,['raw'])[0],
  'temporal_zstd':calibrate(train,['raw','time_xor','time_sub'])[0],
  'triangular_zstd':calibrate(train,MODES)[0]
}
rows=[]
for name,policy in policies.items():
    frame=encode_frame(test,policy)
    ok=np.array_equal(decode_frame(frame),test)
    assert ok
    rows.append({'method':name,'bytes':len(frame),'bit_exact':ok,'full_metadata_included':True})
# LZMA and raw Zstd baselines include all original 64-bit words; a fixed magic+shape header is counted.
raw=np.ascontiguousarray(test,dtype='<u8').tobytes()
header=b'SGX0'+struct.pack('<II',test.shape[0],test.shape[1])
lz=header+lzma.compress(raw,preset=6)
assert lzma.decompress(lz[len(header):])==raw
rows.append({'method':'raw_lzma','bytes':len(lz),'bit_exact':True,'full_metadata_included':True})
rawz=header+COMPRESSOR.compress(raw)
assert DECOMPRESSOR.decompress(rawz[len(header):])==raw
rows.append({'method':'raw_zstd_contiguous','bytes':len(rawz),'bit_exact':True,'full_metadata_included':True})
df=pd.DataFrame(rows).sort_values('bytes');df.to_csv(OUT/'benchmark.csv',index=False)
leading=df.iloc[0]['method']; tri=int(df[df.method=='triangular_zstd'].iloc[0]['bytes']); strongest_other=int(df[df.method!='triangular_zstd']['bytes'].min())
report={'status':'COMPLETE','source_sha256':source_sha,'rows_train':cut,'rows_test':int(test.shape[1]),'best_method':leading,'triangular_bytes':tri,'best_competing_bytes':strongest_other,'triangular_beats_tested':bool(tri<strongest_other),'all_bit_exact':True,'algorithmic_novelty_established':False,'published_SOTA_checked':False,'warning':'A result on NB96 only cannot demonstrate independent domain generality or novelty.'}
(OUT/'decision.json').write_text(json.dumps(report,indent=2))
print(df.to_string(index=False));print(json.dumps(report,indent=2))



import zipfile
out_files=sorted(p for p in OUT.rglob('*') if p.is_file())
assert out_files, 'No output files generated'
archive=ROOT/(OUTPUT.replace('_outputs','_results.zip'))
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in out_files:z.write(p,arcname=p.relative_to(OUT))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print('Created',archive,'bytes',archive.stat().st_size)
files.download(str(archive))

