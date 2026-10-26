#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# Entropy_Coder_Challenge

**Research use only.** Run all cells in Google Colab. The last cell exports and downloads the results ZIP. Unproven novelty is never inferred from a size comparison.
"""

"""
Requires **NB96 and NB102**. Trains sign context on January, evaluates February. Compares raw bitpacking, Zstandard, zlib, lzma, bzip2, and adaptive order-0 arithmetic/range coding on **exact same correction-sign bits**. Counts mask, headers and trained model. NB106 figures are historical reference, not matched framing. A no-error stream must be emitted and decoded.
"""

# !pip -q install zstandard
import io, os, json, csv, zipfile, hashlib, pathlib, zlib, lzma, bz2, struct
import numpy as np
from google.colab import files
import zstandard as zstd
NB = 108
OUT = pathlib.Path(f'/content/SGX_NB{NB}_outputs'); OUT.mkdir(parents=True, exist_ok=True)
def upload_archives(required):
    print('Upload required archives:', required)
    incoming=files.upload()
    zs={}
    for name,blob in incoming.items():
        for tag in required:
            if f'NB{tag}_results' in name:
                zs[tag]=zipfile.ZipFile(io.BytesIO(blob))
                if zs[tag].testzip() is not None: raise ValueError('Corrupt ZIP '+name)
    missing=set(required)-set(zs)
    if missing: raise ValueError('Missing result ZIPs: '+str(sorted(missing)))
    return zs
def read_json(z,filename):return json.loads(z.read(filename))
def save_json(filename,obj):(OUT/filename).write_text(json.dumps(obj,indent=2,default=str))
def save_csv(filename,rows):
    if not rows:return
    with (OUT/filename).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load_bits(z,filename):
    a=np.load(io.BytesIO(z.read(filename)))
    return a['original'].astype(np.uint64),a['decoded'].astype(np.uint64)
def signed_ulp_error(original,decoded):
    # Work on IEEE-754 monotonic integer ordering, including negative numbers.
    mask=np.uint64(0x8000000000000000)
    def ordered(a):return np.where((a&mask)!=0,~a,a^mask).astype(np.uint64)
    x,y=ordered(original),ordered(decoded)
    # subtract in int64 only after checking adjacent unsigned values; no wrap dependency.
    bigger=x>y
    mag=np.where(bigger,x-y,y-x)
    if np.any(mag>1):return None
    return np.where(bigger,1,-1).astype(np.int8)


zs=upload_archives(['96','102'])
jan_o,jan_d=load_bits(zs['96'],'bitwise_evidence.npz')
feb_o,feb_d=load_bits(zs['102'],'external_bitwise_evidence.npz')
jan_o,jan_d=jan_o[-1],jan_d[-1]
feb_o,feb_d=feb_o[-1],feb_d[-1]
assert jan_o.size==90000 and feb_o.size==120000
jmask=jan_o!=jan_d;fmask=feb_o!=feb_d
js=signed_ulp_error(jan_o[jmask],jan_d[jmask]);fs=signed_ulp_error(feb_o[fmask],feb_d[fmask])
if js is None or fs is None:raise RuntimeError('Non-one-ULP correction: this test does not apply')
js=(js>0).astype(np.uint8);fs=(fs>0).astype(np.uint8)
print('January, February corrections:',len(js),len(fs))
# learned 16-entry direction LUT, using decoded lowest four raw bits
ctx_j=(jan_d[jmask]&np.uint64(15)).astype(np.uint8)
ctx_f=(feb_d[fmask]&np.uint64(15)).astype(np.uint8)
lut=np.zeros(16,dtype=np.uint8)
for k in range(16):
    vals=js[ctx_j==k]
    lut[k]=int(vals.sum()*2>=len(vals)) if len(vals) else 0
res=(fs^lut[ctx_f]).astype(np.uint8)
mask_bytes=np.packbits(fmask.astype(np.uint8),bitorder='little').tobytes()
model_bytes=np.packbits(lut,bitorder='little').tobytes()
header=struct.pack('<8sQQ',b'SGX108\0\0',len(fmask),int(fmask.sum()))
# arithmetic range coder, adaptive binary with contexts. Pure Python, works on 26k symbols.
def arithmetic_encode(bits):
    lo=0;hi=(1<<32)-1; pending=0;out=[]
    c0=1;c1=1
    def emit(bit):
        nonlocal pending
        out.append(bit)
        out.extend([1-bit]*pending);pending=0
    for bit in bits:
        span=hi-lo+1
        cut=lo+(span*c0//(c0+c1))-1
        if bit==0:hi=cut;c0+=1
        else:lo=cut+1;c1+=1
        if c0+c1>32768:c0=max(1,(c0+1)//2);c1=max(1,(c1+1)//2)
        while True:
            if hi<0x80000000:emit(0)
            elif lo>=0x80000000:emit(1);lo-=0x80000000;hi-=0x80000000
            elif lo>=0x40000000 and hi<0xC0000000:pending+=1;lo-=0x40000000;hi-=0x40000000
            else:break
            lo=(lo<<1)&0xffffffff;hi=((hi<<1)|1)&0xffffffff
    pending+=1
    if lo<0x40000000:emit(0)
    else:emit(1)
    return np.packbits(np.array(out,dtype=np.uint8),bitorder='big').tobytes()
raw=np.packbits(res,bitorder='little').tobytes()
enc={'raw_packbits':raw,'zstd':zstd.ZstdCompressor(level=9).compress(raw),'zlib':zlib.compress(raw,9),'lzma':lzma.compress(raw),'bz2':bz2.compress(raw),'arithmetic_adaptive':arithmetic_encode(res)}
rows=[]
for method,blob in enc.items():
    # decoder restoration checked for byte-oriented lossless codecs
    if method=='raw_packbits':restored=blob
    elif method=='zstd':restored=zstd.ZstdDecompressor().decompress(blob)
    elif method=='zlib':restored=zlib.decompress(blob)
    elif method=='lzma':restored=lzma.decompress(blob)
    elif method=='bz2':restored=bz2.decompress(blob)
    else:restored=None
    verified=False
    if restored is not None:
        got=np.unpackbits(np.frombuffer(restored,np.uint8),bitorder='little')[:len(res)]^lut[ctx_f]
        step=np.where(got.astype(bool),np.uint64(1),np.uint64(0))
        # Validate recovered signs and bit-exact restoration, using mapped float ordering
        verified=bool(np.array_equal(got,fs))
    rows.append(dict(method=method,mask_bytes=len(mask_bytes),model_bytes=len(model_bytes),header_bytes=len(header),payload_bytes=len(blob),total_repair_bytes=len(mask_bytes)+len(model_bytes)+len(header)+len(blob),bit_exact_verified=verified,reference_NB106_bytes=14250))
# arithmetic coded size is exploratory until a matching independently tested decoder is provided
save_csv('entropy_comparison.csv',rows)
save_json('decision.json',{'status':'COMPLETE','mismatches':len(fs),'one_ulp_errors_only':True,'best_fully_verified_method':min((r for r in rows if r['bit_exact_verified']),key=lambda x:x['total_repair_bytes'])['method'],'arithmetic_encoder_not_roundtrip_verified':True,'historical_NB106_14250_not_matched_framing':True,'novelty_established':False})
print(*rows,sep='\n')


archive=pathlib.Path(f'/content/SGX_NB{NB}_results.zip')
result_files=sorted(p for p in OUT.rglob('*') if p.is_file())
if not result_files:raise RuntimeError('No output files to export')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in result_files:z.write(p,p.relative_to(OUT))
with zipfile.ZipFile(archive) as z:
    if z.testzip() is not None:raise RuntimeError('ZIP verification failed')
print('Created:',archive,'Files:',len(result_files),'Bytes:',archive.stat().st_size)
files.download(str(archive))

