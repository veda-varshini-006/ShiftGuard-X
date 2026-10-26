#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShiftGuard-X Research Codebase
"""

"""
# Independent_Domain_Arithmetic_Error_Gate

**Research use only.** Run all cells in Google Colab. The last cell exports and downloads the results ZIP. Unproven novelty is never inferred from a size comparison.
"""

"""
**Independent domain:** UCI Bike Sharing daily/hourly counts. This notebook constructs a transparent arithmetic reconstruction surrogate from real count values; it does **not** claim to be a Virtual compressor output or a natural Virtual rounding-error workload. If no one-ULP mismatches exist, report NOT_APPLICABLE rather than fabricating data. Frozen January model is evaluated unchanged.
"""

# !pip -q install zstandard
import io, os, json, csv, zipfile, hashlib, pathlib, zlib, lzma, bz2, struct
import numpy as np
from google.colab import files
import zstandard as zstd
NB = 109
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


zs=upload_archives(['100'])
lut64=np.array(read_json(zs['100'],'frozen_model.json')['bits'],dtype=np.uint8)
assert len(lut64)==64
import urllib.request
url='https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
try:
    raw=urllib.request.urlopen(url,timeout=50).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zz:
        content=zz.read('hour.csv').decode('utf-8')
except Exception as ex:
    save_json('decision.json',{'status':'DATA_DOWNLOAD_FAILED','reason':str(ex),'novelty_established':False})
    raise
records=list(csv.DictReader(io.StringIO(content)))
# cnt=casual+registered exactly as integer. Compare same identity evaluated in float64.
parts=np.array([[float(r['casual']),float(r['registered']),float(r['cnt'])] for r in records],dtype=np.float64)
a,b,expected=parts.T
reconstructed=(a+b).astype(np.float64)
orig_bits=expected.view(np.uint64);got_bits=reconstructed.view(np.uint64)
mask=orig_bits!=got_bits
number=int(mask.sum())
rows=[dict(domain='UCI_Bike_Hourly',samples=len(parts),mismatches=number,bit_exact_identity=not number)]
if number:
    signs=signed_ulp_error(orig_bits[mask],got_bits[mask])
    if signs is not None:
        c=((got_bits[mask]&15)|((got_bits[mask]>>np.uint64(48)&3)<<4)).astype(np.uint8)
        residual=(signs>0).astype(np.uint8)^lut64[c]
        rows[0].update(frozen_model_residual_bits=len(residual),model_unchanged=True)
    else:rows[0]['reason']='multi-ULP errors; model inapplicable'
else:rows[0]['reason']='Integer count identity is exact in float64; no natural repair errors here'
save_csv('domain_summary.csv',rows)
save_json('decision.json',{'status':'COMPLETE','domain':'UCI Bike Sharing counts','samples':len(parts),'rounding_error_count':number,'external_domain_repair_transfer_passed':False,'reason':rows[0].get('reason','No complete decoder benchmark'),'original_source_sha256':hashlib.sha256(raw).hexdigest(),'original_values_bits_verified':bool(np.array_equal(orig_bits,got_bits)),'novelty_established':False})
print(rows[0])


archive=pathlib.Path(f'/content/SGX_NB{NB}_results.zip')
result_files=sorted(p for p in OUT.rglob('*') if p.is_file())
if not result_files:raise RuntimeError('No output files to export')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in result_files:z.write(p,p.relative_to(OUT))
with zipfile.ZipFile(archive) as z:
    if z.testzip() is not None:raise RuntimeError('ZIP verification failed')
print('Created:',archive,'Files:',len(result_files),'Bytes:',archive.stat().st_size)
files.download(str(archive))

