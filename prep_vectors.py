# prep_vectors.py
import json, numpy as np, pandas as pd

SAMPLE = 100
OUT = 64

meta = pd.read_csv('blocks_manifest.csv')
cpu = meta[meta['target']=='CPU'].reset_index(drop=True)
if len(cpu) < SAMPLE:
    SAMPLE = len(cpu)
cpu = cpu.iloc[:SAMPLE]

# infer N from first tile
b0 = np.load(cpu.iloc[0]['path'])
N = int(np.prod(b0.shape))

# concatenate vectors into one .bin
vecs = np.empty((SAMPLE, N), dtype=np.float32)
for i, path in enumerate(cpu['path'].tolist()):
    v = np.load(path).astype(np.float32).ravel(order='C')  # vector packing (C-order)
    vecs[i] = v
vecs.tofile('vectors.bin')

# fixed W (same for all kernels)
rng = np.random.default_rng(1234)
W = rng.standard_normal((N, OUT), dtype=np.float32)
W.tofile('W.bin')

with open('sizes.json','w') as f:
    json.dump({'N': int(N), 'OUT': int(OUT), 'count': int(SAMPLE)}, f, indent=2)

print('Prepared vectors.bin, W.bin, sizes.json — N=', N, 'OUT=', OUT, 'count=', SAMPLE)
