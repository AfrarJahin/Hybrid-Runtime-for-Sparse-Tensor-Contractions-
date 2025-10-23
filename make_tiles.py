# 02_make_tiles.py
import os, math, csv
import numpy as np
from tqdm import tqdm

NPZ = 'organmnist3d_train.npz'
os.makedirs('blocks', exist_ok=True)

# tiling & routing params
BLOCK = 14          # 28^3 → 8 tiles of 14^3
THRESH = 0.05       # values below → 0 to create sparsity
DENS_THR = 0.30     # density < 0.30 → CPU ; else GPU

data = np.load(NPZ)
imgs = data['imgs']           # [N,1,D,H,W]
N, _, D, H, W = imgs.shape
print('Loaded', imgs.shape)

manifest_path = 'blocks_manifest.csv'
with open(manifest_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['block_id','vol_id','tile_idx','d','h','w','density','target','path'])
    bid = 0
    for n in tqdm(range(N), desc='Tiling volumes'):
        vol = imgs[n,0]  # [D,H,W]
        # normalize to [0,1]
        vmin, vmax = vol.min(), vol.max()
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
        # 3D tiling
        t = 0
        for z0 in range(0, D, BLOCK):
            for y0 in range(0, H, BLOCK):
                for x0 in range(0, W, BLOCK):
                    blk = vol[z0:z0+BLOCK, y0:y0+BLOCK, x0:x0+BLOCK]
                    if blk.shape != (BLOCK,BLOCK,BLOCK):
                        continue
                    b = blk.copy()
                    b[np.abs(b) < THRESH] = 0.0
                    dens = float(np.count_nonzero(b)) / b.size
                    target = 'GPU' if dens >= DENS_THR else 'CPU'
                    path = f'blocks/blk_{bid:07d}.npy'
                    np.save(path, b.astype('float32'))
                    w.writerow([bid, n, t, *b.shape, f'{dens:.6f}', target, path])
                    bid += 1; t += 1

print('Wrote', manifest_path)
