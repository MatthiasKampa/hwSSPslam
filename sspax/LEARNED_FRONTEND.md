# Learned FPGA-fittable front-end + semantic SSP map (2026-07-15)

A learned, quantizable pre-processing front-end for **both** modalities, feeding
a **binary-descriptor binding** into the bounded SSP map so the map is
**queryable by object** ("highlight the chairs"). Everything imports the frozen
core (rule 1); training is on deterministic synthetic geometry (lidar) and an
external pretraining set (vision); GT poses only align contrastive targets, never
seed the encoder (rule 2). Trains on GPU — the cuDNN-free im2col convs in
`sspax/nnconv.py` sidestep this box's cuDNN/jaxlib mismatch.

## Unifying architecture

    image  ->  tiny CNN  ->  binary descriptor  ┐
                                                 ├─(x) position SSP ─► + ─► bounded map (one D-vector)
    scan   ->  BEV + tiny CNN saliency ─► keypts ┘        (bind)      (bundle)

- Both modalities pass through a small CNN; the **binary** output is the FPGA
  substrate (BNN XNOR-popcount) **and** the VSA binding key.
- A feature is written by binding each set bit's role hypervector with the
  feature POSITION (an SSP phasor) and bundling. **Significance = bit count**:
  a position bound under more bits reads out proportionally stronger.
- The map stays a single fixed-size D-vector (D=360, the swept winner lattice
  `ring_arc_const_s0.5_r6` + oct6 ladder). No history stored.

## 1. Learned lidar saliency — `sspax/learn_lidar.py`

BEV-rasterize the scan (48×48×3: log-occupancy, max/mean height) → a **3-conv,
826-param** CNN scores per-cell saliency on a **2×-downsampled** 24×24 map →
threshold to the top-k salient cells → encode those (weighted) into the SSP. The
SSP encode `Σ wᵢ exp(iW·pᵢ)` is differentiable, so a **contrastive place-rec
loss backprops through the encoder into the CNN** (end-to-end for SLAM).

Result (tight k=32-feature budget, interior clutter the net must reject,
rotation-invariant matching):

| features | place-rec AUC |
|----------|---------------|
| uniform random k=32 | 0.707 |
| **learned saliency k=32** | **0.940** (**+0.232**) |

826 params (0.8 KB int8, 103 B as a BNN). The net learns to spend a tiny sparse
budget on repeatable structure and drop the moving clutter — the "better
trackable features" the directive asked for, at a sparser encode. Trains on GPU
(im2col convs in `sspax/nnconv.py` sidestep the box's cuDNN mismatch): 400 steps
in ~10 s.

## 2. Semantic binary-binding map — `sspax/semantic.py`

Each object writes `Σ_{i∈bits} ROLES[i] ⊗ encode(xyz)` into the bundle. Query a
class (a bit-set, e.g. "chair") by unbinding its roles and decoding the spatial
density; peak height ∝ |query ∩ feature bits|, so matching classes light up and
other classes cross-talk only through random bit overlap.

- **Query "chair"** (16 objects, D=360, k=12 bits): precision **1.00**, recall
  **1.00**, position error **0.04 m**, **4.0×** contrast (chair 14.4 vs
  non-chair 3.6).
- **Significance = committed bits** (query 'chair'): readout 5.4 / 9.7 / 12.9 /
  14.9 for 3 / 6 / 9 / 12 committed bits — the map grades importance by how many
  bits a feature sets.
- **Capacity** (chairs recalled vs objects bundled into ONE D=360 vector):
  0.75 @ 10 obj, 0.38 @ 20, 0.03 @ 40 — a bounded map saturates ~15–20 objects;
  more bits/object *raises* SNR (recall 0.33→0.50 as k 6→20 at 20 obj). Capacity
  scales with D; the query cost is O(D) regardless of object count.

## 3. Learned vision — `sspax/vision/tinycnn.py`

Tiny CNN (**34 k params**, 33 KB int8 / **4.2 KB BNN**) pretrained on CIFAR-100
(which has fine labels chair/couch/table/bed/wardrobe → a furniture detector),
with a **64-bit binary descriptor head** that plugs straight into §2. Built and
FPGA-costed; pretraining runs `JAX_PLATFORMS=cpu python3 -m sspax.vision.tinycnn
train` once the (slow) CIFAR download completes.

## FPGA cost (`python3 -m sspax.vision.tinycnn cost`)

| front-end | params | int8 | BNN | output |
|-----------|--------|------|-----|--------|
| lidar SaliencyNet | 826 | 0.8 KB | **103 B** | per-cell saliency, 24×24 |
| vision TinyCNN | 34 244 | 33 KB | **4.2 KB** | 64-bit descriptor |
| descriptor bind | — | — | — | O(D) phase mult, D=360 |

Both fit the ECP5 fusion target; the lidar net fits even an ice40. The map is
one bounded D=360 vector; object queries are O(D) phase ops — no per-object
storage, consistent with the project's history-free thesis.

## Honest status

- Learned lidar saliency + semantic binding + significance + capacity + FPGA
  cost: **built and verified** (numbers above, deterministic).
- Vision CNN: **built + FPGA-costed**; CIFAR pretraining pending the slow
  external download (auto-runs on completion).
- Fusion caveat carried from the sweep: geometric channels are redundant; the
  learned **appearance** descriptor (vision) is the independent cue that makes
  cam+lidar binding non-redundant — the reason to pretrain vision on real images.
