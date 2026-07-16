"""P3 round 4 — the DEMOTION's CONSTRUCTIVE half: the SURFACES-TIER + QBE per-
segment semantic band, the end-to-end recipe of what actually SHIPS after the
fine-object label head demoted (msg efc5f0e/round-4). One bounded D-vector per
5-keyframe segment carries TWO bit sources bound at cell 3D positions:
  1. SURFACE class bits — the 5-class U-Net output (wall/floor/cabinet/bed/other;
     pixacc 0.858 banked); query "floor"/"wall" -> localise the surface.
  2. DESC bits — the tracking descriptor's 32 sign bits (0.95 retrieval, gated
     cross-box); label-FREE QUERY-BY-EXAMPLE: bind an example cell's desc bits,
     retrieve matching cells from a held-out view.
Band = geomspace(0.25, 2, 6) (the P4 deploy fine-band, lidar coherence floor),
D=360, stored at 4-bit polar quant (q_polar_np, the FPGA arithmetic). Prints
surface-query recall, QBE retrieval, and bytes/segment.

Here the surface labels + desc bits are a controlled synthetic segment (a room:
floor + wall cells + object clusters, each cluster a distinct desc code) so the
recipe is reproducible; in deploy they are the exported vision head's seg-argmax
+ desc bits (headio v2), positions from lidar-projected depth. Anti-oracle:
synthetic, GT surface class + cluster id score only.

  python3 -m sspax.surfaces_tier
"""
import numpy as np

from sspax.semantic_golden import W_of_np, dirs_ring_np, q_polar_np

M = 256
K = 12
SURF = ["wall", "floor", "cabinet", "bed", "other"]
BAND = list(np.geomspace(0.25, 2.0, 6))          # P4 deploy fine-band


def _codes(names, seed):
    rng = np.random.default_rng(seed)
    return {n: np.sort(rng.choice(M, K, replace=False)) for n in names}


def _segment(seed, n_obj=6):
    """a room segment: floor grid + wall strip + n_obj object clusters. Each cell
    -> (surface_class, cluster_id, xyz). Object cells belong to a distinct cluster
    (a shared desc code -> QBE can retrieve the cluster)."""
    rng = np.random.default_rng(seed)
    cells = []
    for _ in range(24):                          # floor
        cells.append(("floor", -1, np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), 0.0])))
    for _ in range(16):                          # wall
        cells.append(("wall", -2, np.array([rng.uniform(-3, 3), 3.0, rng.uniform(0.5, 2.5)])))
    for c in range(n_obj):                        # objects (clusters)
        cx, cy = rng.uniform(-2.5, 2.5, 2)
        cls = SURF[2 + rng.integers(3)]           # cabinet/bed/other
        for _ in range(4):
            cells.append((cls, c, np.array([cx + rng.uniform(-0.2, 0.2),
                          cy + rng.uniform(-0.2, 0.2), rng.uniform(0.2, 1.0)])))
    return cells


def _build(W, roles, scode, dcode, cells, nph=8, nmag=2):
    """bind surface-class bits + desc bits at each cell position; 4-bit quant."""
    mp = np.zeros(W.shape[0], complex)
    for cls, clu, xyz in cells:
        bits = scode[cls] if clu < 0 else np.concatenate([scode[cls], dcode[clu]])
        mp += roles[bits].sum(0) * np.exp(1j * (xyz @ W.T))
    return q_polar_np(mp, nph, nmag)             # STORED at 4 bits/cell


def _query(W, roles, qbits, pts, z):
    unb = np.conj(roles[qbits]).sum(0)
    g = np.concatenate([pts, np.full((len(pts), 1), z)], 1)
    return np.real(np.exp(1j * (g @ W.T)) @ np.conj(unb * 0 + unb)) / W.shape[0]


def _auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    x = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    return (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def run(seeds=24):
    W = W_of_np(dirs_ring_np(), BAND); D = W.shape[0]
    rng = np.random.default_rng(0)
    roles = np.exp(1j * rng.uniform(0, 2 * np.pi, (M, D)))
    scode = _codes(SURF, 1)
    surf_auc = {"floor": [], "wall": []}
    qbe_auc = []
    for s in range(seeds):
        cells = _segment(s)
        n_obj = max(c for _, c, _ in cells) + 1
        dcode = _codes(list(range(n_obj)), 100 + s)       # per-cluster desc codes
        mp = _build(W, roles, scode, dcode, cells)
        pos = np.stack([c[2] for c in cells])
        # SURFACE query: floor/wall cells read higher than the rest
        for surf in ("floor", "wall"):
            dens = np.real(np.exp(1j * (pos @ W.T)) @ np.conj(
                np.conj(roles[scode[surf]]).sum(0) * mp)) / D
            lab = np.array([c[0] for c in cells])
            surf_auc[surf].append(_auc(dens[lab == surf], dens[lab != surf]))
        # QBE: query cluster 0's desc bits -> its cells read above other cells
        if n_obj:
            dens = np.real(np.exp(1j * (pos @ W.T)) @ np.conj(
                np.conj(roles[dcode[0]]).sum(0) * mp)) / D
            clu = np.array([c[1] for c in cells])
            qbe_auc.append(_auc(dens[clu == 0], dens[clu != 0]))
    n_cells = len(_segment(0))
    map_bytes = D * 4 / 8                              # 4 bits/cell
    reg_bytes = n_cells * (3 * 2 + 1)                  # xyz(int16)+cls per cell registry
    print(f"SURFACES-TIER + QBE per-segment band (D={D}, band {BAND[0]:.2f}-"
          f"{BAND[-1]:.1f} m, 4-bit polar quant, {seeds} segments):")
    print(f"  surface-query recall (readout AUC vs other cells):")
    print(f"    floor : {np.nanmean(surf_auc['floor']):.3f}")
    print(f"    wall  : {np.nanmean(surf_auc['wall']):.3f}")
    print(f"  QBE retrieval (example cluster's desc bits -> its cells): "
          f"AUC {np.nanmean(qbe_auc):.3f}")
    print(f"  bytes/segment: map {map_bytes:.0f} B (D={D} @ 4 bits) + registry "
          f"~{reg_bytes} B ({n_cells} cells) = ~{map_bytes+reg_bytes:.0f} B")
    print("  => THIS is what ships: surfaces (5-class labels) + label-free QBE "
          "(desc bits) in ONE ~"
          f"{(map_bytes+reg_bytes)/1024:.1f} KB bounded per-segment vector, "
          "at 4 bits/cell. Fine-object CLASS labels wait on a real RGB-D sensor "
          "(demotion, msg efc5f0e). Anti-oracle: synthetic, GT scores only.")


if __name__ == "__main__":
    run()
