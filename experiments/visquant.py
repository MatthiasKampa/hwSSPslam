"""Vision snapshot-library QUANTIZATION law (S-corner; deploy variant
v1.1 claims <=0.5 KB/anchor at 2 b for the vision channels — measured
here for the first time; the lidar side's law is banked in lidarscale
fidelity: 2b 0.925 vs float 0.928 at D1920).

Channels quantized phase-only (house 2b semantics = nph 4, nmag 1):
  gridint c16   the cross-view place channel (W_vis_fib D240)
  intphase      the precision/verification channel
Metrics per nph {4, 8, 16, float}: vis place-AUC, max-fused AUC with the
float lidar rot-searched channel, adjacent-frame repeatability (the
ego-motion proxy), bytes/anchor. Venues: school_run2 (est labels,
DIAGNOSTIC margins) + spot classroom (withheld-odometry labels).

Usage: python3 -m experiments.visquant [run]
"""
import sys

import numpy as np

import experiments.detzoo as DZ
import experiments.lattice3d as L3
import experiments.twomap as TM


def quant(V, nph):
    if nph == 0:
        return V
    ph = np.round(np.angle(V) * nph / (2 * np.pi)) * (2 * np.pi / nph)
    Q = np.exp(1j * ph)
    return Q / np.linalg.norm(Q, axis=-1, keepdims=True)


def main(run):
    labels = "ref" if run == "spot" else "est"
    grays, clouds, pose, kind, K = DZ.frame_set(run, labels)
    pr = dict(same_r=1.0, far_lo=4.0) if labels == "est" else {}
    (si, sj), (di, dj) = L3._pairs(pose, **pr)
    adj = np.arange(len(pose) - 1)
    Sl = TM._znorm(TM.lidar_rotsim(clouds))
    auc_l = L3._auc(Sl[si, sj], Sl[di, dj])
    Wv = TM.W_vis_fib()
    chans = {
        "gridint": np.stack([L3.encode(Wv, TM.cam_bearings(
            K, *TM.grid_int(g)[:2]), TM.grid_int(g)[2]) for g in grays]),
        "intphase": np.stack([TM.enc_intphase(g, K, Wv) for g in grays]),
    }
    print(f"vision snapshot quantization ({run}, {len(grays)} frames, "
          f"labels={kind}; lidar float rot-searched {auc_l:.3f}):")
    for name, V in chans.items():
        for nph, tag in ((0, "float"), (16, "4b"), (8, "3b"), (4, "2b")):
            Vq = quant(V, nph)
            Sv = TM._znorm(np.abs(Vq @ Vq.conj().T))
            a = L3._auc(Sv[si, sj], Sv[di, dj])
            Sm = np.maximum(Sl, Sv)
            am = L3._auc(Sm[si, sj], Sm[di, dj])
            rep = L3._auc(Sv[adj, adj + 1], Sv[di, dj])
            by = (2 * len(Wv) * 8 if nph == 0
                  else 2 * len(Wv) * int(np.log2(nph))) // 8
            print(f"  {name:8s} {tag:5s}: vis {a:.3f}  max-f {am:.3f}  "
                  f"adj {rep:.3f}  ({by:4d} B/anchor)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "school_run2")
