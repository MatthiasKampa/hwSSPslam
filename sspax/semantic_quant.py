"""Does the bounded SEMANTIC MAP survive low-precision arithmetic? The project
thesis is that binary/low-precision is the FPGA substrate (BNN XNOR-popcount).
The vision DESCRIPTOR net quantizes to int4 (banked); here we test the SEMANTIC
MAP's stored phasors — the bounded EBR content — under the FPGA polar-quant model
(`core.q_polar_np`: nph phase bins x nmag magnitude levels). If chair-query
recall holds down to a few phase bits, the queryable map is FPGA-deployable at
its claimed BNN-regime footprint.

Two things to quantize: (a) the STORED map vector (EBR memory), and (b) also the
ROLES (the fixed attribute codebook). Query at full precision otherwise.

Anti-oracle: synthetic, GT scores only. Pure numpy.

  python3 -m sspax.semantic_quant
"""
import numpy as np

from sspax import core as C
from sspax.semantic import class_codes, CLASSES, detect, _match, _grid, query
from sspax.semantic_capacity import W_of_D, roles_of, _scene, build_and_query

M = 256


def _recall_quant(W, roles, nph, nmag, seeds=24, k=12, n_obj=6, quant_roles=False):
    grid, shape = _grid()
    rs = []
    for s in range(seeds):
        codes = class_codes(CLASSES, k=k, m=M)
        objs = _scene(n_obj, s)
        chairs = [x[:2] for n, x in objs if n == "chair"]
        if not chairs:
            continue
        mp = np.zeros(W.shape[0], complex)
        for name, xyz in objs:
            mp += roles[codes[name]].sum(0) * np.exp(1j * (xyz @ W.T))
        if nph:
            mp = C.q_polar_np(mp, nph, nmag)              # quantize STORED map
        R = C.q_polar_np(roles, nph, nmag) if (nph and quant_roles) else roles
        unb = np.conj(R[codes["chair"]]).sum(0) * mp
        g = np.concatenate([grid, np.full((len(grid), 1), 0.5)], 1)
        dens = np.real(np.exp(1j * (g @ W.T)) @ np.conj(unb)) / W.shape[0]
        tp, fp, _ = _match(detect(dens, grid, shape), chairs)
        rs.append(tp / len(chairs))
    return float(np.mean(rs))


def run(D_dirs=60):
    W = W_of_D(D_dirs); roles = roles_of(M, W.shape[0])
    print(f"SEMANTIC MAP under FPGA polar-quant (D={W.shape[0]}, chair recall; "
          f"nph phase bins x nmag mag levels):\n")
    base = _recall_quant(W, roles, None, None)
    print(f"  float baseline: recall {base:.2f}\n", flush=True)
    print(f"  {'nph':>4} {'nmag':>5} {'map-only':>9} {'map+roles':>10} "
          f"{'~bits/cell':>11}", flush=True)
    for nph, nmag in [(64, 8), (32, 8), (16, 4), (8, 4), (8, 2), (4, 2)]:
        r_map = _recall_quant(W, roles, nph, nmag)
        r_both = _recall_quant(W, roles, nph, nmag, quant_roles=True)
        bits = np.log2(nph) + np.log2(nmag)
        print(f"  {nph:>4} {nmag:>5} {r_map:>9.2f} {r_both:>10.2f} "
              f"{bits:>11.0f}", flush=True)
    print("\n  => how few phase/magnitude bits keep the query alive = the "
          "deployable EBR footprint of the semantic map. If recall holds at "
          "nph=8-16 (3-4 phase bits), the bounded map fits the low-precision "
          "FPGA substrate the thesis claims (a handful of bits per D-cell).",
          flush=True)


if __name__ == "__main__":
    run()
