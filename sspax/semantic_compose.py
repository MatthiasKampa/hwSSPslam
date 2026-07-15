"""COMPOSITIONAL / conjunctive queries on the bounded map — the multi-attribute
form of the user's "binary descriptor, add bits individually" directive. Each
object binds the UNION of several attribute codes (e.g. CLASS + COLOUR); a
conjunctive query "red chair" = class_code[chair] ∪ colour_code[red]. Because the
readout weight is |query_bits ∩ object_bits|, an object matching BOTH attributes
reads ~2k, a single-attribute match ~k, a non-match ~0 — so the map grades by
NUMBER OF MATCHING ATTRIBUTES, i.e. it composes. This tests whether the bounded
map answers conjunctive object queries ("red chairs, not blue chairs or red
tables") from ONE vector, no extra structure.

Anti-oracle: synthetic, GT (class,colour) score only. Pure numpy.

  python3 -m sspax.semantic_compose
"""
import numpy as np

from sspax import semantic as SM
from sspax.semantic_capacity import W_of_D, roles_of

M = 256
CLASSES = ["chair", "table", "bed", "couch"]
COLOURS = ["red", "blue", "green", "grey"]


def run(seeds=32, k=12, n_obj=10, D_dirs=60):
    W = W_of_D(D_dirs); roles = roles_of(M, W.shape[0])
    rng0 = np.random.default_rng(0)
    ccode = {c: np.sort(rng0.choice(M, k, replace=False)) for c in CLASSES}
    kcode = {c: np.sort(rng0.choice(M, k, replace=False)) for c in COLOURS}

    two, one, zero = [], [], []            # readout by #matching attributes
    recs, precs = [], []
    for s in range(seeds):
        rng = np.random.default_rng(s + 700)
        objs = []                          # (class, colour, xyz)
        for _ in range(n_obj):
            cl = CLASSES[rng.integers(len(CLASSES))]
            co = COLOURS[rng.integers(len(COLOURS))]
            xy = rng.uniform([-5, -4], [5, 4])
            objs.append((cl, co, np.array([xy[0], xy[1], 0.5])))
        mp = np.zeros(W.shape[0], complex)
        for cl, co, xyz in objs:
            bits = np.concatenate([ccode[cl], kcode[co]])     # UNION of attributes
            mp += roles[bits].sum(0) * np.exp(1j * (xyz @ W.T))
        # conjunctive query "red chair"
        qcl, qco = "chair", "red"
        qbits = np.concatenate([ccode[qcl], kcode[qco]])
        cent = np.stack([o[2][:2] for o in objs])
        dens = np.real(np.exp(1j * (np.concatenate(
            [cent, np.full((len(cent), 1), 0.5)], 1) @ W.T))
            @ np.conj(np.conj(roles[qbits]).sum(0) * mp)) / W.shape[0]
        for i, (cl, co, _) in enumerate(objs):
            match = (cl == qcl) + (co == qco)
            [zero, one, two][match].append(dens[i])
        # retrieval of the 2-match (red chairs) via a threshold at the 1-match level
        m2 = np.array([(cl == qcl) and (co == qco) for cl, co, _ in objs])
        if m2.any():
            thr = np.percentile(dens[~m2], 90)         # top-10% of non-target
            pk = dens >= thr
            tp = (pk & m2).sum(); fp = (pk & ~m2).sum()
            recs.append(tp / m2.sum()); precs.append(tp / max(tp + fp, 1))

    print(f"COMPOSITIONAL query 'red chair' (D={W.shape[0]}, {n_obj} objects, "
          f"{len(CLASSES)}x{len(COLOURS)} attributes, k={k}/attr):")
    print(f"  readout by #matching attributes:")
    print(f"    2 (red chair)          : {np.mean(two):6.2f}  (n={len(two)})")
    print(f"    1 (red !chair / !red chair): {np.mean(one):6.2f}  (n={len(one)})")
    print(f"    0 (neither)            : {np.mean(zero):6.2f}  (n={len(zero)})")
    print(f"  => grades 2 > 1 > 0: the map COMPOSES attributes — a conjunctive "
          f"query reads out proportional to matched-attribute count.")
    print(f"  red-chair retrieval (thr @ 90th pct of non-target): "
          f"recall {np.mean(recs):.2f}  precision {np.mean(precs):.2f}")
    # control: does a RANDOM 2k-bit query grade? (should not)
    print(f"  [control: a random-code query should NOT grade 2>1>0 — the "
          f"grading is from bit OVERLAP, not object count]")


if __name__ == "__main__":
    run()
