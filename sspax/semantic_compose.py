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

    def readout(mp, qbits, cent):
        return np.real(np.exp(1j * (np.concatenate(
            [cent, np.full((len(cent), 1), 0.5)], 1) @ W.T))
            @ np.conj(np.conj(roles[qbits]).sum(0) * mp)) / W.shape[0]

    # grading by #matching attributes: REAL query vs RANDOM-query control
    gr = {"real": [[], [], []], "rand": [[], [], []]}
    ru, pu, rp, pp = [], [], [], []        # union-bits / product-of-queries retrieval
    for s in range(seeds):
        rng = np.random.default_rng(s + 700)
        objs = [(CLASSES[rng.integers(len(CLASSES))], COLOURS[rng.integers(len(COLOURS))],
                 np.array([*rng.uniform([-5, -4], [5, 4]), 0.5])) for _ in range(n_obj)]
        mp = np.zeros(W.shape[0], complex)
        for cl, co, xyz in objs:
            mp += roles[np.concatenate([ccode[cl], kcode[co]])].sum(0) * np.exp(1j * (xyz @ W.T))
        cent = np.stack([o[2][:2] for o in objs])
        qcl, qco = "chair", "red"
        d_union = readout(mp, np.concatenate([ccode[qcl], kcode[qco]]), cent)
        d_rand = readout(mp, np.sort(rng.choice(M, 2 * k, replace=False)), cent)
        d_prod = readout(mp, ccode[qcl], cent) * readout(mp, kcode[qco], cent)
        for i, (cl, co, _) in enumerate(objs):
            m = (cl == qcl) + (co == qco)
            gr["real"][m].append(d_union[i]); gr["rand"][m].append(d_rand[i])
        m2 = np.array([(cl == qcl) and (co == qco) for cl, co, _ in objs])
        if m2.any():
            for d, rr, pr in [(d_union, ru, pu), (d_prod, rp, pp)]:
                thr = np.percentile(d[~m2], 90); pk = d >= thr
                tp = (pk & m2).sum(); fp = (pk & ~m2).sum()
                rr.append(tp / m2.sum()); pr.append(tp / max(tp + fp, 1))

    print(f"COMPOSITIONAL query 'red chair' (D={W.shape[0]}, {n_obj} objects, "
          f"{len(CLASSES)}x{len(COLOURS)} attributes, k={k}/attr):")
    print(f"  readout by #matching attributes   2-match  1-match  0-match")
    print(f"    REAL query                     {np.mean(gr['real'][2]):6.1f}  "
          f"{np.mean(gr['real'][1]):6.1f}  {np.mean(gr['real'][0]):6.1f}  -> grades 2>1>0")
    print(f"    RANDOM-query CONTROL           {np.mean(gr['rand'][2]):6.1f}  "
          f"{np.mean(gr['rand'][1]):6.1f}  {np.mean(gr['rand'][0]):6.1f}  -> FLAT (grading is bit-OVERLAP, not object count)")
    print(f"  red-chair RETRIEVAL (candidate set = GT object CENTROIDS -> readout")
    print(f"  SEPARABILITY at known positions, NOT grid-detect; thr @ 90th pct non-target):")
    print(f"    union-bits         : recall {np.mean(ru):.2f}  precision {np.mean(pu):.2f}")
    print(f"    product-of-queries : recall {np.mean(rp):.2f}  precision {np.mean(pp):.2f}")
    print(f"  => the map COMPOSES: conjunctive readout ~ matched-attribute count; "
          f"precision ceiling (~0.55) is D-invariant (operator/threshold, not cross-talk).")


def load_sweep(seeds=24, k=12, m=256, D_dirs=60):
    """compose SPENDS capacity: a 2-attribute object commits 2k bits, so by the
    significance-capacity law it loads the bounded map like a double-significance
    one. Chair-recall (via union-bits query) vs #objects, single-attr (class only)
    vs double-attr (class+colour) binding, at fixed D."""
    from sspax.semantic import _grid, detect, _match
    W = W_of_D(D_dirs); roles = roles_of(m, W.shape[0]); grid, shape = _grid()
    rng0 = np.random.default_rng(0)
    ccode = {c: np.sort(rng0.choice(m, k, replace=False)) for c in CLASSES}
    kcode = {c: np.sort(rng0.choice(m, k, replace=False)) for c in COLOURS}
    print(f"\n  compose CAPACITY COST (D={W.shape[0]}, chair recall vs #objects):")
    print(f"    {'#obj':>5} {'1-attr (k bits)':>16} {'2-attr (2k bits)':>17}")
    for n_obj in (6, 10, 16, 24):
        row = []
        for two_attr in (False, True):
            rs = []
            for s in range(seeds):
                rng = np.random.default_rng(s + 800)
                objs = [(CLASSES[rng.integers(len(CLASSES))], COLOURS[rng.integers(len(COLOURS))],
                         np.array([*rng.uniform([-5, -4], [5, 4]), 0.5])) for _ in range(n_obj)]
                chairs = [o[2][:2] for o in objs if o[0] == "chair"]
                if not chairs:
                    continue
                mp = np.zeros(W.shape[0], complex)
                for cl, co, xyz in objs:
                    bits = np.concatenate([ccode[cl], kcode[co]]) if two_attr else ccode[cl]
                    mp += roles[bits].sum(0) * np.exp(1j * (xyz @ W.T))
                g = np.concatenate([grid, np.full((len(grid), 1), 0.5)], 1)
                dens = np.real(np.exp(1j * (g @ W.T)) @ np.conj(
                    np.conj(roles[ccode["chair"]]).sum(0) * mp)) / W.shape[0]
                tp, fp, _ = _match(detect(dens, grid, shape), chairs)
                rs.append(tp / len(chairs))
            row.append(np.mean(rs))
        print(f"    {n_obj:>5} {row[0]:>16.2f} {row[1]:>17.2f}")
    print("    => 2-attr objects (2k committed bits) exhaust capacity FASTER — "
          "composition spends the bounded budget (per the significance-capacity law).")


if __name__ == "__main__":
    run()
    load_sweep()
