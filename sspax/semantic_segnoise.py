"""How accurate must the SEG head be for the queryable map to be USEFUL? The
deploy-budget seg head is weak (P4: pixacc ~0.33, mIoU 0.04), and the objmap
round-trip only measures SELF-consistency (query returns what the net predicted,
right or wrong). The load-bearing deploy question is different: with a seg error
rate p (each object mislabeled with prob p), can a query for the TRUE class still
find the TRUE objects? This bridges seg accuracy -> map usefulness and sets the
seg-accuracy bar the P4 head must clear.

Build the map from CORRUPTED (noisy-seg) classes; query the TRUE class; report
recall (true objects found) + precision (found objects truly that class) vs p.

Anti-oracle: synthetic, TRUE classes score only (never seed the map — the map is
built from the corrupted labels, standing in for the net's predictions).

  python3 -m sspax.semantic_segnoise
"""
import numpy as np

from sspax import semantic as SM
from sspax.semantic_capacity import W_of_D, roles_of

M = 256


def run(seeds=32, k=12, n_obj=8, D_dirs=60):
    W = W_of_D(D_dirs)
    roles = roles_of(M, W.shape[0])
    grid, shape = SM._grid()
    cls = SM.CLASSES
    print(f"SEG-ACCURACY -> MAP USEFULNESS (D={W.shape[0]}, {n_obj} objects, "
          f"query TRUE class under seg error rate p):\n")
    print(f"  {'p(mislabel)':>11} {'recall':>7} {'precision':>10}", flush=True)
    for p in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7):
        recs, precs = [], []
        for s in range(seeds):
            codes = SM.class_codes(cls, k=k, m=M)
            rng = np.random.default_rng(s + 400)
            true = [(cls[rng.integers(len(cls))],
                     np.array([rng.uniform(-5, 5), rng.uniform(-4, 4), 0.5]))
                    for _ in range(n_obj)]
            # corrupt: each object's LABEL flips to a random other class w.p. p
            noisy = []
            for name, xyz in true:
                if rng.random() < p:
                    name = cls[(cls.index(name) + 1 +
                                rng.integers(len(cls) - 1)) % len(cls)]
                noisy.append((name, xyz))
            mp = np.zeros(W.shape[0], complex)
            for name, xyz in noisy:
                mp += roles[codes[name]].sum(0) * np.exp(1j * (xyz @ W.T))
            true_chairs = [x[:2] for n, x in true if n == "chair"]
            if not true_chairs:
                continue
            dens = np.real(np.exp(1j * (np.concatenate(
                [grid, np.full((len(grid), 1), 0.5)], 1) @ W.T))
                @ np.conj(np.conj(roles[codes["chair"]]).sum(0) * mp)) \
                / W.shape[0]
            pk = SM.detect(dens, grid, shape)
            tp, fp, _ = SM._match(pk, true_chairs)
            recs.append(tp / len(true_chairs))
            precs.append(tp / max(tp + fp, 1))
        print(f"  {p:>11.1f} {np.mean(recs):>7.2f} {np.mean(precs):>10.2f}",
              flush=True)
    print("\n  => query recall/precision vs seg error rate = the seg-accuracy "
          "BAR the deploy head must clear for a USEFUL queryable map (not just "
          "self-consistent). Where recall/precision cross ~0.5 tells you how "
          "good the P4 seg/label head (or its distillation teacher) must be.",
          flush=True)


if __name__ == "__main__":
    run()
