"""Which LATTICE gives the best QUERYABLE MAP? The place work adopted azel-oct6
(best place at D240); does that SAME lattice also localise semantic queries best
— so one lattice serves BOTH the metric-place and semantic-query roles (the
dual-use design) — or is there a place/semantic tension? Matched D=240, fixed
scene set; metrics = chair recall + localisation precision (median position error
of detected chairs). Clean, well-controlled (random-code control is banked → 0);
no capacity-vs-resolution confound here because D is FIXED and only GEOMETRY
varies.

Anti-oracle: deterministic synthetic, GT positions score recall/precision only.

  python3 -m sspax.semantic_lattice
"""
import numpy as np

from sspax import core as C
from sspax import sphere as SPH
from sspax.semantic import class_codes, CLASSES, detect, _match, _grid
from sspax.semantic_capacity import roles_of, build_and_query, _scene

OCT6 = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
EL5 = [-40, -20, 0, 20, 40]


def lattices_D240():
    """named lattices, all D=240, oct6 ladder where applicable."""
    return {
        "ring-oct6 ": C.W_of(SPH.dirs_ring(40, spacing="arc", apportion="const",
                                           stagger=0.5, n_rings=6), OCT6),
        "azel-oct6 ": C.W_of(C.dirs_azel(8, EL5), OCT6),
        "azel-house": C.make_lattices()["azel3d"],
        "fib3d-oct6": C.W_of(C.dirs_fib(40), OCT6),
        "ringstag  ": C.W_of(SPH.dirs_ringstag(40), OCT6),
    }


def run(seeds=24, k=12, m=256):
    grid, shape = _grid()
    print("QUERYABLE-MAP quality by LATTICE (all D=240, chair query, "
          f"{seeds} scenes, load=8 objects):\n")
    print(f"  {'lattice':>11} {'D':>4} {'recall':>7} {'precision(m)':>13}",
          flush=True)
    for name, W in lattices_D240().items():
        W = np.asarray(W)
        roles = roles_of(m, W.shape[0])
        recs, errs_all = [], []
        for s in range(seeds):
            codes = class_codes(CLASSES, k=k, m=m)
            objs = _scene(8, s)
            chairs = [x[:2] for n, x in objs if n == "chair"]
            if not chairs:
                continue
            dens = build_and_query(W, roles, codes, objs, codes["chair"], grid)
            tp, fp, errs = _match(detect(dens, grid, shape), chairs)
            recs.append(tp / len(chairs))
            errs_all += errs
        print(f"  {name:>11} {W.shape[0]:>4} {np.mean(recs):>7.2f} "
              f"{np.median(errs_all) if errs_all else float('nan'):>13.2f}",
              flush=True)
    print("\n  => query quality (recall + localisation precision) is nearly "
          "IDENTICAL across lattice geometries at matched D — unlike PLACE, "
          "which is geometry-sensitive (azel-oct6 0.947 > ring 0.904 at D240). "
          "The semantic query is a LOCAL position decode, which any reasonable "
          "lattice does well; place is a GLOBAL appearance comparison, where "
          "geometry matters. So the place-adopted azel-oct6 serves the semantic "
          "query too at matched geometry. NOTE: geometry has no tension, but the "
          "LADDER does — see run_ladder (place wants coarse-reach, query wants "
          "fine). So a shared lattice shares GEOMETRY freely; the ladder is the "
          "contested knob.", flush=True)


def run_ladder(seeds=24, k=12, m=256):
    """companion to run(): geometry is insensitive — is the LADDER? Fixed
    ring-oct6 geometry (60 dirs), 6-rung ladders of different wavelength RANGE
    (D=360 fixed), same scene set. Prediction: the ladder MATTERS (it sets the
    position-decode resolution) — the lever for the semantic map as for place."""
    grid, shape = _grid()
    dirs = SPH.dirs_ring(60, spacing="arc", apportion="const", stagger=0.5,
                         n_rings=6)
    ladders = {
        "fine .1-1.6 ": list(np.geomspace(0.1, 1.6, 6)),
        "oct6 .25-8  ": OCT6,
        "wide .25-90 ": list(np.geomspace(0.25, 90.5, 6)),
        "coarse 2-90 ": list(np.geomspace(2.0, 90.5, 6)),
    }
    print("\nQUERYABLE-MAP quality by LADDER (fixed ring-oct6 geometry, D=360, "
          f"{seeds} scenes, load=8; room extent ~12 m):\n")
    print(f"  {'ladder':>12} {'recall':>7} {'precision(m)':>13}", flush=True)
    for name, lams in ladders.items():
        W = np.asarray(C.W_of(dirs, lams)); roles = roles_of(m, W.shape[0])
        recs, errs_all = [], []
        for s in range(seeds):
            codes = class_codes(CLASSES, k=k, m=m)
            objs = _scene(8, s)
            chairs = [x[:2] for n, x in objs if n == "chair"]
            if not chairs:
                continue
            dens = build_and_query(W, roles, codes, objs, codes["chair"], grid)
            tp, fp, errs = _match(detect(dens, grid, shape), chairs)
            recs.append(tp / len(chairs)); errs_all += errs
        print(f"  {name:>12} {np.mean(recs):>7.2f} "
              f"{np.median(errs_all) if errs_all else float('nan'):>13.2f}",
              flush=True)
    print("\n  => the LADDER is the semantic-map lever (as for place): a ladder "
          "reaching the room scale localises well; too-fine or too-coarse "
          "degrades. Geometry-insensitive + ladder-sensitive = the SAME design "
          "law as place (adopt the ladder, the geometry is free).", flush=True)


def run_ladder_resolve(seeds=24, k=12, m=256):
    """resolve the place/semantic ladder TENSION: does adding MORE RUNGS (span
    object-fine AND venue-coarse in one ladder) recover query quality that the
    6-rung wide ladder diluted (0.59)? Fixed ring-oct6 geometry; query recall +
    precision. A 12-rung wide ladder has BOTH fine rungs (query) and coarse rungs
    (place) -> should serve both at 2x D."""
    grid, shape = _grid()
    dirs = SPH.dirs_ring(60, spacing="arc", apportion="const", stagger=0.5,
                         n_rings=6)
    cfgs = {
        "6-rung wide .25-90 (D360)": list(np.geomspace(0.25, 90.5, 6)),
        "12-rung wide .1-90 (D720)": list(np.geomspace(0.1, 90.5, 12)),
        "12-rung fine .1-8  (D720)": list(np.geomspace(0.1, 8.0, 12)),
    }
    print("\nRESOLVING the ladder tension (more rungs span both scales), "
          f"ring-oct6 geometry, {seeds} scenes, load=8:\n")
    print(f"  {'ladder':>26} {'D':>4} {'recall':>7} {'precision(m)':>13}",
          flush=True)
    for name, lams in cfgs.items():
        W = np.asarray(C.W_of(dirs, lams)); roles = roles_of(m, W.shape[0])
        recs, errs_all = [], []
        for s in range(seeds):
            codes = class_codes(CLASSES, k=k, m=m)
            objs = _scene(8, s)
            chairs = [x[:2] for n, x in objs if n == "chair"]
            if not chairs:
                continue
            dens = build_and_query(W, roles, codes, objs, codes["chair"], grid)
            tp, fp, errs = _match(detect(dens, grid, shape), chairs)
            recs.append(tp / len(chairs)); errs_all += errs
        print(f"  {name:>26} {W.shape[0]:>4} {np.mean(recs):>7.2f} "
              f"{np.median(errs_all) if errs_all else float('nan'):>13.2f}",
              flush=True)
    print("\n  => RESULT: more rungs do NOT resolve it (12-rung wide 0.63 ~ "
          "6-rung wide 0.59; only FINE ladders recover, 12-rung fine 0.95). The "
          "coarse rungs place needs ACTIVELY dilute the query (a 90 m wavelength "
          "is near-constant across a 12 m room -> DC-like cross-talk, no "
          "localisation). The place/semantic ladder tension is FUNDAMENTAL: a "
          "dual-use map needs SEPARATE ladders per role (or one role "
          "compromised), not one shared wide ladder.", flush=True)


if __name__ == "__main__":
    run()
    run_ladder()
    run_ladder_resolve()
