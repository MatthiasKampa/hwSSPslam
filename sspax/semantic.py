"""Semantic SSP map: binary-descriptor binding with bit-count significance.

The architecture (user directive 2026-07-15): a CNN (vision or lidar) emits a
BINARY descriptor per feature; each set bit i has a fixed random role hypervector
ROLES[i]; the feature is written to the bounded map by binding each active bit's
role with the feature POSITION (an SSP phasor) and bundling:

    map += sum_{i in bits} ROLES[i] (x) encode(xyz)          ( (x) = phase mult )

Two properties fall out:
  * SIGNIFICANCE = bit count. A position bound under k bits appears k times in
    the bundle, so it reads out k x stronger — the system grades a feature's
    importance by how many bits it sets ("add bits to space individually").
  * SEMANTIC QUERY. To find class Q (a bit set, e.g. "chair"): unbind each of
    Q's roles and decode the spatial density. An object o reads out with peak
    height proportional to |Q ∩ bits_o| — matching classes light up, other
    classes cross-talk only through random bit overlap (suppressed).

Binarization is the FPGA substrate (BNN XNOR-popcount) AND the binding algebra.
Everything is O(D) phase ops on a fixed-size vector — no history stored.
"""
import numpy as np
import jax.numpy as jnp

from sspax import core as C
from sspax import sphere as SPH

OCT6 = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
W_POS = np.asarray(C.W_of(SPH.dirs_ring(60, spacing="arc", apportion="const",
                                        stagger=0.5, n_rings=6), OCT6))
D = W_POS.shape[0]
M_ROLES = 256                              # descriptor width (bits)


def make_roles(m=M_ROLES, d=D, seed=0):
    """m fixed random unit-phasor role vectors (the attribute/bit book)."""
    rng = np.random.default_rng(seed)
    return np.exp(1j * rng.uniform(0, 2 * np.pi, (m, d)))


ROLES = make_roles()


def class_codes(names, k=12, m=M_ROLES, seed=1):
    """Assign each class a random k-bit code over the m roles (a sparse binary
    descriptor). Distinct classes overlap only randomly -> low cross-talk."""
    rng = np.random.default_rng(seed)
    return {n: np.sort(rng.choice(m, k, replace=False)) for n in names}


def pos_ssp(xyz):
    """Unnormalized position phasor encode(W, xyz) (single point)."""
    return np.exp(1j * (np.asarray(xyz) @ W_POS.T))


def bind_feature(bits, xyz, roles=ROLES):
    """One feature -> its contribution to the map: sum over active bits of
    role (x) position. `bits` is an index array (the set bits)."""
    p = pos_ssp(xyz)
    return roles[bits].sum(0) * p            # (D,) ; magnitude ~ len(bits)


def build_map(objects, codes, roles=ROLES, gate=1):
    """objects: list of (class_name, xyz). Bundle bound features into ONE
    bounded map vector. gate = min bits to admit a feature (significance gate)."""
    m = np.zeros(D, complex)
    for name, xyz in objects:
        bits = codes[name]
        if len(bits) >= gate:
            m += bind_feature(bits, xyz, roles)
    return m


def query(mp, query_bits, grid_xy, z=0.5, roles=ROLES):
    """Spatial density readout for a class bit-set over an (x,y) grid at height
    z. Peaks at positions whose feature shares bits with query_bits, height
    ~ |query ∩ feature_bits|. Returns density (len(grid_xy),)."""
    unbound = (np.conj(roles[query_bits]).sum(0)) * mp        # (D,)
    g = np.concatenate([np.asarray(grid_xy),
                        np.full((len(grid_xy), 1), z)], 1)
    return np.real(np.exp(1j * (g @ W_POS.T)) @ np.conj(unbound)) / D


def peaks(density, grid_xy, thresh, min_sep=0.6):
    """Non-max-suppressed peaks above thresh (greedy)."""
    order = np.argsort(-density)
    out = []
    for i in order:
        if density[i] < thresh:
            break
        if all(np.linalg.norm(grid_xy[i] - p) > min_sep for p, _ in out):
            out.append((grid_xy[i], float(density[i])))
    return out


# --------------------------------------------------------------------------
#  demo / evaluation
# --------------------------------------------------------------------------
CLASSES = ["chair", "table", "bed", "couch", "wardrobe"]


def _scene(n_obj, seed, extent=(12.0, 10.0)):
    rng = np.random.default_rng(seed)
    objs = []
    for _ in range(n_obj):
        name = CLASSES[rng.integers(len(CLASSES))]
        xy = rng.uniform([-extent[0] / 2, -extent[1] / 2],
                         [extent[0] / 2, extent[1] / 2])
        objs.append((name, np.array([xy[0], xy[1], 0.5])))
    return objs


def _grid(extent=(12.0, 10.0), step=0.2):
    xs = np.arange(-extent[0] / 2, extent[0] / 2, step)
    ys = np.arange(-extent[1] / 2, extent[1] / 2, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], 1), (len(xs), len(ys))


def detect(density, grid_xy, shape, n_sigma=4.0):
    """2D local-maxima above (median + n_sigma * MAD-std), robust to the SSP
    decode background. Returns peaks [(xy, height)] sorted by height."""
    D2 = density.reshape(shape)
    med = np.median(density)
    std = 1.4826 * np.median(np.abs(density - med)) + 1e-9
    thr = med + n_sigma * std
    out = []
    nx, ny = shape
    for i in range(nx):
        for j in range(ny):
            v = D2[i, j]
            if v < thr:
                continue
            lo_i, hi_i = max(0, i - 1), min(nx, i + 2)
            lo_j, hi_j = max(0, j - 1), min(ny, j + 2)
            if v >= D2[lo_i:hi_i, lo_j:hi_j].max():
                out.append((grid_xy[i * ny + j], float(v)))
    return sorted(out, key=lambda t: -t[1])


def _match(pk, truth, tol=0.8):
    """One-to-one greedy match of peaks to truth positions. -> (tp, fp, errs)."""
    used = [False] * len(truth)
    tp, errs = 0, []
    for p, _ in pk:
        d = [np.linalg.norm(p - t) if not used[k] else 9e9
             for k, t in enumerate(truth)]
        if truth and min(d) < tol:
            used[int(np.argmin(d))] = True
            tp += 1
            errs.append(min(d))
    return tp, len(pk) - tp, errs


def demo(n_obj=16, seed=0, k=12, verbose=True):
    codes = class_codes(CLASSES, k=k)
    objs = _scene(n_obj, seed)
    mp = build_map(objs, codes)
    grid, shape = _grid()
    chairs = [o[1][:2] for o in objs if o[0] == "chair"]
    others = [o[1][:2] for o in objs if o[0] != "chair"]
    dens = query(mp, codes["chair"], grid)
    pk = detect(dens, grid, shape)
    tp, fp, errs = _match(pk, chairs)
    recall = tp / max(len(chairs), 1)
    prec = tp / max(tp + fp, 1)

    def readout_at(pts):
        if not pts:
            return float("nan")
        return float(np.mean(query(mp, codes["chair"], np.array(pts))))
    if verbose:
        print(f"semantic query 'chair'  ({n_obj} objects, {len(chairs)} chairs, "
              f"{len(others)} other; k={k} bits, D={D}):")
        print(f"  precision {prec:.2f}  recall {recall:.2f}  "
              f"pos err med {np.median(errs) if errs else float('nan'):.2f} m "
              f"({tp} TP / {fp} FP)")
        print(f"  readout @ chairs {readout_at(chairs):.2f}  "
              f"@ non-chairs {readout_at(others):.2f}  "
              f"(contrast {readout_at(chairs)/max(readout_at(others),1e-9):.1f}x)")
    return prec, recall


def capacity(seed=0, k=12):
    """Recall of chairs vs map load (objects bundled into ONE D-vector)."""
    print(f"capacity (chair recall vs #objects in one D={D} map, k={k}):")
    for n_obj in (10, 20, 40, 80, 160):
        ps, rs = [], []
        for s in range(4):
            p, r = _quiet_demo(n_obj, seed + s, k)
            ps.append(p); rs.append(r)
        print(f"  {n_obj:4d} objects: recall {np.mean(rs):.2f}  "
              f"precision {np.mean(ps):.2f}", flush=True)


def significance(seed=3, k=12):
    """Significance = how many of the class's bits a feature COMMITS. A confident
    chair sets all k chair bits (readout ~k); a weak/ambiguous detection sets a
    fraction -> reads out proportionally weaker. The map grades a feature's
    importance by the number of bits it sets ("add bits individually")."""
    codes = class_codes(CLASSES, k=k)
    rng = np.random.default_rng(seed)
    # 5 chairs written with a rising number of committed chair-bits
    commits = [3, 6, 9, 12]
    objs = [(f"chair@{c}bits", np.array([-4.5 + 3 * i, 0.0, 0.5]), c)
            for i, c in enumerate(commits)]
    mp = np.zeros(D, complex)
    for _, xyz, c in objs:
        mp += ROLES[codes["chair"][:c]].sum(0) * pos_ssp(xyz)
    print(f"significance = committed bit count (query 'chair', k={k}, D={D}):")
    for _, xyz, c in objs:
        h = float(query(mp, codes["chair"], xyz[None, :2])[0])
        print(f"  chair committing {c:2d}/{k} bits: readout {h:5.1f}  "
              f"(~ proportional to bits set)")


def _quiet_demo(n_obj, seed, k):
    return demo(n_obj, seed, k, verbose=False)


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    elif cmd == "capacity":
        capacity()
    elif cmd == "significance":
        significance()
    elif cmd == "all":
        demo(); print(); capacity(); print(); significance()
