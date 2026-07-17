"""Unified-net HEAD EXPORT CONTRACT + deploy-box gates (pure numpy, no
JAX/torch — the transfer-gate side of TRAINING_PROGRAM.md P4b).

THE CONTRACT (one .npz per trained unified net):
  meta        json string:
    version   1 | 2 (v2 = 2026-07-15c, shaped to the FIRST TRAINED nets
              sspax/vision/segnet.py + sspax/lidar_ring.py; v1 files
              still load — in_div defaults to 255, desc to absent.
              v2.2 EXTENSION 2026-07-16: vision RGB geometries
              (240x320x3 / 120x160x3) are pinned deploy inputs — the
              ADE-150 bottleneck encoder measured colour as a real
              lever (gray -40% rel pixacc) — and the TRACK head is
              OPTIONAL (the bottleneck exports trunk+desc only);
              same version tag, absent keys default)
    modality  "vision" | "lidar"
    in_h/in_w input resolution — MUST be a pinned deploy geometry
              (vision 240x320 or 120x160, Y8 or RGB; lidar 3x1024 or
              3x512 ring raster, rings-as-channels)
    in_ch     1 (vision Y8) | 3 (vision RGB / lidar rings)
    in_div    input scale divisor (v2): vision 255.0 (Y8 -> [0,1]),
              lidar 1.0 (ring ranges stay in METERS — the nets train
              on raw meters; /255 here was a v1 vision-ism)
    cell      trunk stride product (any pinned combo that lands on the
              30x40 vision cell grid: full res stride 8 or half res
              stride 4; lidar: azimuth downsample, e.g. 4 -> 256 bins)
    act       "relu" | "crelu" (hidden layers; last layer of every
              stack is linear; CReLU doubles the NEXT layer's cin)
    trunk     [{kind: conv|dw|conv1d, k, s, cin, cout}, ...]
    track     (OPTIONAL since v2.2) layer list -> (h, w, 1) or
              (h, w, 2): [:, :, 0] = per-cell
              saliency weight (use max(0, .)); [:, :, 1] (optional) =
              thermometer cutoff — RETIRED per the 2026-07-15 P1
              retraction (kept loadable; two same-input 1x1 convs merge
              exactly into one cout=2 conv if a trained net has both)
    desc      (v2, optional) layer list -> (h, w, desc_bits) tracking-
              descriptor activations; bits = act > 0 (sign binarize).
              THE CARRYING HEAD on trained nets (NYUv2 retrieval 0.95,
              int4-robust) — v1 had no slot for it.
    desc_bits len of the descriptor (32 on the trained nets)
    seg       layer list -> (h, w, k_bits) LABEL-LATENT bit logits (the
              binarized class embedding — export THIS head, not the
              40-class training softmax); bits = logits > thresh.
              OPTIONAL in v2 when desc is present (desc-only tracking
              heads export now; the k-bit latent arrives with the
              distillation round)
    k_bits    16 | 32 (0 when seg absent)
    thresh    per-bit binarization thresholds (len k_bits)
  arrays      per stack prefix T (trunk) / A (track) / G (seg) /
              D (desc, v2):
    {P}{i}_w  int8 weights — conv: (cout, cin, k, k); dw: (cin, k, k);
              conv1d: (cout, cin, k)
    {P}{i}_s  float32 per-out-channel dequant scale (dw: per cin)
    {P}{i}_b  float32 bias
  semantics   x = raw/in_div float32, SAME zero pad (k//2),
              y = conv(x, w_int8 * s) + b; QAT on the training side
              maps into this exactly.

Gates provided here (run on the deploy box, rule-4 style):
  selftest    shape/determinism/roundtrip on random-weight fixtures at
              BOTH pinned vision resolutions + budget cross-check
  fixture     write a random-weight contract fixture (shape contract
              only — outputs are meaningless by construction)
  stability   cross-view bit agreement on TUM (same physical cell,
              adjacent vs far frames — the failure mode that killed raw
              census). NOTE (measured 2026-07-15): the RANDOM-weights
              fixture already scores 0.892 adj / 0.828 far — random
              convs are smooth, so input statistics alone correlate
              bits. Read the adj-far GAP and the objmap AUC, not the
              absolute agreement.
  gate        the objmap2 semantic-key gate: per-cell bits ->
              BIPOLAR-SPATTER map codes (the objmap census law:
              A = (1-2b) @ exp(iK)/sqrt(k), zero-mean cross-code) ->
              the banked two-stage/multi-view harness vs the banked
              references (int combined AUC 0.805, census 3-view
              0.165 m; poses DIAGNOSTIC as in the banked runs).
              Map-bind pieces are replicated from experiments/objmap*
              (frozen post-verdict) with the appearance hook — the
              replicate-with-cite discipline.

Usage:
  python3 -m sspax.headio selftest
  python3 -m sspax.headio fixture scratch/head_fixture.npz [full|half|lidar] [k]
  python3 -m sspax.headio stability <head.npz> [seq]     (desc bits if present)
  python3 -m sspax.headio gate <head.npz> [seq] [seg|desc]
"""
import json
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HEAD_KEY_SEED = 4                 # objmap._FAM_SEED uses 1..3; head = 4


# --------------------------------------------------------------------------
#  serialization
# --------------------------------------------------------------------------
def save_head(path, meta, arrays):
    np.savez_compressed(path, meta=np.frombuffer(
        json.dumps(meta).encode(), np.uint8), **arrays)


def load_head(path):
    z = np.load(path)
    meta = json.loads(bytes(z["meta"]).decode())
    assert meta["version"] in (1, 2)
    meta.setdefault("in_div", 255.0)          # v1 semantics
    meta.setdefault("desc", [])               # v1: no descriptor slot
    meta.setdefault("seg", [])                # v2: desc-only heads allowed
    meta.setdefault("track", [])              # v2.2: track optional
    meta.setdefault("k_bits", 0)              # (the k-bit latent arrives
    meta.setdefault("thresh", [])             #  with the distillation round)
    stacks = {}
    for pre, name in (("T", "trunk"), ("A", "track"), ("G", "seg"),
                      ("D", "desc")):
        stack = []
        for i, L in enumerate(meta[name]):
            w = z[f"{pre}{i}_w"].astype(np.float32)
            s = z[f"{pre}{i}_s"].astype(np.float32)
            b = z[f"{pre}{i}_b"].astype(np.float32)
            wq = w * (s[:, None, None, None] if L["kind"] == "conv" else
                      s[:, None, None] if L["kind"] == "dw" else
                      s[:, None, None])
            stack.append((L, wq, b))
        stacks[name] = stack
    _validate(meta)
    return dict(meta=meta, **stacks)


def _validate(meta):
    res = (meta["in_h"], meta["in_w"], meta["in_ch"])
    ok_res = {("vision"): [(240, 320, 1), (120, 160, 1),
                           (240, 320, 3), (120, 160, 3)],   # v2.2 RGB
              ("lidar"): [(3, 1024, 3), (3, 512, 3)]}
    assert res in ok_res[meta["modality"]], \
        f"input {res} is not a pinned deploy geometry (2026-07-15b/16)"
    assert float(meta["in_div"]) > 0
    crelu = meta["act"] == "crelu"
    for name in ("trunk", "track", "seg", "desc"):
        if not meta[name]:
            continue
        ch = meta["trunk"][0]["cin"] if name == "trunk" else \
            _out_ch(meta, "trunk", crelu, last_linear=False)
        for i, L in enumerate(meta[name]):
            assert L["cin"] == ch, (name, i, L["cin"], ch)
            ch = L["cout"]
            hidden = i < len(meta[name]) - 1 or name == "trunk"
            if crelu and hidden:
                ch *= 2
    s = 1
    for L in meta["trunk"]:
        s *= L["s"]
    assert s == meta["cell"], (s, meta["cell"])
    assert meta["seg"] or meta["desc"], "head must carry seg or desc bits"
    if meta["seg"]:
        assert len(meta["thresh"]) == meta["k_bits"]
        assert meta["seg"][-1]["cout"] == meta["k_bits"]
    if meta["track"]:                            # v2.2 optional; 1 =
        assert meta["track"][-1]["cout"] in (1, 2)   # weight-only (cutoff
    if meta["desc"]:                             # retired, P1 retraction)
        assert meta["desc"][-1]["cout"] == meta["desc_bits"]


def _out_ch(meta, name, crelu, last_linear=True):
    c = meta[name][-1]["cout"]
    return c * 2 if crelu and not last_linear else c


# --------------------------------------------------------------------------
#  numpy forward
# --------------------------------------------------------------------------
def _conv2d(x, w, b, k, s, dw):
    p = k // 2
    xp = np.pad(x, ((p, p), (p, p), (0, 0))) if p else x
    win = sliding_window_view(xp, (k, k), axis=(0, 1))[::s, ::s]
    if dw:                                     # win (h,w,c,k,k) w (c,k,k)
        return np.einsum("hwcab,cab->hwc", win, w) + b
    return np.einsum("hwcab,ocab->hwo", win, w) + b


def _conv1d(x, w, b, k, s):
    p = k // 2
    xp = np.pad(x, ((p, p), (0, 0))) if p else x
    win = sliding_window_view(xp, k, axis=0)[::s]   # (w', c, k)
    return np.einsum("wck,ock->wo", win, w) + b


def _run_stack(x, stack, act, last_linear=True):
    for i, (L, w, b) in enumerate(stack):
        if L["kind"] == "conv1d":
            x = _conv1d(x, w, b, L["k"], L["s"])
        else:
            x = _conv2d(x, w, b, L["k"], L["s"], L["kind"] == "dw")
        if i < len(stack) - 1 or not last_linear:
            x = (np.concatenate([np.maximum(x, 0), np.maximum(-x, 0)], -1)
                 if act == "crelu" else np.maximum(x, 0))
    return x


def forward(head, raw):
    """raw: vision (H, W) uint8/float | lidar (3, W) float ring raster
    (raw METERS — in_div=1 for lidar; /255 only where meta says so).
    -> dict(track=(h, w, 1|2) raw outputs, logits=(h, w, k), bits bool,
    + desc=(h, w, desc_bits) sign bits when the head has one)."""
    m = head["meta"]
    x = np.asarray(raw, np.float32) / float(m["in_div"])
    if m["modality"] == "vision":
        if m["in_ch"] == 1:                    # Y8: (H, W) -> (H, W, 1)
            assert x.shape == (m["in_h"], m["in_w"]), x.shape
            x = x[:, :, None]
        else:                                  # v2.2 RGB: (H, W, 3)
            assert x.shape == (m["in_h"], m["in_w"], m["in_ch"]), x.shape
    else:
        assert x.shape == (m["in_h"], m["in_w"]), x.shape
        x = np.ascontiguousarray(x.T)          # (beams, rings-as-ch)
    trunk = _run_stack(x, head["trunk"], m["act"], last_linear=False)
    out = {}
    if head["track"]:                          # v2.2: optional
        out["track"] = _run_stack(trunk, head["track"], m["act"])
    if head["seg"]:
        logits = _run_stack(trunk, head["seg"], m["act"])
        out["logits"] = logits
        out["bits"] = logits > np.asarray(m["thresh"], np.float32)
    if head["desc"]:
        da = _run_stack(trunk, head["desc"], m["act"])
        out["desc_act"] = da
        out["desc"] = da > 0
    return out


def cell_bits(head, gray, pool=2, source="seg"):
    """Per-cell bits pooled to a coarser cell grid (mean activations over
    pool x pool net cells -> threshold): the c16 objmap grid from the
    30x40 trunk grid. source: "seg" (k-bit label latent > thresh) or
    "desc" (tracking descriptor > 0 — the carrying head on trained nets).
    A half-res head on full-res stored frames (TUM 240x320) subsamples 2x
    (nearest — the trained nets' resize convention)."""
    m = head["meta"]
    gray = np.asarray(gray)
    if (m["modality"] == "vision"
            and gray.shape[:2] == (2 * m["in_h"], 2 * m["in_w"])):
        gray = gray[::2, ::2]
    out = forward(head, gray)
    a = out["logits"] if source == "seg" else out["desc_act"]
    h, w, k = a.shape
    a = a[:h - h % pool, :w - w % pool]
    a = a.reshape(h // pool, pool, w // pool, pool, k).mean((1, 3))
    th = (np.asarray(head["meta"]["thresh"], np.float32)
          if source == "seg" else 0.0)
    return a > th


# --------------------------------------------------------------------------
#  fixtures (shape contract only — random weights)
#  "full"  = the v1 full-res vision shape (stride-8 trunk, no desc):
#            exercises v1 back-compat through the v2 loader.
#  "half"  = MIRRORS the first trained vision net (sspax/vision/segnet.py,
#            NYUv2 2026-07-15): half res, stride-4 CReLU trunk -> 30x40x64,
#            track = w+cut merged cout 2, desc 32, seg k-bit.
#  "lidar" = MIRRORS the trained lidar net (sspax/lidar_ring.py): 3x1024
#            ring raster in METERS (in_div 1), 1D CReLU trunk -> 256x64,
#            track = w only (cout 1), desc 32, seg k-bit.
# --------------------------------------------------------------------------
def make_fixture(path, res="full", k_bits=16, seed=0):
    if res == "full":
        T = [dict(kind="conv", k=3, s=2, cin=1, cout=8),
             dict(kind="dw", k=3, s=1, cin=8, cout=8),
             dict(kind="conv", k=1, s=1, cin=8, cout=16),
             dict(kind="dw", k=3, s=2, cin=16, cout=16),
             dict(kind="conv", k=1, s=1, cin=16, cout=32),
             dict(kind="dw", k=3, s=2, cin=32, cout=32),
             dict(kind="conv", k=1, s=1, cin=32, cout=64)]
        A = [dict(kind="conv", k=1, s=1, cin=64, cout=2)]
        G = [dict(kind="dw", k=3, s=1, cin=64, cout=64),
             dict(kind="conv", k=1, s=1, cin=64, cout=128),
             dict(kind="conv", k=1, s=1, cin=128, cout=k_bits)]
        meta = dict(version=1, modality="vision", in_h=240, in_w=320,
                    in_ch=1, cell=8, act="relu", trunk=T, track=A, seg=G,
                    k_bits=k_bits, thresh=[0.0] * k_bits)
        D = []
    elif res == "rgb":
        # v2.2 MIRROR of the ADE-150 bottleneck encoder export shape
        # (sspax/vision/bottleneck_seg.py): full QVGA RGB, stride-8
        # relu trunk -> 30x40, desc-only (track/seg absent).
        T = [dict(kind="conv", k=3, s=2, cin=3, cout=16),
             dict(kind="conv", k=3, s=2, cin=16, cout=32),
             dict(kind="conv", k=3, s=2, cin=32, cout=64)]
        A = []
        D = [dict(kind="conv", k=1, s=1, cin=64, cout=32)]
        G = []
        meta = dict(version=2, modality="vision", in_h=240, in_w=320,
                    in_ch=3, in_div=255.0, cell=8, act="relu", trunk=T,
                    track=A, desc=D, desc_bits=32, seg=G, k_bits=0,
                    thresh=[])
    elif res == "half":
        T = [dict(kind="conv", k=3, s=2, cin=1, cout=16),
             dict(kind="conv", k=3, s=2, cin=32, cout=16),
             dict(kind="conv", k=3, s=1, cin=32, cout=32),
             dict(kind="conv", k=1, s=1, cin=64, cout=32)]
        A = [dict(kind="conv", k=1, s=1, cin=64, cout=2)]
        D = [dict(kind="conv", k=1, s=1, cin=64, cout=32)]
        G = [dict(kind="conv", k=1, s=1, cin=64, cout=k_bits)]
        meta = dict(version=2, modality="vision", in_h=120, in_w=160,
                    in_ch=1, in_div=255.0, cell=4, act="crelu", trunk=T,
                    track=A, desc=D, desc_bits=32, seg=G,
                    k_bits=k_bits, thresh=[0.0] * k_bits)
    else:                                       # lidar
        T = [dict(kind="conv1d", k=7, s=2, cin=3, cout=16),
             dict(kind="conv1d", k=7, s=2, cin=32, cout=16),
             dict(kind="conv1d", k=5, s=1, cin=32, cout=32),
             dict(kind="conv1d", k=1, s=1, cin=64, cout=32)]
        A = [dict(kind="conv1d", k=1, s=1, cin=64, cout=1)]
        D = [dict(kind="conv1d", k=1, s=1, cin=64, cout=32)]
        G = [dict(kind="conv1d", k=1, s=1, cin=64, cout=k_bits)]
        meta = dict(version=2, modality="lidar", in_h=3, in_w=1024,
                    in_ch=3, in_div=1.0, cell=4, act="crelu", trunk=T,
                    track=A, desc=D, desc_bits=32, seg=G,
                    k_bits=k_bits, thresh=[0.0] * k_bits)
    rng = np.random.default_rng(seed)
    arrays = {}
    for pre, LL in (("T", T), ("A", A), ("G", G), ("D", D)):
        for i, L in enumerate(LL):
            shp = ((L["cin"], L["k"], L["k"]) if L["kind"] == "dw"
                   else (L["cout"], L["cin"], L["k"]) if L["kind"] == "conv1d"
                   else (L["cout"], L["cin"], L["k"], L["k"]))
            nch = L["cin"] if L["kind"] == "dw" else L["cout"]
            fan = (L["k"] * L["cin"] if L["kind"] == "conv1d"
                   else L["k"] ** 2 * (1 if L["kind"] == "dw" else L["cin"]))
            arrays[f"{pre}{i}_w"] = rng.integers(
                -64, 65, shp).astype(np.int8)
            arrays[f"{pre}{i}_s"] = np.full(
                nch, 0.03 / np.sqrt(fan), np.float32)
            arrays[f"{pre}{i}_b"] = (0.01 * rng.standard_normal(nch)
                                     ).astype(np.float32)
    save_head(path, meta, arrays)
    return path


def _mmacs(meta):
    tot, h, w = {}, meta["in_h"], meta["in_w"]
    if meta["modality"] == "lidar":
        h = 1                        # rings are channels; azimuth = spatial
    for name in ("trunk", "track", "seg", "desc"):
        if not meta.get(name):       # raw metas may lack optional stacks
            continue
        m, hh, ww = 0, h, w          # heads start at trunk-output res
        for L in meta[name]:
            if L["kind"] == "conv1d":
                ww = ww // L["s"]
                m += ww * L["k"] * L["cin"] * L["cout"]
            else:
                hh, ww = hh // L["s"], ww // L["s"]
                mult = L["k"] ** 2 * (L["cin"] if L["kind"] != "dw" else 1)
                m += hh * ww * mult * (L["cout"] if L["kind"] != "dw" else
                                       L["cin"])
        tot[name] = m
        if name == "trunk":
            h, w = hh, ww
    return tot


def selftest():
    import tempfile
    import os
    # (fixture, trunk-grid, track cout, has desc)
    cases = (("full", (30, 40), 2, False),      # v1 back-compat
             ("half", (30, 40), 2, True),       # trained segnet mirror
             ("lidar", (256,), 1, True))        # trained lidar_ring mirror
    for res, hw, tc, has_d in cases:
        p = os.path.join(tempfile.gettempdir(), f"headio_{res}.npz")
        make_fixture(p, res, k_bits=16, seed=3)
        head = load_head(p)
        rng = np.random.default_rng(0)
        m = head["meta"]
        raw = (rng.integers(0, 256, (m["in_h"], m["in_w"])).astype(np.uint8)
               if m["modality"] == "vision" else
               rng.uniform(0, 60, (m["in_h"], m["in_w"])))   # ranges, METERS
        o1 = forward(head, raw)
        o2 = forward(load_head(p), raw)
        assert o1["track"].shape == hw + (tc,), o1["track"].shape
        assert o1["bits"].shape == hw + (16,)
        assert np.array_equal(o1["logits"], o2["logits"])   # determinism
        assert np.array_equal(o1["bits"], o2["bits"])
        if has_d:
            assert o1["desc"].shape == hw + (32,)
            assert np.array_equal(o1["desc"], o2["desc"])
        mm = _mmacs(m)
        fr = mm["trunk"] + mm["track"] + mm.get("desc", 0)
        if res == "full":
            cb = cell_bits(head, raw)
            assert cb.shape == (15, 20, 16)
            assert abs(fr / 1e6 - 11.3) < 0.2, fr           # budget line
        elif res == "half":
            cb = cell_bits(head, raw, source="desc")
            assert cb.shape == (15, 20, 32)
            # cross-impl budget check vs the trained segnet's own print
            # (22 MMAC trunk+track+desc, sspax/vision/segnet.py fpga_cost)
            assert abs(fr / 1e6 - 22.3) < 0.3, fr
        else:
            # lidar_ring fpga_cost prints 4.12 MMAC incl. its 40-class lab
            # head; with the k16 latent head instead: 3.47 + 0.26 @kf
            assert abs(fr / 1e6 - 3.47) < 0.1, fr
        print(f"  {res}: forward {hw} cells (track {tc}"
              f"{', desc 32' if has_d else ''}), deterministic, roundtrip "
              f"exact; @frame {fr/1e6:.1f} MMAC, seg {mm['seg']/1e6:.1f}")
    # v2.2: RGB desc-only head (the ADE bottleneck export shape) —
    # track absent, colour input, half-res auto-subsample
    import tempfile as _tf
    import os as _os
    p = _os.path.join(_tf.gettempdir(), "headio_rgb.npz")
    make_fixture(p, "rgb", seed=4)
    head = load_head(p)
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (240, 320, 3)).astype(np.uint8)
    o1 = forward(head, rgb)
    o2 = forward(load_head(p), rgb)
    assert "track" not in o1 and "logits" not in o1
    assert o1["desc"].shape == (30, 40, 32)
    assert np.array_equal(o1["desc"], o2["desc"])
    cb = cell_bits(head, rgb, source="desc")
    assert cb.shape == (15, 20, 32)
    big = rng.integers(0, 256, (480, 640, 3)).astype(np.uint8)
    assert cell_bits(head, big, source="desc").shape == (15, 20, 32)
    print("  rgb: v2.2 desc-only QVGA RGB head — forward/cell_bits/"
          "half-res subsample ok, no track stack")
    # desc-only head (seg deferred to the distillation round) must load
    p = os.path.join(tempfile.gettempdir(), "headio_lidar.npz")
    z = np.load(p)
    meta = json.loads(bytes(z["meta"]).decode())
    meta["seg"], meta["k_bits"], meta["thresh"] = [], 0, []
    arrays = {k: z[k] for k in z.files if k != "meta" and not
              k.startswith("G")}
    p2 = os.path.join(tempfile.gettempdir(), "headio_desconly.npz")
    save_head(p2, meta, arrays)
    head = load_head(p2)
    o = forward(head, np.random.default_rng(0).uniform(0, 60, (3, 1024)))
    assert "logits" not in o and o["desc"].shape == (256, 32)
    print("  desc-only head: loads, forwards (no seg stack)")
    print("headio selftest ok (v1 back-compat + v2 trained-net mirrors)")


# --------------------------------------------------------------------------
#  gate 1: cross-view bit stability (the raw-census killer)
# --------------------------------------------------------------------------
def bits_stability(head_path, seq=None, source="auto"):
    import experiments.objmap as OM
    import experiments.vision6d as V6
    head = load_head(head_path)
    if source == "auto":               # desc = the carrying head when present
        source = "desc" if head["desc"] else "seg"
    seq = seq or OM.SEQ
    feats, gt = OM.load_feats(seq)
    gray, _, _, _ = V6.load(seq)
    B = np.stack([cell_bits(head, g, source=source) for g in gray])
    pos = gt[:, :3]
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    adj, far = [], []
    rng = np.random.default_rng(5)
    for i in range(len(feats) - 1):
        for j, bucket in ((i + 1, adj), (None, far)):
            if j is None:
                cand = np.flatnonzero(d[i] > 2.0)
                if not len(cand):
                    continue
                j = int(rng.choice(cand))
            Fi, Fj = feats[i], feats[j]
            oki = np.argwhere(Fi["ok"])
            for cy, cx in oki[::7]:
                pw = Fi["pw"][cy, cx]
                dd = np.linalg.norm(Fj["pw"] - pw, axis=2)
                dd[~Fj["ok"]] = 9e9
                fl = int(np.argmin(dd))
                if dd.ravel()[fl] > 0.15:
                    continue
                y2, x2 = np.unravel_index(fl, dd.shape)
                bucket.append(
                    float((B[i, cy, cx] == B[j, y2, x2]).mean()))
    print(f"cross-view bit agreement ({seq.split('_freiburg')[-1]}, "
          f"{source} bits k={B.shape[-1]}; RANDOM-weights baseline 0.892 "
          f"adj / 0.828 far — read the gap, not the absolute):")
    print(f"  same cell adjacent frames: {np.mean(adj):.3f} (n={len(adj)})")
    print(f"  same cell far frames     : {np.mean(far):.3f} (n={len(far)})",
          flush=True)
    return float(np.mean(adj)), float(np.mean(far))


# --------------------------------------------------------------------------
#  gate 2: objmap2 semantic-key harness with head bits
# --------------------------------------------------------------------------
def _head_keys(kb, D):
    rng = np.random.default_rng(HEAD_KEY_SEED * 100_000 + D)
    return np.exp(1j * rng.uniform(0, 2 * np.pi, (kb, D)))


def _app_h(bits, keys):
    """(gh, gw, kb) bool -> (gh, gw, D) bipolar-spatter amplitudes."""
    kb = bits.shape[-1]
    sgn = 1.0 - 2.0 * bits.reshape(-1, kb)
    return (sgn @ keys / np.sqrt(kb)).reshape(bits.shape[:2] + (-1,))


def _frame_vec_h(F, app, W):
    # replicated from objmap.frame_vec with the appearance hook
    ok = F["ok"]
    ph = np.exp(1j * (F["pw"][ok] @ W.T))
    w = np.sqrt(F["w"][ok])
    a = app[ok]
    a = a - a.mean(0)
    v = (w[:, None] * ph * a).sum(0)
    return v / max(np.linalg.norm(v), 1e-12)


def _make_query_h(F, app, W, cy, cx, PATCH=2):
    # replicated from objmap.make_query with the appearance hook
    ok = F["ok"]
    ys, xs = np.mgrid[cy - PATCH:cy + PATCH + 1, cx - PATCH:cx + PATCH + 1]
    ys, xs = ys.ravel(), xs.ravel()
    keep = ok[ys, xs]
    ys, xs = ys[keep], xs[keep]
    if len(ys) < 6:
        return None
    pw = F["pw"][ys, xs]
    wgt = np.sqrt(F["w"][ys, xs])
    x0 = (pw * wgt[:, None]).sum(0) / wgt.sum()
    if np.linalg.norm(pw - x0, axis=1).max() > 0.5:
        return None
    ph = np.exp(1j * ((pw - x0) @ W.T))
    a = app[ys, xs]
    a = a - a.mean(0)
    q = (wgt[:, None] * ph * a).sum(0)
    return q, x0, len(ys)


def _mv_query_h(feats, apps, k, cy, cx, W, span=2, rad=0.4):
    # replicated from objmap2.mv_query with the appearance hook
    base = _make_query_h(feats[k], apps[k], W, cy, cx)
    if base is None:
        return None
    _, x0, n0 = base
    q = np.zeros(len(W), complex)
    ntot = 0
    for kk in (k - span, k, k + span):
        if kk < 0 or kk >= len(feats):
            continue
        F = feats[kk]
        m = F["ok"] & (np.linalg.norm(F["pw"] - x0, axis=2) < rad)
        if m.sum() < 3:
            continue
        pw = F["pw"][m]
        wgt = np.sqrt(F["w"][m])
        a = apps[kk][m]
        a = a - a.mean(0)
        q += (wgt[:, None] * np.exp(1j * ((pw - x0) @ W.T)) * a).sum(0)
        ntot += int(m.sum())
    return q, x0, ntot


def gate_objmap(head_path, seq=None, nq=48, topk=3, key="seg"):
    # nq=48 drawn -> ~16 kept matches the banked objmap2 protocol.
    # key="seg" binds the k-bit label latent (the semantic key);
    # key="desc" binds the tracking-descriptor bits (appearance key —
    # the head the trained nets actually carry).
    import experiments.objmap as OM
    import experiments.objmap2 as O2
    import experiments.twomap as TM
    import experiments.lattice3d as L3
    import experiments.vision6d as V6
    head = load_head(head_path)
    assert key == "seg" or head["desc"], "no desc head in this file"
    seq = seq or OM.SEQ
    rng = np.random.default_rng(OM.RNG)
    feats, gt = OM.load_feats(seq)
    gray, depth, _, K = V6.load(seq)
    map_kf, qry_kf = OM.split(len(feats))
    W = OM.W_map(960)
    kb = head["meta"]["k_bits"] if key == "seg" else head["meta"]["desc_bits"]
    keys = _head_keys(kb, len(W))
    apps = [_app_h(cell_bits(head, g, source=key), keys) for g in gray]
    groups = [list(map_kf[i:i + OM.SEG])
              for i in range(0, len(map_kf), OM.SEG)]
    maps = [(g, sum(_frame_vec_h(feats[k], apps[k], W) for k in g))
            for g in groups]
    boxes = O2.seg_bboxes(feats, [g for g, _ in maps])
    bbox_room = OM.world_bbox(feats, map_kf)
    Wv = TM.W_vis_fib()
    snaps = {k: O2.snap_vec(gray[k], K, Wv) for k in map_kf}
    kf2seg = {}
    for si, (g, _) in enumerate(maps):
        for k in g:
            kf2seg[k] = si
    queries = OM.pick_queries(feats, qry_kf, rng, n=nq)

    def stage1(g, Kc):
        vq = O2.snap_vec(g, Kc, Wv)
        sims = {}
        for k, v in snaps.items():
            si = kf2seg[k]
            sims[si] = max(sims.get(si, 0.0), abs(np.vdot(vq, v)))
        order = sorted(sims, key=lambda s: -sims[s])
        return order[:topk], sims[order[0]]

    def decode_scoped(q, segs):
        best = None
        for si in segs:
            x, sc = OM.decode([maps[si]], q, W, boxes[si])
            if best is None or sc > best[1]:
                best = (x, sc)
        return best

    arms = {t: dict(err=[], sc=[], s1=[], hit=0, n=0)
            for t in ("2stage-1view", "2stage-3view")}
    for k, cy, cx in queries:
        mq = _make_query_h(feats[k], apps[k], W, cy, cx)
        mv = _mv_query_h(feats, apps, k, cy, cx, W)
        if mq is None or mv is None:
            continue
        q1, x_gt, n1 = mq
        qm, _, nm = mv
        segs, s1 = stage1(gray[k], K)
        hit = any(np.all(x_gt > boxes[si][0]) and np.all(x_gt < boxes[si][1])
                  for si in segs)
        for tag, q, n in (("2stage-1view", q1, n1),
                          ("2stage-3view", qm, nm)):
            x, sc = decode_scoped(q, segs)
            a = arms[tag]
            a["err"].append(np.linalg.norm(x - x_gt))
            a["sc"].append(sc / max(n, 1))
            a["s1"].append(s1)
            a["hit"] += hit
            a["n"] += 1
    ff, _ = OM.load_feats(OM.FOIL_SEQ)
    fgray, _, _, fK = V6.load(OM.FOIL_SEQ)
    fapps = [_app_h(cell_bits(head, g, source=key), keys) for g in fgray]
    fq = OM.pick_queries(ff, np.arange(len(ff)), rng, n=nq)
    foil = {t: dict(sc=[], s1=[]) for t in arms}
    for k, cy, cx in fq:
        mq = _make_query_h(ff[k], fapps[k], W, cy, cx)
        mv = _mv_query_h(ff, fapps, k, cy, cx, W)
        if mq is None or mv is None:
            continue
        segs, s1 = stage1(fgray[k], fK)
        for tag, (q, _, n) in (("2stage-1view", mq), ("2stage-3view", mv)):
            _, sc = decode_scoped(q, segs)
            foil[tag]["sc"].append(sc / max(n, 1))
            foil[tag]["s1"].append(s1)
    print(f"objmap2 semantic-key gate ({seq.split('_freiburg')[-1]}, "
          f"{key} k={kb} head bits -> bipolar spatter D{len(W)}, "
          f"{arms['2stage-1view']['n']} queries, top-{topk}; banked refs: "
          f"int combined 0.805, census 3-view err 0.165 m):")
    a0 = arms["2stage-1view"]
    print(f"  stage-1 shortlist hit rate: {a0['hit']}/{a0['n']}")
    for tag in arms:
        a = arms[tag]
        e = np.array(a["err"])
        sr, sf = np.array(a["sc"]), np.array(foil[tag]["sc"])
        s1r, s1f = np.array(a["s1"]), np.array(foil[tag]["s1"])
        auc = L3._auc(sr, sf)
        mu, sd = np.concatenate([sr, sf]).mean(), \
            max(np.concatenate([sr, sf]).std(), 1e-9)
        m1, d1 = np.concatenate([s1r, s1f]).mean(), \
            max(np.concatenate([s1r, s1f]).std(), 1e-9)
        zc = L3._auc((sr - mu) / sd + (s1r - m1) / d1,
                     (sf - mu) / sd + (s1f - m1) / d1)
        print(f"  {tag}: err med {np.median(e):.3f} p90 "
              f"{np.percentile(e, 90):.3f} | AUC stage2 {auc:.3f} "
              f"combined {zc:.3f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        selftest()
    elif cmd == "fixture":
        out = sys.argv[2] if len(sys.argv) > 2 else "scratch/head_fixture.npz"
        res = sys.argv[3] if len(sys.argv) > 3 else "full"
        kb = int(sys.argv[4]) if len(sys.argv) > 4 else 16
        print("wrote", make_fixture(out, res, kb))
    elif cmd == "stability":
        bits_stability(sys.argv[2],
                       sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "gate":
        gate_objmap(sys.argv[2],
                    sys.argv[3] if len(sys.argv) > 3 else None,
                    key=sys.argv[4] if len(sys.argv) > 4 else "seg")
