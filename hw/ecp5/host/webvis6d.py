#!/usr/bin/env python3
"""webvis6d — 6-DoF FPGA-SIM WEB VISUALIZATION (TUM dataset replay).

The 3D/6-DoF half of the deploy target (user 2026-07-16: "opt both 2D
3DoF and 3D 6DoF"), replayed from the TUM RGB-D sets in data/tum with
the BANKED deploy6d recipe — every op is one the chip formulation
already carries:

  ego-motion   two-space block-stacked GN (deploy6d._gn_stacked; the
               fuse2/chain winner): vision gridint-3D points in the
               camera-range ladder (W_vis3d) + depth cloud in azel3d,
               spaces distinct, ONE 6x6 solve; frame-chained.
  anchors      bounded map of QUANTIZED snapshots at novelty poses
               (>0.5 m / >20 deg from every anchor — the v8 novelty
               law in 6-DoF): per channel v0 + the 6 derivative
               vectors, stored at nph (deploy6d.qvec/qstack — the
               2b/3b phase-only FPGA store model, bench_verify).
  verify       near an anchor (est-relative), the stored-anchor 6x6
               solve (deploy6d.solve66) against the QUANTIZED (v0, D)
               re-registers the pose to the anchor — drift repair
               from the frozen store, the SOLO commit analog.

Recipes: "quantized 2b (FPGA sim)" nph=4 store | "float" nph=0 — the
same A/B the 2D webvis carries. Mocap GT is a DISPLAY GHOST + scoring
readout only (anti-oracle; it never enters the pipeline — the est
starts AT gt[0] so the ghost overlays without alignment).

  python3 hw/ecp5/host/webvis6d.py selftest   # numbers only
  python3 hw/ecp5/host/webvis6d.py serve [port=8791]
"""
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import experiments.deploy6d as D6                              # noqa: E402
import experiments.lattice3d as L3                             # noqa: E402
import experiments.lidar6d as L6                               # noqa: E402
import experiments.vision6d as V6                              # noqa: E402

TUM = ROOT / "data" / "tum"
SEQS = {p.stem.split("dataset_")[-1]: p.stem
        for p in sorted(TUM.glob("rgbd_dataset_*.npz"))}
DEFAULT = "freiburg3_long_office_household_s2"   # 15 Hz: chain regime
NOV_T, NOV_R = 0.5, 20.0            # anchor novelty (m, deg)
VER_T, VER_R = 0.35, 10.0           # verify range (est-relative)
CLAMP_R, CLAMP_T = 8.0, 0.15        # verify correction clamp (deg, m)
GN_W = [1 / 1.82, 1 / 1.34]         # banked fixed-precision weights


def rotmat_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


class Track6D:
    """The replayed 6-DoF pipeline (one dataset, one recipe)."""

    def __init__(self, seq=DEFAULT, nph=4):
        z = np.load(TUM / f"{SEQS[seq]}.npz")
        self.gray = z["gray"]
        self.depth = z["depth_mm"]
        self.gt = z["gt"]
        self.K = z["K"]
        self.n = len(self.gray)
        self.nph = nph
        self.Wv = D6.W_vis3d()
        self.Wl = L3.make_lattices()["azel3d"]
        self.C = np.zeros((self.n, 3, 3))       # est world<-cam
        self.c = np.zeros((self.n, 3))
        self.Cg = np.stack([V6.quat_R(q) for q in self.gt[:, 3:7]])
        self.cg = self.gt[:, :3].copy()
        self.anchors = []       # (k, C, c, per-channel (W, P0, w0,
        self.prev = None        #  v0_q, D_q))
        self.n_verify = 0
        self.n_rescue = 0
        self.rot_step_errs = []
        self.pos_errs = []
        self._cv = (np.eye(3), np.zeros(3))     # last step (CV seed)

    # ---- channels -----------------------------------------------------
    def _chans(self, k):
        Fv = V6.feats(self.gray[k], self.K, "gridint", self.depth[k])
        Pc, wc = L6.depth_cloud(self.depth[k], self.K)
        return [(self.Wv, Fv[0], Fv[1]), (self.Wl, Pc, wc)]

    def _store_anchor(self, k, chans):
        """FPGA store model: v0 + 6 derivative vectors per channel,
        phase-quantized at nph (bench_verify semantics)."""
        st = []
        for (W, P0, w0) in chans:
            v0 = V6._enc_raw(W, P0, w0)
            Dm = np.concatenate([V6._deriv_axes(W, P0, w0),
                                 V6._deriv_transl(W, P0, w0)], 1)
            st.append((W, P0, w0, D6.qvec(v0, self.nph),
                       D6.qstack(Dm, self.nph)))
        self.anchors.append((k, self.C[k].copy(), self.c[k].copy(), st))

    def _novel(self, k):
        for (_, Ca, ca, _) in self.anchors:
            if np.linalg.norm(self.c[k] - ca) < NOV_T and \
               rotmat_deg(Ca.T @ self.C[k]) < NOV_R:
                return False
        return True

    def _verify(self, k, chans):
        """Nearest stored anchor within (VER_T, VER_R) of the est ->
        quantized-anchor 6x6 solve -> re-register (clamped)."""
        best = None
        for ai, (ka, Ca, ca, st) in enumerate(self.anchors):
            d = np.linalg.norm(self.c[k] - ca)
            if d < VER_T and rotmat_deg(Ca.T @ self.C[k]) < VER_R:
                if best is None or d < best[0]:
                    best = (d, ai)
        if best is None:
            return False
        _, ai = best
        ka, Ca, ca, st = self.anchors[ai]
        ths = []
        for (W, P0, w0, v0q, Dq), (Wc, P1, w1) in zip(st, chans):
            v1 = V6._enc_raw(W, P1, w1)
            ths.append(D6.solve66(v0q, Dq, v1))
        th = np.mean(ths, 0)
        rot = np.degrees(np.linalg.norm(th[:3]))
        if rot > CLAMP_R or np.linalg.norm(th[3:6]) > CLAMP_T:
            return False                        # out of the linear model
        dR = L3._axis_rot("yaw", np.degrees(th[0])) \
            @ L3._axis_rot("pitch", np.degrees(th[1])) \
            @ L3._axis_rot("roll", np.degrees(th[2]))
        # anchor-frame relative pose (p_now = dR p_anchor + th[3:6])
        self.C[k] = Ca @ dR.T
        self.c[k] = ca - self.C[k] @ th[3:6]
        self.n_verify += 1
        return True

    # ---- one frame ----------------------------------------------------
    def step(self, k):
        chans = self._chans(k)
        verified = False
        if k == 0:
            self.C[0] = self.Cg[0]
            self.c[0] = self.cg[0]              # start AT gt[0] (ghost
        else:                                   # overlays; scoring only)
            v1s = [V6._enc_raw(W, P, w)
                   for (W, P, w) in self._prev_chans]
            # CV seed: handheld/robot motion is smooth — start GN at
            # the LAST step (large TUM steps sit outside the identity
            # linearization basin: fr1_desk 7-12 deg/frame)
            R0, t0 = self._cv
            R, t = D6._gn_stacked(chans, v1s, R0.copy(), t0.copy(),
                                  GN_W, iters=3)
            # rescue: if the fit exploded, re-seed from a small
            # rotation ball around CV (the banked coarse-stage move)
            if rotmat_deg(R0.T @ R) > 15.0 or \
                    np.linalg.norm(t - t0) > 0.35:
                best = None
                for w in D6._ball_grid(40, 12.0):
                    Rc = D6._R_of_w(w) @ R0
                    Rr, tr = D6._gn_stacked(chans, v1s, Rc,
                                            t0.copy(), GN_W, iters=2)
                    r2 = 0.0
                    for (W, P0, w0), v1, wt in zip(chans, v1s, GN_W):
                        vc = V6._enc_raw(W, P0 @ Rr.T + tr, w0)
                        r2 += wt * np.linalg.norm(v1 - vc) \
                            / max(np.linalg.norm(v1), 1e-9)
                    if best is None or r2 < best[0]:
                        best = (r2, Rr, tr)
                _, R, t = best
                self.n_rescue += 1
            self._cv = (R.copy(), t.copy())
            # (R, t) = rel pose prev<-cur in camera frame (v1 = prev):
            # p_prev = R p_cur + t  ->  C_cur = C_prev R, c_cur =
            # c_prev + C_prev t
            self.C[k] = self.C[k - 1] @ R
            self.c[k] = self.c[k - 1] + self.C[k - 1] @ t
            Rg, tg = V6._rel_pose(self.gt[k], self.gt[k - 1])
            self.rot_step_errs.append(
                np.degrees(np.linalg.norm(D6.rotvec(R @ Rg.T))))
            verified = self._verify(k, chans)
        if self._novel(k):
            self._store_anchor(k, chans)
        self._prev_chans = chans
        self.pos_errs.append(
            float(np.linalg.norm(self.c[k] - self.cg[k])))
        return verified

    def cloud_world(self, k, sub=5):
        Pc, _ = L6.depth_cloud(self.depth[k], self.K)
        P = Pc[::sub] @ self.C[k].T + self.c[k]
        return P

    def anchor_bytes(self):
        """store cost per anchor at the current recipe (phase bits x
        (v0 + 6 deriv) x 2 channels + pose)."""
        if self.nph == 0:
            return (240 * 7 * 2 * 8) + 28       # c64 float store
        bits = 2 if self.nph <= 4 else 3
        return (240 * 7 * 2 * bits) // 8 + 28


class Demo6:
    def __init__(self, seq=DEFAULT, nph=4):
        self.seq = seq
        self.nph = nph
        self.trk = Track6D(seq, nph)
        self.k = 0
        self.done = False
        self.clients = []
        self.lock = threading.Lock()
        self._req = None

    def broadcast(self, obj):
        dead = []
        for q_ in self.clients:
            try:
                q_.put_nowait(obj)
            except queue.Full:
                dead.append(q_)
        for q_ in dead:
            self.clients.remove(q_)

    def snapshot(self):
        t = self.trk
        return dict(init=True, seq=self.seq, n=t.n, k=self.k,
                    seqs=sorted(SEQS), nph=self.nph,
                    gt=[[round(float(x), 3) for x in p]
                        for p in t.cg[:self.k]],
                    est=[[round(float(x), 3) for x in p]
                         for p in t.c[:self.k]])

    def step(self):
        t0 = time.time()
        trk = self.trk
        k = self.k
        verified = trk.step(k)
        P = trk.cloud_world(k, sub=6)
        st = dict(k=k, n=trk.n, seq=self.seq, nph=self.nph,
                  est=[round(float(x), 3) for x in trk.c[k]],
                  gt=[round(float(x), 3) for x in trk.cg[k]],
                  pos_err=round(trk.pos_errs[-1], 3),
                  rot_step=round(float(np.median(
                      trk.rot_step_errs[-20:])), 2)
                  if trk.rot_step_errs else 0.0,
                  verified=bool(verified),
                  n_anchor=len(trk.anchors),
                  n_verify=trk.n_verify, n_rescue=trk.n_rescue,
                  bytes_anchor=trk.anchor_bytes(),
                  ms=round((time.time() - t0) * 1e3),
                  pts=np.round(P, 3).tolist())
        anc = [[round(float(x), 3) for x in a[2]] for a in trk.anchors]
        st["anchors"] = anc
        self.k += 1
        if self.k >= trk.n:
            st["done"] = True
            self.done = True
        self.broadcast(st)

    def run(self):
        while True:
            if self._req:
                (seq, nph), self._req = self._req, None
                self.seq, self.nph = seq, nph
                self.trk = Track6D(seq, nph)
                self.k = 0
                self.done = False
                self.broadcast(dict(reset=True, seq=seq, nph=nph))
            if not self.done:
                self.step()
            else:
                time.sleep(1.0)


def make_handler(demo):
    html = (HERE / "webvis6d.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path.startswith("/select"):
                q = self.path.split("?", 1)[-1]
                kv = dict(p.split("=") for p in q.split("&") if "=" in p)
                seq = kv.get("seq", demo.seq)
                nph = int(kv.get("nph", demo.nph))
                if seq in SEQS and nph in (0, 4, 8):
                    demo._req = (seq, nph)
                self._json(dict(ok=True))
            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html)
            elif self.path == "/status":
                t = demo.trk
                self._json(dict(k=demo.k, n=t.n, seq=demo.seq,
                                nph=demo.nph, anchors=len(t.anchors),
                                verifies=t.n_verify,
                                pos_err_med=round(float(np.median(
                                    t.pos_errs)), 3)
                                if t.pos_errs else None))
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q_ = queue.Queue(maxsize=32)
                demo.clients.append(q_)
                try:
                    self.wfile.write(
                        f"data: {json.dumps(demo.snapshot())}\n\n"
                        .encode())
                    self.wfile.flush()
                    while True:
                        obj = q_.get(timeout=30)
                        self.wfile.write(
                            f"data: {json.dumps(obj)}\n\n".encode())
                        self.wfile.flush()
                except (queue.Empty, BrokenPipeError,
                        ConnectionResetError):
                    pass
                finally:
                    if q_ in demo.clients:
                        demo.clients.remove(q_)
            else:
                self.send_response(404)
                self.end_headers()

    return H


def serve(port=8791):
    import os
    demo = Demo6(seq=os.environ.get("SSP_SEQ6", DEFAULT),
                 nph=int(os.environ.get("SSP_NPH6", "4")))
    threading.Thread(target=demo.run, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(demo))
    print(f"[webvis6d] http://127.0.0.1:{port}  seq={demo.seq} "
          f"recipe={'2b FPGA store' if demo.nph == 4 else 'float'}",
          flush=True)
    srv.serve_forever()


def selftest(n=60):
    """Numbers only: fr1_desk first n frames, BOTH recipes; assert the
    tracker holds and the quantized store stays in family."""
    for nph in (0, 4):
        trk = Track6D(DEFAULT, nph)
        t0 = time.time()
        for k in range(min(n, trk.n)):
            trk.step(k)
        e = np.array(trk.pos_errs)
        r = np.array(trk.rot_step_errs)
        print(f"  nph={nph}: {min(n, trk.n)} kf | pos err vs mocap med "
              f"{np.median(e):.3f} max {e.max():.3f} m | rot step med "
              f"{np.median(r):.2f} deg | anchors {len(trk.anchors)} "
              f"({trk.anchor_bytes()} B) | verifies {trk.n_verify} "
              f"rescues {trk.n_rescue} | "
              f"{(time.time() - t0) / min(n, trk.n) * 1e3:.0f} ms/kf",
              flush=True)
        assert np.median(r) < 1.5, "rot step out of family"
        assert np.median(e) < 0.5, "pos err out of family"
        assert trk.n_verify > 0, "verify never fired"
    print("WEBVIS6D SELFTEST PASS")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "selftest":
        selftest(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
    else:
        serve(int(sys.argv[2]) if len(sys.argv) > 2 else 8791)
