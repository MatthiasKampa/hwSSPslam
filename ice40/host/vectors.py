#!/usr/bin/env python3
"""Golden-vector harness: golden model (ssp_ice40) vs simulation vs hardware.

  python3 ice40/host/vectors.py gen [n]     # build/points.hex + golden
  python3 ice40/host/vectors.py sim         # iverilog run, bit-exact compare
  python3 ice40/host/vectors.py hw [port]   # board run, bit-exact compare
  python3 ice40/host/vectors.py hw-scan     # full real SPOT scan on hw
"""
import glob
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ssp_ice40 as G                                    # noqa: E402

ICE = ROOT / "ice40"
BUILD = ICE / "build"
TOOLS = Path.home() / "tools" / "oss-cad-suite" / "bin"


def test_points(n=64, seed=5):
    """Deterministic mixed-magnitude test vector (edge cases included)."""
    rng = np.random.default_rng(seed)
    az = rng.integers(0, 1024, n).astype(np.int32)
    r = rng.integers(300, 31200, n).astype(np.int32)
    w = rng.integers(1, 128, n).astype(np.int32)
    az[:4] = [0, 1023, 512, 256]            # wrap/edge cases
    r[:4] = [300, 31200, 15000, 999]
    w[:4] = [127, 1, 64, 127]
    return az, r, w


def spot_points(k=0):
    import ssp_datasets as DS
    b = DS.load("spot", cap=k + 1)
    rr = DS.clean(b, b["keys"][k][0])
    return G.scan_to_ints(rr)


def write_vec(az, r, w):
    BUILD.mkdir(exist_ok=True)
    with open(BUILD / "points.hex", "w") as f:
        for a, rr, ww in zip(az, r, w):
            f.write(f"{a:04x}\n{rr:04x}\n{ww:04x}\n")
    acc = G.encode_int(az, r, w, G.make_luts())
    np.save(BUILD / "golden.npy", acc)
    return acc


def compare(acc_hw, tag):
    gold = np.load(BUILD / "golden.npy")
    same = np.array_equal(acc_hw, gold)
    if same:
        print(f"{tag}: BIT-EXACT ({gold.shape[0]} components, "
              f"max|acc| {np.abs(gold).max()})")
        return True
    d = np.abs(acc_hw.astype(np.int64) - gold.astype(np.int64))
    bad = np.flatnonzero(d.max(1))
    print(f"{tag}: MISMATCH on {len(bad)}/{len(gold)} components; "
          f"first {bad[:6].tolist()} max|d| {d.max()}")
    for i in bad[:4]:
        print(f"   [{i}] hw {acc_hw[i].tolist()}  gold {gold[i].tolist()}")
    return False


def gen(n=64):
    az, r, w = test_points(int(n))
    acc = write_vec(az, r, w)
    print(f"wrote {len(az)} points + golden (max|acc| {np.abs(acc).max()})")


def sim(encfile="rtl/encoder.v", tbfile="rtl/tb_encoder.v"):
    cells = TOOLS.parent / "share" / "yosys" / "ice40" / "cells_sim.v"
    subprocess.run([str(TOOLS / "iverilog"), "-g2012", "-o",
                    str(BUILD / "tb.vvp"), encfile,
                    tbfile, str(cells)], cwd=ICE, check=True)
    out = subprocess.run([str(TOOLS / "vvp"), str(BUILD / "tb.vvp")],
                         cwd=ICE, check=True, capture_output=True, text=True)
    acc = np.zeros((240, 2), np.int64)
    for line in out.stdout.splitlines():
        if line.startswith("ACC "):
            _, i, re, im = line.split()
            acc[int(i)] = [int(re), int(im)]
    sys.exit(0 if compare(acc.astype(np.int32), "sim") else 1)


class SweepError(Exception):
    """A paced sweep lost replies twice — protocol resync required."""


class Board:
    def __init__(self, port=None, baud=3_000_000):
        import serial
        if not port:
            c = sorted(glob.glob("/dev/tty.usbserial-*1"))
            port = c[-1]
        self.s = serial.Serial(port, baud, timeout=2.0)
        self.s.reset_input_buffer()

    def clear(self):
        self.s.write(b"\x01")
        assert self.s.read(1) == b"\x2B", "clear ack missing"

    def point(self, az, r, w):
        self.s.write(bytes([2, az & 255, (az >> 8) & 3, r & 255,
                            (r >> 8) & 255, w & 127]))

    def stream(self, az, r, w, chunk=128):   # chunk <= fifo(256) - drain slack
        """Send points; collect one 0x2A ack per point (pipelined by chunk)."""
        n = len(az)
        for i0 in range(0, n, chunk):
            sl = slice(i0, min(i0 + chunk, n))
            msg = b"".join(bytes([2, a & 255, (a >> 8) & 3, rr & 255,
                                  (rr >> 8) & 255, ww & 127])
                           for a, rr, ww in zip(az[sl], r[sl], w[sl]))
            self.s.write(msg)
            k = min(i0 + chunk, n) - i0
            acks = self.s.read(k)
            assert acks == b"\x2A" * k, f"acks {acks.hex()} at {i0}"

    def stream_bulk(self, az, r, w, n0=0):
        """One write, no per-point acks (cmd 0x05); poll the counter until
        all points are folded. Encoder outruns 3 Mbaud, FIFO never fills."""
        n = len(az)
        msg = b"".join(bytes([5, a & 255, (a >> 8) & 3, rr & 255,
                              (rr >> 8) & 255, ww & 127])
                       for a, rr, ww in zip(az, r, w))
        self.s.write(msg)
        for _ in range(50):
            got = self.npts()
            if got == n0 + n:
                return
        raise AssertionError(f"bulk stream: counter {got} != {n0 + n}")

    def readback(self):
        self.s.write(b"\x03")
        raw = self.s.read(240 * 8)
        assert len(raw) == 1920, f"short readback {len(raw)}"
        v = np.frombuffer(raw, "<i4").reshape(240, 2)
        return v.astype(np.int32)

    def npts(self):
        self.s.write(b"\x04")
        lo, hi = self.s.read(2)
        return lo | (hi << 8)

    def load_m(self, mcode):
        """240 QPSK codes (component order k*60+j) -> cmd 0x06."""
        mcode = np.asarray(mcode, np.int64) & 3
        assert len(mcode) == 240
        by = bytes(int(mcode[i] | (mcode[i + 1] << 2) | (mcode[i + 2] << 4)
                       | (mcode[i + 3] << 6)) for i in range(0, 240, 4))
        self.s.write(b"\x06" + by)
        ack = self.s.read(1)
        assert ack == b"\x2C", f"M-load ack {ack.hex()}"

    def match(self, dx, dy, rho, sh):
        """cmd 0x07 -> (4, 2) i32 per-ring partial sums."""
        self.s.write(bytes([7, dx & 255, (dx >> 8) & 255, dy & 255,
                            (dy >> 8) & 255, rho & 63, sh & 15]))
        raw = self.s.read(32)
        assert len(raw) == 32, f"short match reply {len(raw)}"
        return np.frombuffer(raw, "<i4").reshape(4, 2).astype(np.int32)

    @staticmethod
    def _mcmd(dx, dy, rho, sh):
        return bytes([7, dx & 255, (dx >> 8) & 255, dy & 255,
                      (dy >> 8) & 255, rho & 63, sh & 15])

    def match_sweep(self, cands):
        """Pipelined candidate sweep — host-only protocol upgrade, no RTL
        change: the always-listening parser buffers ONE pre-fed command
        while the exec FSM streams the current reply, so we keep exactly
        one command in flight ahead of the reply stream. Replies then
        arrive back-to-back (fabric-paced, ~127 us/cand at 3 Mbaud) and
        pack the FTDI buffer instead of eating a 16 ms latency-timer
        stall per candidate. Window discipline: send cmd k+1 only after
        reply k's first byte (guarantees k+1 was dispatched-or-buffered,
        never overwritten)."""
        n = len(cands)
        out = np.zeros((n, 4, 2), np.int32)
        self.s.write(self._mcmd(*cands[0]))
        for k in range(n):
            head = self.s.read(1)               # reply k begins
            assert len(head) == 1, f"timeout at candidate {k}"
            if k + 1 < n:
                self.s.write(self._mcmd(*cands[k + 1]))
            raw = head + self.s.read(31)
            assert len(raw) == 32, f"short reply at {k}"
            out[k] = np.frombuffer(raw, "<i4").reshape(4, 2)
        return out

    def match_sweep_paced(self, cands, period=250e-6):
        """Time-paced sweep: the fabric is DETERMINISTIC (reply stream =
        106.7 us, match = 20.1 us at 24 MHz), so pacing writes at a fixed
        period > reply-duration keeps at most one undispatched command in
        the parser regs (the overwrite invariant) with NO reads in the
        loop — replies pack the FTDI buffer back-to-back and the 16 ms
        latency timer never engages. A short final read means the period
        was too aggressive (a command got overwritten) — fail loudly."""
        n = len(cands)
        msgs = [self._mcmd(*c) for c in cands]
        t0 = time.perf_counter()
        for k, m in enumerate(msgs):
            while time.perf_counter() - t0 < k * period:
                pass
            self.s.write(m)
        raw = self.s.read(32 * n)
        assert len(raw) == 32 * n, \
            f"short sweep read {len(raw)}/{32 * n} — period {period} too fast"
        return np.frombuffer(raw, "<i4").reshape(n, 4, 2).astype(np.int32)

    def match_sweep_safe(self, cands, period=250e-6):
        """match_sweep_paced with one backoff retry: OS write-coalescing
        can bunch commands past the parser's one-in-flight invariant
        (observed ~1/10^5 candidates). On double failure raises
        SweepError — callers should resync the protocol and re-encode
        (the root fix is the fabric-side candidate FIFO, roadmap)."""
        try:
            return self.match_sweep_paced(cands, period=period)
        except AssertionError:
            time.sleep(0.05)
            self.s.reset_input_buffer()
            try:
                return self.match_sweep_paced(cands, period=2 * period)
            except AssertionError as e:
                raise SweepError(str(e))

    def match_batch(self, cands, totals=False, chunk=120):
        """cmd 0x08 (v5 tops): candidates buffered in the fabric FIFO —
        one write per chunk, replies stream back-to-back, NO host pacing
        invariant. totals=True -> 4-byte ring-summed Re totals (the
        deployed argmax criterion); else full 32-byte per-ring replies."""
        out_t = np.zeros(len(cands), np.int64)
        out_f = np.zeros((len(cands), 4, 2), np.int32)
        for c0 in range(0, len(cands), chunk):
            cc = cands[c0:c0 + chunk]
            msg = bytes([8, 1 if totals else 0, len(cc) & 255,
                         (len(cc) >> 8) & 1])
            msg += b"".join(self._mcmd(*c)[1:] for c in cc)
            self.s.write(msg)
            need = len(cc) * (4 if totals else 32)
            raw = self.s.read(need)
            assert len(raw) == need, \
                f"batch short read {len(raw)}/{need} at {c0}"
            if totals:
                out_t[c0:c0 + len(cc)] = np.frombuffer(raw, "<i4")
            else:
                out_f[c0:c0 + len(cc)] = np.frombuffer(
                    raw, "<i4").reshape(-1, 4, 2)
        return out_t if totals else out_f

    def match_argmax(self, cands, chunk=120):
        """cmd 0x08 mode=2 (v6 tops): ON-FABRIC argmax over the batch —
        6-byte reply per chunk (idx u16 LE + best total i32 LE); host
        merges chunks (strict >, earlier chunk wins ties == np.argmax
        over the full send order)."""
        best, bidx = None, -1
        for c0 in range(0, len(cands), chunk):
            cc = cands[c0:c0 + chunk]
            msg = bytes([8, 2, len(cc) & 255, (len(cc) >> 8) & 1])
            msg += b"".join(self._mcmd(*c)[1:] for c in cc)
            self.s.write(msg)
            raw = self.s.read(6)
            assert len(raw) == 6, f"argmax short reply {len(raw)}"
            idx = raw[0] | (raw[1] << 8)
            tot = int(np.frombuffer(raw[2:6], "<i4")[0])
            if best is None or tot > best:
                best, bidx = tot, c0 + idx
        return bidx, best

    def resync(self):
        """Recover a possibly mid-frame parser: complete any partial
        frame with padding (may fire one garbage match and/or push a
        garbage point — both neutralized by the caller re-clearing and
        re-encoding), drain, verify with a counter echo."""
        time.sleep(0.2)
        self.s.reset_input_buffer()
        self.s.write(b"\x00" * 8)
        time.sleep(0.25)
        self.s.reset_input_buffer()
        return self.npts()


# ---------------------------------------------------------------- matcher
def synth_room_scan():
    """The selftest_match scene: a realistic-envelope 1024-beam scan."""
    rng = np.random.default_rng(7)
    a = np.arange(G.N_BEAM) * 2 * np.pi / G.N_BEAM - np.pi
    r = 3.5 / np.maximum(np.abs(np.cos(a)), 0.3)
    r += rng.normal(0, 0.02, G.N_BEAM)
    return G.scan_to_ints(r), rng


def match_vectors():
    """Scan + M codes + candidate list + golden partials (match_int)."""
    (az, r_mm, w), rng = synth_room_scan()
    luts = G.make_luts()
    Q = G.encode_int(az, r_mm, w, luts)
    mc = G.mcode_from_vec(G.encode_float_ref(az, r_mm, w))
    mc = mc.copy()
    mc[120:] = rng.integers(0, 4, 120)      # adversarial random half
    sh = G.shift_for(Q)
    cands = [(dx, dy, 0, sh) for dx in (-2048, -64, 0, 33, 511)
             for dy in (-1500, 0, 97)]
    cands += [(100, -50, rho, sh) for rho in (1, 13, 30, 45, 59)]
    cands += [(0, 0, 0, min(sh + 2, 15)), (8000, -8000, 7, sh)]
    gold = np.stack([G.match_int(Q, mc, dx, dy, rho, s, luts)
                     for dx, dy, rho, s in cands])
    return (az, r_mm, w), mc, cands, gold


def gen_match():
    (az, r_mm, w), mc, cands, gold = match_vectors()
    write_vec(az, r_mm, w)
    with open(BUILD / "mcodes.hex", "w") as f:
        for v in mc:
            f.write(f"{v:04x}\n")
    with open(BUILD / "mcmds.hex", "w") as f:
        for dx, dy, rho, s in cands:
            f.write(f"{dx & 0xFFFF:04x}\n{dy & 0xFFFF:04x}\n"
                    f"{rho:04x}\n{s:04x}\n")
    np.save(BUILD / "golden_match.npy", gold)
    print(f"wrote {len(az)} pts, {len(cands)} candidates, "
          f"sh={cands[0][3]}, |gold| max {np.abs(gold).max()}")


def compare_match(sc, gold, tag):
    same = np.array_equal(sc, gold)
    if same:
        print(f"{tag}: BIT-EXACT ({gold.shape[0]} candidates x 4 rings, "
              f"max|s| {np.abs(gold).max()})")
        return True
    bad = np.flatnonzero((sc != gold).any((1, 2)))
    print(f"{tag}: MISMATCH on {len(bad)}/{len(gold)} candidates; "
          f"first {bad[:4].tolist()}")
    for c in bad[:2]:
        print(f"   [{c}] hw   {sc[c].tolist()}")
        print(f"   [{c}] gold {gold[c].tolist()}")
    return False


def sim_match():
    gen_match()
    cells = TOOLS.parent / "share" / "yosys" / "ice40" / "cells_sim.v"
    subprocess.run([str(TOOLS / "iverilog"), "-g2012", "-o",
                    str(BUILD / "tbm.vvp"), "rtl/encoder_match.v",
                    "rtl/tb_match.v", str(cells)], cwd=ICE, check=True)
    out = subprocess.run([str(TOOLS / "vvp"), str(BUILD / "tbm.vvp")],
                         cwd=ICE, check=True, capture_output=True, text=True)
    gold = np.load(BUILD / "golden_match.npy")
    sc = np.zeros_like(gold)
    for line in out.stdout.splitlines():
        if line.startswith("SC "):
            _, c, k, re, im = line.split()
            sc[int(c), int(k)] = [int(re), int(im)]
    sys.exit(0 if compare_match(sc, gold, "sim-match") else 1)


def hw_match(port=None):
    """Encode the vector scan on-fabric, load M, run all candidates;
    demand bit-exact vs match_int. Prints wall-per-candidate (the wire
    dominates: 38 payload bytes + USB turnaround; fabric ~484 cycles)."""
    (az, r_mm, w), mc, cands, gold = match_vectors()
    b = Board(port)
    b.clear()
    b.stream_bulk(az, r_mm, w)
    assert b.npts() == len(az)
    b.load_m(mc)
    t0 = time.time()
    sc = np.stack([b.match(dx, dy, rho, s) for dx, dy, rho, s in cands])
    dt = time.time() - t0
    ok = compare_match(sc, gold,
                       f"hw-match[{len(cands)}c {dt / len(cands) * 1e3:.1f}"
                       f"ms/cand]")
    sys.exit(0 if ok else 1)


def hw_match_sweep(port=None, n=441):
    """Deploy-shaped sweep: a 21x21 translation grid x rotations, every
    reply demanded bit-exact vs match_int. Reports the pipelined wall
    rate (the deploy figure of merit for the frontend correlation)."""
    (az, r_mm, w), mc, _, _ = match_vectors()
    luts = G.make_luts()
    Q = G.encode_int(az, r_mm, w, luts)
    sh = G.shift_for(Q)
    n = int(n)
    g = np.linspace(-2048, 2048, 21).astype(int)
    cands = [(int(dx), int(dy), rho, sh)
             for rho in (0, 3, 57) for dx in g for dy in g][:n]
    gold = np.stack([G.match_int(Q, mc, dx, dy, rho, s, luts)
                     for dx, dy, rho, s in cands])
    b = Board(port)
    b.clear()
    b.stream_bulk(az, r_mm, w)
    assert b.npts() == len(az)
    b.load_m(mc)
    t0 = time.time()
    sc = b.match_sweep_paced(cands)
    dt = time.time() - t0
    ok = compare_match(sc, gold,
                       f"hw-match-sweep[{len(cands)}c "
                       f"{dt / len(cands) * 1e6:.0f}us/cand "
                       f"{len(cands) / dt:.0f}/s]")
    sys.exit(0 if ok else 1)


def hw_batch(port=None):
    """v5 acceptance for cmd 0x08: full-mode replies bit-exact vs
    match_int; totals-mode == the golden ring-summed totals; then a
    441-candidate throughput measurement in both modes."""
    (az, r_mm, w), mc, cands, gold = match_vectors()
    b = Board(port)
    b.clear()
    b.stream_bulk(az, r_mm, w)
    assert b.npts() == len(az)
    b.load_m(mc)
    full = b.match_batch(cands, totals=False)
    ok1 = compare_match(full, gold, "hw-batch[full 22c]")
    tot = b.match_batch(cands, totals=True)
    gt = gold[:, :, 0].astype(np.int64).sum(1)
    ok2 = np.array_equal(tot, gt)
    print(f"hw-batch[totals 22c]: {'BIT-EXACT' if ok2 else 'MISMATCH'}")
    luts = G.make_luts()
    Q = G.encode_int(az, r_mm, w, luts)
    sh = G.shift_for(Q)
    g = np.linspace(-2048, 2048, 21).astype(int)
    sweep = [(int(dx), int(dy), rho, sh)
             for rho in (0, 3, 57) for dx in g for dy in g][:441]
    gsw = np.stack([G.match_int(Q, mc, dx, dy, rho, s, luts)
                    for dx, dy, rho, s in sweep])
    t0 = time.time()
    tt = b.match_batch(sweep, totals=True)
    dt_t = time.time() - t0
    ok3 = np.array_equal(tt, gsw[:, :, 0].astype(np.int64).sum(1))
    t0 = time.time()
    ff = b.match_batch(sweep, totals=False)
    dt_f = time.time() - t0
    ok4 = np.array_equal(ff, gsw)
    print(f"hw-batch[441 totals]: {'BIT-EXACT' if ok3 else 'MISMATCH'} "
          f"{dt_t / 441 * 1e6:.0f}us/cand ({441 / dt_t:.0f}/s)")
    print(f"hw-batch[441 full  ]: {'BIT-EXACT' if ok4 else 'MISMATCH'} "
          f"{dt_f / 441 * 1e6:.0f}us/cand ({441 / dt_f:.0f}/s)")
    sys.exit(0 if (ok1 and ok2 and ok3 and ok4) else 1)


def hw(port=None):
    az, r, w = test_points(64)
    write_vec(az, r, w)
    b = Board(port)
    t0 = time.time()
    b.clear()
    b.stream(az, r, w)
    dt = time.time() - t0
    assert b.npts() == len(az), "point count mismatch"
    ok = compare(b.readback(), f"hw[{len(az)}pts {dt * 1e3:.0f}ms]")
    sys.exit(0 if ok else 1)


def hw_scan(port=None):
    az, r, w = spot_points(0)
    write_vec(az, r, w)
    b = Board(port)
    t0 = time.time()
    b.clear()
    b.stream(az, r, w)
    dt = time.time() - t0
    assert b.npts() == len(az)
    ok = compare(b.readback(), f"hw-spot-scan[{len(az)}pts {dt * 1e3:.0f}ms]")
    sys.exit(0 if ok else 1)


def _scan_iter(src, cap):
    """Yield (k, cleaned_ranges) from 'spot' or a dynenv name."""
    if src == "spot":
        import ssp_datasets as DS
        bnd = DS.load("spot", cap=int(cap) if cap else None)
        for k, (r, _, _) in enumerate(bnd["keys"]):
            yield k, DS.clean(bnd, r)
    else:                                   # dynenv: classroom / school
        import ssp_dynenv as DE
        people = 5 if "p5" in src else 0
        env = "school" if "school" in src else "classroom"
        bnd = DE.make(env, people=people, moving=True, seed=11,
                      n_beams=1024, cap=int(cap) if cap else 120)
        for k, (r, _, _) in enumerate(bnd["keys"]):
            yield k, np.where(r < 30, r, np.inf)


def hw_replay(port=None, src="spot", cap=None):
    """Stream every scan of a dataset through the fabric; demand zero
    mismatches. This is the corner-S 'no glitches on test data' acceptance
    check for the encode block (spot + synth envs)."""
    luts = G.make_luts()
    b = Board(port)
    bad = tot = 0
    t0 = time.time()
    n_total = 0
    for k, rr in _scan_iter(src, cap):
        n_total = k + 1
        az, r_mm, w = G.scan_to_ints(rr)
        if len(az) < 5:
            continue
        gold = G.encode_int(az, r_mm, w, luts)
        b.clear()
        b.stream_bulk(az, r_mm, w)
        hw_acc = b.readback()
        tot += 1
        if not np.array_equal(hw_acc, gold):
            bad += 1
            print(f"  scan {k}: MISMATCH")
        if (k + 1) % 50 == 0:
            print(f"  {k + 1} scans, {bad} mismatches, "
                  f"{(time.time() - t0) / (k + 1) * 1e3:.0f} ms/scan",
                  flush=True)
    dt = time.time() - t0
    print(f"hw-replay[{src}]: {tot} scans, {bad} mismatches, "
          f"{dt / max(tot, 1) * 1e3:.0f} ms/scan ({tot / dt:.1f} scans/s)"
          f" -> {'PASS (zero glitches)' if bad == 0 else 'FAIL'}")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "gen"
    fn = dict(gen=gen, sim=sim, hw=hw)
    fn["hw-scan"] = hw_scan
    fn["hw-replay"] = hw_replay
    fn["gen-match"] = gen_match
    fn["sim-match"] = sim_match
    fn["hw-match"] = hw_match
    fn["hw-match-sweep"] = hw_match_sweep
    fn["hw-batch"] = hw_batch
    fn[what](*sys.argv[2:])
