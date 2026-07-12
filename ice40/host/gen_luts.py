#!/usr/bin/env python3
"""Generate $readmemh ROM images from the golden model (ssp_ice40.make_luts).
Writes to ice40/build/. RTL and golden model share ONE source of constants."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ssp_ice40 as G                                    # noqa: E402

OUT = ROOT / "ice40" / "build"


def w_hex(path, vals, bits):
    mask = (1 << bits) - 1
    with open(path, "w") as f:
        for v in vals:
            f.write(f"{int(v) & mask:0{bits // 4}x}\n")


def az_quarter():
    """Quarter-wave fold of the az ROMs (v7-lean: 8 EBR -> 1).
    T[k] = round(scale*cos(k*2pi/1024)), k in [0,255] (T[256]=0 is a
    logic special case). Fold (EXACT, np.round is odd-symmetric):
      cosfold(i): q=i>>8, o=i&255 ->
        q0:+T[o]  q1:(o? -T[256-o] : 0)  q2:-T[o]  q3:(o? +T[256-o] : 0)
      az_c[i] = -cosfold(i); az_s[i] = -cosfold((i-256)&1023)."""
    scale = G.U_PER_MM * (1 << G.F_AZ)
    T = np.round(scale * np.cos(np.arange(257) * 2 * np.pi / 1024)
                 ).astype(np.int64)
    assert T[256] == 0

    def cosfold(i):
        q, o = i >> 8, i & 255
        if q == 0:
            return int(T[o])
        if q == 1:
            return -int(T[256 - o]) if o else 0
        if q == 2:
            return -int(T[o])
        return int(T[256 - o]) if o else 0
    return T[:256], cosfold


def cs_quarter():
    """Quarter-wave fold of the heading ROM (cs_of contract, TAU_Q=960).
    Tc[o] = round(32767*cos(o*2pi/960)), o in [0,240] (Tc[240]=0 special
    case). csfold as az; cq(h)=csfold(h); sq(h)=csfold((h+720) mod 960)."""
    Tc = np.array([round(32767 * float(np.cos(o * 2 * np.pi / 960)))
                   for o in range(241)], np.int64)
    assert Tc[240] == 0

    def csfold(h):
        q, o = h // 240, h % 240
        if q == 0:
            return int(Tc[o])
        if q == 1:
            return -int(Tc[240 - o]) if o else 0
        if q == 2:
            return -int(Tc[o])
        return int(Tc[240 - o]) if o else 0
    return Tc[:240], csfold


def main():
    OUT.mkdir(exist_ok=True)
    luts = G.make_luts()
    w_hex(OUT / "az_c.hex", luts["az_c"], 16)
    w_hex(OUT / "az_s.hex", luts["az_s"], 16)
    ang_c = np.zeros(64, np.int32)
    ang_s = np.zeros(64, np.int32)
    ang_c[:luts["n_ang"]] = luts["ang_c"]
    ang_s[:luts["n_ang"]] = luts["ang_s"]
    w_hex(OUT / "ang_c.hex", ang_c, 16)
    w_hex(OUT / "ang_s.hex", ang_s, 16)
    w_hex(OUT / "cis_re.hex", luts["cis_re"], 8)
    w_hex(OUT / "cis_im.hex", luts["cis_im"], 8)
    packed = ((luts["cis_im"].astype(np.int64) & 0xFF) << 8) \
        | (luts["cis_re"].astype(np.int64) & 0xFF)
    w_hex(OUT / "cis.hex", packed, 16)      # {im[7:0], re[7:0]} per entry

    # ---- v7-lean quarter-wave images + EXHAUSTIVE fold checks ----
    T, cosfold = az_quarter()
    for i in range(1024):
        assert -cosfold(i) == int(luts["az_c"][i]), f"az_c fold @{i}"
        assert -cosfold((i - 256) & 1023) == int(luts["az_s"][i]), \
            f"az_s fold @{i}"
    w_hex(OUT / "az_t.hex", T, 16)

    sys.path.insert(0, str(ROOT / "ice40" / "host"))
    import solo                                            # noqa: E402
    Tc, csfold = cs_quarter()
    for h in range(960):
        cq, sq = solo.cs_of(h)
        assert csfold(h) == cq, f"cs_t cos fold @{h}: {csfold(h)} != {cq}"
        assert csfold((h + 720) % 960) == sq, f"cs_t sin fold @{h}"
    cs_pad = np.zeros(256, np.int64)
    cs_pad[:240] = Tc
    w_hex(OUT / "cs_t.hex", cs_pad, 16)
    print(f"wrote ROMs to {OUT} (n_ang={luts['n_ang']}); quarter-wave "
          f"az_t/cs_t folds verified exhaustively (1024 az, 960 h)")


if __name__ == "__main__":
    main()
