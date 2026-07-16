#!/usr/bin/env python3
"""Generate encoder ROM images + sim vectors for the ECP5 stream+encode top.
ROMs come from hw.ice40.golden.make_luts (ONE source of constants — the
same golden the silicon must match bit-exactly). Writes hw/ecp5/build/.

  python3 hw/ecp5/host/gen_enc_roms.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import hw.ice40.golden as G                                    # noqa: E402

OUT = ROOT / "hw" / "ecp5" / "build"


def w_hex(path, vals, bits):
    mask = (1 << bits) - 1
    with open(path, "w") as f:
        for v in vals:
            f.write(f"{int(v) & mask:0{bits // 4}x}\n")


def main():
    OUT.mkdir(exist_ok=True)
    luts = G.make_luts()
    w_hex(OUT / "az_c.hex", luts["az_c"], 16)
    w_hex(OUT / "az_s.hex", luts["az_s"], 16)
    w_hex(OUT / "ang_c.hex", np.pad(luts["ang_c"], (0, 64 - G.N_ANG)), 16)
    w_hex(OUT / "ang_s.hex", np.pad(luts["ang_s"], (0, 64 - G.N_ANG)), 16)
    w_hex(OUT / "cis_re.hex", luts["cis_re"], 8)
    w_hex(OUT / "cis_im.hex", luts["cis_im"], 8)

    # sim vectors: a mini single-ring scan (8 columns at az0=100) with a
    # miss (0) and an out-of-mask (>31200 mm) case; expected accumulators
    # from the golden on the KEPT points (w=127 unit weights).
    mm = np.array([1234, 0, 5000, 31201, 250, 60000, 7777, 31200],
                  np.uint16)
    az0 = 100
    w_hex(OUT / "tb_enc_scan.hex", mm, 16)
    keep = (mm > 0) & (mm <= G.R_MASK_MM)
    az = (az0 + np.flatnonzero(keep)).astype(np.int32)
    exp = G.encode_int(az, mm[keep].astype(np.int32),
                       np.full(int(keep.sum()), 127, np.int32), luts)
    inter = np.empty(480, np.int64)
    inter[0::2] = exp[:, 0]
    inter[1::2] = exp[:, 1]
    w_hex(OUT / "tb_enc_exp.hex", inter, 32)
    # expected 2-bit QPSK map codes (the on-chip compressed store),
    # packed 4 components/byte little-first — golden mcode convention
    mc = G.mcode_from_vec(exp[:, 0] + 1j * exp[:, 1])
    packed = (mc[0::4] | (mc[1::4] << 2) | (mc[2::4] << 4)
              | (mc[3::4] << 6))
    w_hex(OUT / "tb_enc_mcode.hex", packed, 8)
    print(f"wrote ROMs + sim vectors to {OUT} "
          f"(mini-scan kept {int(keep.sum())}/8 pts)")


if __name__ == "__main__":
    main()
