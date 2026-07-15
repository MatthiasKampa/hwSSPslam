"""ECP5 CNN feasibility envelope at FULL Icepi-Zero SYSTEM resources —
what nets FIT, to inform the TRAINING_PROGRAM architecture search.

FULL-BOARD allocation model (user 2026-07-15: "tune for full system
resources including SDRAM"; measured basis = this repo's builds):

  fabric   24k LUT4 total. Explicit SLAM reserve: encoder/matcher ~6k
           (iCE40 v6 was 5.1k LC), SDRAM ctrl ~1k, IO (UART/DVP/SCCB/
           SPI/ts) ~1.5k, glue ~1.5k -> 10k reserved, ~14k for the CNN
           engine; 50% routing/control derate -> 7k usable ->
           BNN lanes at ~170 LUT/128b = 40 lanes.
  clock    MAC arrays at 75 MHz conservative (fast9 72, snap 83; PLL
           to ~100 exists — treat as upside, not the plan).
  DSP      28 MULT18 total; cis-encode datapath reserves 8-12 (iCE40
           matcher used 8 MAC16) -> 18 for CNN; int8 1 MAC/DSP/cyc,
           'packed' 2 (two 8-bit products per 18x18 + correction).
  EBR      112 KB total: map/mc codes ~35 KB + SLAM working ~20 KB ->
           ~55 KB for CNN weights + line buffers.
  SDRAM    32 MB, 16-bit @ 100 MHz -> 200 MB/s peak, ~85% sequential
           -> 170 MB/s practical, SHARED. Standing traffic: camera Y8
           ingest 9.2 MB/s @120, lidar cloud 2.6 MB/s, reservoir/map
           ~1 MB/s -> ~13 MB/s -> >150 MB/s headroom; CNN gets a
           60 MB/s planning share (weights + streamed activations).
           CAPACITY split: 16 MB reservoir/history, 8 MB maps/frames,
           8 MB CNN weights (=> up to ~8M int8 / ~64M binary params
           STORABLE; per-frame streaming is the binding limit).

Rates: vision 120/60 fps (ego-motion tier), 5 Hz (keyframe tier);
lidar 20 Hz. 'bnn' = first layer int8 on DSPs, rest on XNOR lanes.

python3 hw/ecp5/host/cnn_budget.py
"""
F_MHZ = 75.0
DSP_TOTAL, DSP_SLAM = 28, 10
DSP = DSP_TOTAL - DSP_SLAM             # 18 for the CNN engine
LUT_CNN = 7_000                        # post-derate CNN LUT budget
BNN_LANES = LUT_CNN // 170             # 41 lanes
EBR_CNN_KB = 55.0
SDRAM_MBS = 60.0                       # CNN share of ~170 practical
SDRAM_W_MB = 8.0                       # weight capacity share

INT8_MACS = DSP * F_MHZ * 1e6          # 1.35 GMAC/s
PACK_MACS = 2 * INT8_MACS              # 2.7 GMAC/s
BNN_OPS = BNN_LANES * 128 * F_MHZ * 1e6   # ~394 Gbop/s


def conv(k, cin, cout, stride=1, dw=False):
    return dict(k=k, cin=cin, cout=cout, s=stride, dw=dw)


def cost(arch, w, h):
    """-> (MACs/frame, params, line-buffer bytes, first-layer MACs)."""
    macs = params = lbuf = 0
    fl_macs = None
    for L in arch:
        w2, h2 = w // L["s"], h // L["s"]
        mults = (L["k"] ** 2 * L["cin"] if L["dw"]
                 else L["k"] ** 2 * L["cin"] * L["cout"])
        m = w2 * h2 * mults
        p = mults * (1 if L["dw"] else 1) + L["cout"]
        if not L["dw"]:
            p = L["k"] ** 2 * L["cin"] * L["cout"] + L["cout"]
        else:
            p = L["k"] ** 2 * L["cin"] + L["cin"]
        macs += m
        params += p
        lbuf += (L["k"] - 1) * w * L["cin"]          # int8 rows at input res
        if fl_macs is None:
            fl_macs = m
        w, h = w2, h2
    return macs, params, lbuf, fl_macs


ARCHS = {
    # UNIFIED vision net (user 2026-07-15: ONE net, tracking +
    # SEGMENTED classification; full/half res). Split across rate
    # tiers: trunk + tracking head at FRAME rate; the seg head reuses
    # the latest trunk features at KEYFRAME rate (entry below).
    "uni-trunk+track FULL": (
        [conv(3, 1, 8, 2), conv(3, 8, 8, 1, dw=True), conv(1, 8, 16),
         conv(3, 16, 16, 2, dw=True), conv(1, 16, 32),
         conv(3, 32, 32, 2, dw=True), conv(1, 32, 64),
         conv(1, 64, 2)], 320, 240,
        [120, 60], "trunk->40x30x64 + w/cutoff head"),
    "uni-trunk+track HALF": (
        [conv(3, 1, 8, 2), conv(3, 8, 8, 1, dw=True), conv(1, 8, 16),
         conv(3, 16, 16, 2, dw=True), conv(1, 16, 32),
         conv(3, 32, 32, 2, dw=True), conv(1, 32, 64),
         conv(1, 64, 2)], 160, 120,
        [120], "trunk->20x15x64 + w/cutoff head"),
    "uni-seg-head @kf (on trunk)": (
        [conv(3, 64, 64, 1, dw=True), conv(1, 64, 128),
         conv(1, 128, 40)], 40, 30,
        [5], "40-class per-cell seg 40x30"),
    # regime A vision: per-c16-cell weight head (ego-motion/place tiers)
    "cellweight-A (vision)": (
        [conv(3, 1, 8, 2), conv(3, 8, 8, 1, dw=True), conv(1, 8, 16),
         conv(3, 16, 16, 2), conv(1, 16, 1)], 320, 240,
        [120, 60], "per-cell weight 80x60"),
    # regime B vision: the sspax tinycnn class (descriptor head)
    "tinycnn-34k (vision)": (
        [conv(3, 1, 16, 2), conv(3, 16, 32, 2), conv(3, 32, 64, 2),
         conv(1, 64, 64)], 320, 240,
        [60, 5], "64b descriptor 40x30"),
    # regime B vision: NYU-class per-cell segmentation net
    "seg-mnet.25 (vision)": (
        [conv(3, 1, 8, 2), conv(3, 8, 8, 1, dw=True), conv(1, 8, 16),
         conv(3, 16, 16, 2, dw=True), conv(1, 16, 32),
         conv(3, 32, 32, 1, dw=True), conv(1, 32, 64),
         conv(3, 64, 64, 2, dw=True), conv(1, 64, 128),
         conv(1, 128, 40)], 320, 240,
        [5], "40-class seg 40x30"),
    # UNIFIED lidar net at the DEPLOY ingest raster, full/half beam res
    # (user 2026-07-15: no toy rasters): rings-as-channels 3 x beams;
    # track head @20 Hz, distilled per-cell label head @keyframe.
    "lidar-track FULL 1024x3": (
        [conv(3, 3, 8), conv(3, 8, 8), conv(1, 8, 2)], 1024, 3,
        [20], "per-beam-cell w/cutoff"),
    "lidar-track HALF 512x3": (
        [conv(3, 3, 8), conv(3, 8, 8), conv(1, 8, 2)], 512, 3,
        [20], "per-beam-cell w/cutoff"),
    "lidar-label @kf (on trunk)": (
        [conv(3, 8, 16), conv(1, 16, 16)], 1024, 3,
        [5], "distilled label bits"),
    # regime B/C keyframe class unlocked by SDRAM weights + full DSP:
    "seg-mnetv2-class (vision)": (
        [conv(3, 1, 16, 2), conv(3, 16, 16, 1, dw=True), conv(1, 16, 32),
         conv(3, 32, 32, 2, dw=True), conv(1, 32, 64),
         conv(3, 64, 64, 1, dw=True), conv(1, 64, 64),
         conv(3, 64, 64, 2, dw=True), conv(1, 64, 128),
         conv(3, 128, 128, 1, dw=True), conv(1, 128, 128),
         conv(1, 128, 40)], 320, 240,
        [5], "40-class seg 40x30, keyframe tier"),
    # upper bound sanity: yolo-nano class (banked OUT at int8@120)
    "yolo-nano-class": (
        [conv(3, 1, 16, 2), conv(3, 16, 32, 1), conv(3, 32, 64, 2),
         conv(3, 64, 64, 1), conv(3, 64, 128, 2), conv(3, 128, 128, 1)],
        320, 240, [120], "control (expected OUT)"),
}


def main():
    print(f"envelope @ {F_MHZ:.0f} MHz: int8 {INT8_MACS/1e9:.1f} GMAC/s | "
          f"packed {PACK_MACS/1e9:.1f} | BNN {BNN_OPS/1e9:.0f} Gbop/s "
          f"({BNN_LANES} lanes, {LUT_CNN} LUT post-derate); "
          f"EBR-for-CNN {EBR_CNN_KB:.0f} KB, "
          f"SDRAM weights {SDRAM_MBS:.0f} MB/s")
    print(f"{'arch':24s} {'MMAC/f':>7s} {'params':>7s} {'lbuf':>6s} "
          f"{'w-int8':>8s}  fps: int8 | packed | bnn   verdict")
    for name, (arch, w, h, rates, out) in ARCHS.items():
        m, p, lb, fl = cost(arch, w, h)
        f_int8 = INT8_MACS / m
        f_pack = PACK_MACS / m
        # bnn: first layer int8 on DSPs (parallel), rest on lanes
        f_bnn = min(BNN_OPS / max(m - fl, 1), INT8_MACS / max(fl, 1))
        wkb = p / 1024                     # int8 KB; BNN = /8
        r_t = max(rates)

        def w_ok(kb, r):
            if kb + lb / 1024 <= EBR_CNN_KB:
                return "EBR"
            if (kb / 1024 * r <= SDRAM_MBS
                    and kb / 1024 <= SDRAM_W_MB * 1024):
                return "SDRAM"
            return None

        modes = []
        if f_pack >= r_t and w_ok(wkb, r_t):
            modes.append(f"int8-packed/{w_ok(wkb, r_t)}")
        if f_bnn >= r_t and w_ok(wkb / 8, r_t):
            modes.append(f"BNN/{w_ok(wkb / 8, r_t)}({wkb/8:.0f}KB)")
        verdict = (f"fits @{r_t} via " + ", ".join(modes) if modes
                   else f"DOES NOT FIT @{r_t}") + f" ({out})"
        print(f"{name:24s} {m/1e6:7.1f} {p/1e3:6.1f}k {lb/1024:5.1f}K "
              f"{wkb:6.1f}KB  {f_int8:5.0f} | {f_pack:6.0f} | "
              f"{f_bnn:5.0f}   {verdict}", flush=True)
    print("\narchitecture-search ENVELOPE (per frame):")
    for r in (120, 60, 20, 5):
        print(f"  @{r:3d} fps: int8 {INT8_MACS/r/1e6:6.1f} MMAC | packed "
              f"{PACK_MACS/r/1e6:6.1f} | BNN {BNN_OPS/r/1e6:7.0f} Mbop")
    print(f"  params: <= ~{EBR_CNN_KB:.0f}k int8 in EBR (minus line "
          f"buffers); up to ~{SDRAM_MBS*1e3/5/1024:.0f}0k int8 at 5 Hz "
          f"via SDRAM streaming; BNN divides bytes by 8")
    print("  line buffers: (k-1) x width x cin bytes per conv at its "
          "input resolution — early stride-2 is the lever")


if __name__ == "__main__":
    main()
