"""OV5640 init-ROM generator (hw/ecp5 camera track, bring-up step 2).

Parses the vendored esp32-camera tables (host/vendor/ov5640_settings.h +
ov5640_regs.h — silicon-proven init source, Apache-2.0, see vendor/README)
and emits build/ov5640_init.hex for the rtl/ov5640_init.v ROM walker.
ROM word = 24 bit {addr[15:0], data[7:0]}, one hex word per line:
  addr 0xFFFF  delay, data = milliseconds (delays > 255 ms are split)
  addr 0x0000  end of ROM (the REGLIST_TAIL convention, kept as-is)

Sections in write order:
  1. sensor_default_regs — verbatim from the vendor table (symbols
     resolved, REG_DLY kept): sensor core + ISP defaults.
  2. format — sensor_fmt_grayscale (Y8: DVP emits luma directly, one
     byte/pixel — halves bandwidth; the pipeline is luma-only) or
     sensor_fmt_yuv422 with --fmt yuyv (Y = every 2nd byte, phase 0).
  3. QVGA windowing — replica of esp32 ov5640.c set_framesize +
     set_image_options, binning branch (320x240 <= half of 2560x1920):
     full-array window, OUTPUT 320x240, TOTAL (2060, 984), offset
     (16, 8), TIMING_TC 2x2 binning bits, 0x4514/0x4520 binning
     companions, X/Y_INCREMENT 0x31.
  4. rate (PLL) — the esp32 non-JPEG profile re-encoded for our chain
     (XCLK 25 MHz from the FPGA): VCO = XCLK/prediv*mult; /2.5 (10-bit
     mode 0x3034=0x1A); /sysdiv -> internal clocks; PCLK further
     /pclk_root/pclk_div. Predicted (NOT measured): PCLK ~12.5 MHz,
     ~25 fps at TOTAL 2060x984. Frame-rate stepping to 60/120 fps is a
     LIVE operation against the snapshot top's measured counters (VSYNC
     rate / lines / bytes-per-line) via its UART->SCCB passthrough —
     every rate register below is patchable without reflash.

python3 hw/ecp5/host/gen_ov5640_rom.py selftest
python3 hw/ecp5/host/gen_ov5640_rom.py [--fmt y8|yuyv]   # -> build/
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
BUILD = HERE.parent / "build"

REG_DLY, TAIL = 0xFFFF, 0x0000


def symbols():
    """#define NAME 0xADDR from both vendored headers."""
    sym = {}
    for f in ("ov5640_regs.h", "ov5640_settings.h"):
        for m in re.finditer(r"#define\s+(\w+)\s+(0[xX][0-9a-fA-F]+|\d+)",
                             (VENDOR / f).read_text()):
            sym[m.group(1)] = int(m.group(2), 0)
    return sym


def table(name, sym):
    """[(addr, data)] from a uint16_t NAME[][2] initializer (tail dropped,
    REG_DLY kept as (0xFFFF, ms))."""
    txt = (VENDOR / "ov5640_settings.h").read_text()
    m = re.search(name + r"\[\]\[2\]\s*=\s*\{(.*?)\n\};", txt, re.S)
    assert m, f"table {name} not found"
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", m.group(1), flags=re.S)
    out = []
    for a, d in re.findall(r"\{\s*([\w]+)\s*,\s*([\w]+)\s*\}", body):
        addr = sym[a] if a in sym else int(a, 0)
        data = sym[d] if d in sym else int(d, 0)
        if addr == TAIL:
            continue
        out.append((addr, data))
    return out


def _pair16(reg_h, val):
    return [(reg_h, (val >> 8) & 0xFF), (reg_h + 1, val & 0xFF)]


def qvga_window(sym):
    """esp32 set_framesize + set_image_options, binning branch, for
    320x240 on the 4x3 ratio row (2560x1920 array, end 2623x1951,
    total 2844x1968, offset 32x16)."""
    w, h = 320, 240
    sx, sy, ex, ey = 0, 0, 2623, 1951
    ox, oy, tx, ty = 32, 16, 2844, 1968
    rows = []
    rows += _pair16(sym["X_ADDR_ST_H"], sx) + _pair16(sym["Y_ADDR_ST_H"], sy)
    rows += _pair16(sym["X_ADDR_END_H"], ex) + _pair16(sym["Y_ADDR_END_H"], ey)
    rows += _pair16(sym["X_OUTPUT_SIZE_H"], w) + \
        _pair16(sym["Y_OUTPUT_SIZE_H"], h)
    # binning branch, w <= 920: TOTAL = (2060, ty/2), OFFSET halved
    rows += _pair16(sym["X_TOTAL_SIZE_H"], 2060) + \
        _pair16(sym["Y_TOTAL_SIZE_H"], ty // 2)
    rows += _pair16(sym["X_OFFSET_H"], ox // 2) + \
        _pair16(sym["Y_OFFSET_H"], oy // 2)
    # set_image_options: binning on, no vflip/mirror
    rows += [(sym["TIMING_TC_REG20"], 0x01), (sym["TIMING_TC_REG21"], 0x01),
             (0x4514, 0xAA), (0x4520, 0x0B),
             (sym["X_INCREMENT"], 0x31), (sym["Y_INCREMENT"], 0x31)]
    return rows


def rate_pll():
    """esp32 non-JPEG set_pll profile (mult 10, prediv 1, sysdiv 1,
    root /1, 10-bit /2.5, pclk_root /2, pclk_div 4, manual PCLK):
    at XCLK 20 MHz that chain is the driver's '10 MHz PCLK' point; our
    25 MHz XCLK scales it to PCLK ~12.5 MHz, internal ~50 MHz -> ~25 fps
    at TOTAL 2060x984. All predicted-not-measured; live-patchable."""
    return [
        (0x3039, 0x00),      # PLL not bypassed
        (0x3034, 0x1A),      # bit mode 10 -> /2.5
        (0x3035, 0x11),      # [7:4] sysdiv 1, [3:0] scale div 1
        (0x3036, 0x0A),      # PLL multiplier 10 -> VCO 250 MHz @ 25 MHz
        (0x3037, 0x01),      # [4] root /1, [3:0] prediv 1
        (0x3108, 0x16),      # [5:4] pclk root /2, sclk2x/sclk dividers
        (0x3824, 0x04),      # DVP PCLK divider (manual)
        (0x460C, 0x22),      # PCLK divider manual (also in defaults)
        (0x3103, 0x13),      # system clock from PLL
    ]


def build_rom(fmt="y8"):
    sym = symbols()
    fmt_tab = {"y8": "sensor_fmt_grayscale",
               "yuyv": "sensor_fmt_yuv422"}[fmt]
    rom = []
    rom += table("sensor_default_regs", sym)
    rom += table(fmt_tab, sym)
    rom += qvga_window(sym)
    rom += rate_pll()
    # settle before first frame is trusted
    rom.append((REG_DLY, 100))
    out = []
    for addr, data in rom:
        if addr == REG_DLY:                     # split >255 ms
            while data > 255:
                out.append((REG_DLY, 255))
                data -= 255
        assert 0 <= data <= 0xFF, (hex(addr), data)
        out.append((addr, data))
    out.append((TAIL, 0x00))
    return out


def write_hex(rom, path):
    with open(path, "w") as f:
        f.writelines(f"{(a << 8) | d:06x}\n" for a, d in rom)


def selftest():
    sym = symbols()
    assert sym["SYSTEM_CTROL0"] == 0x3008
    assert sym["FORMAT_CTRL"] == 0x501F and sym["FORMAT_CTRL00"] == 0x4300
    assert sym["REG_DLY"] == 0xFFFF and sym["REGLIST_TAIL"] == 0x0000
    d = table("sensor_default_regs", sym)
    assert d[0] == (0x3008, 0x82) and d[1] == (0xFFFF, 10)
    assert (0x3103, 0x13) in d and d[-1] == (0xFFFF, 300)
    assert all(a != TAIL for a, _ in d)
    y8 = table("sensor_fmt_grayscale", sym)
    assert y8 == [(0x501F, 0x00), (0x4300, 0x10)]
    win = qvga_window(sym)
    wd = dict(win)
    assert wd[0x3808] == 0x01 and wd[0x3809] == 0x40      # 320
    assert wd[0x380A] == 0x00 and wd[0x380B] == 0xF0      # 240
    assert wd[0x380C] == 0x08 and wd[0x380D] == 0x0C      # HTS 2060
    assert wd[0x380E] == 0x03 and wd[0x380F] == 0xD8      # VTS 984
    assert wd[0x3814] == 0x31 and wd[0x3820] == 0x01
    rom = build_rom("y8")
    assert rom[-1] == (TAIL, 0)
    assert all(0 <= a <= 0xFFFF and 0 <= v <= 0xFF for a, v in rom)
    dly = sum(v for a, v in rom if a == REG_DLY)
    n_dly = sum(a == REG_DLY for a, _ in rom)
    print(f"selftest ok: {len(rom)} ROM entries ({n_dly} delays, "
          f"{dly} ms total), QVGA window + Y8 + rate section verified")


def main():
    fmt = "y8"
    if "--fmt" in sys.argv:
        fmt = sys.argv[sys.argv.index("--fmt") + 1]
    rom = build_rom(fmt)
    BUILD.mkdir(exist_ok=True)
    write_hex(rom, BUILD / "ov5640_init.hex")
    print(f"wrote {BUILD / 'ov5640_init.hex'}: {len(rom)} entries "
          f"(fmt={fmt}, QVGA 2x2-binned, predicted ~25 fps / "
          f"PCLK ~12.5 MHz @ XCLK 25 MHz)")


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "selftest":
        selftest()
    else:
        main()
