# ECP5 / Icepi Zero build flow (oss-cad-suite). Usage:
#   make TOP=top_fast9 build      # synth + pnr + pack -> build/top_fast9.bit
#   make TOP=top_fast9 timing     # PnR timing summary (vs 50 MHz osc)
#   make sim                      # iverilog: fast9.v vs golden vectors
#   make prog                     # openFPGALoader -b icepi-zero (board attached)
TOOLS ?= $(HOME)/tools/oss-cad-suite/bin
TOP   ?= top_fast9
RTL   ?= rtl/fast9.v rtl/$(TOP).v
LPF   ?= icepi-zero.lpf
B     := build/$(TOP)

build: $(B).bit

$(B).json: $(RTL)
	@mkdir -p build
	$(TOOLS)/yosys -q -p "synth_ecp5 -top top -json $@" $(RTL)

$(B).config: $(B).json $(LPF)
	$(TOOLS)/nextpnr-ecp5 -q --25k --package CABGA256 --speed 6 \
	  --lpf $(LPF) --json $< --textcfg $@ --report $(B)_pnr.json

$(B).bit: $(B).config
	$(TOOLS)/ecppack --compress $< $@

timing: $(B).config
	@python3 -c "import json; r = json.load(open('$(B)_pnr.json')); \
	  fmax = r['fmax']; \
	  [print(f'{k}: {v[\"achieved\"]:.1f} MHz (constraint {v[\"constraint\"]:.1f})') for k, v in fmax.items()]"

sim: build/tb_fast9.vvp
	cd build && $(TOOLS)/vvp tb_fast9.vvp

build/tb_fast9.vvp: rtl/fast9.v rtl/tb_fast9.v build/fast9_thresh.txt
	$(TOOLS)/iverilog -g2012 -DTHRESH=$(shell cat build/fast9_thresh.txt) \
	  -o $@ rtl/fast9.v rtl/tb_fast9.v

build/fast9_thresh.txt:
	cd ../.. && python3 -m runners.spot_cam vectors school_run2

# camera-front sim gates (sccb master / init ROM walker / DVP capture)
sim-sccb:
	@mkdir -p build
	$(TOOLS)/iverilog -g2012 -o build/tb_sccb.vvp rtl/sccb.v rtl/tb_sccb.v
	cd build && $(TOOLS)/vvp tb_sccb.vvp

build/ov5640_init.hex: host/gen_ov5640_rom.py host/vendor/ov5640_settings.h
	python3 host/gen_ov5640_rom.py

sim-init: build/ov5640_init.hex
	$(TOOLS)/iverilog -g2012 -o build/tb_init.vvp rtl/sccb.v \
	  rtl/ov5640_init.v rtl/tb_ov5640_init.v
	cd build && $(TOOLS)/vvp tb_init.vvp

sim-dvp:
	@mkdir -p build
	$(TOOLS)/iverilog -g2012 -o build/tb_dvp.vvp rtl/dvp_capture.v \
	  rtl/tb_dvp.v
	cd build && $(TOOLS)/vvp tb_dvp.vvp

# camera tops need the cam pin constraints appended to the board LPF
build/cam.lpf: icepi-zero.lpf cam-pins.lpf
	@mkdir -p build
	cat icepi-zero.lpf cam-pins.lpf > $@

# full sensor set (camera + ISM330DHCX IMU) — safe for any top: nextpnr
# ignores LOCATEs for ports a design does not have
build/full.lpf: icepi-zero.lpf cam-pins.lpf imu-pins.lpf
	@mkdir -p build
	cat icepi-zero.lpf cam-pins.lpf imu-pins.lpf > $@

sim-spi:
	@mkdir -p build
	$(TOOLS)/iverilog -g2012 -o build/tb_spi.vvp rtl/spi_reg.v \
	  rtl/tb_spi.v
	cd build && $(TOOLS)/vvp tb_spi.vvp

prog: $(B).bit
	$(TOOLS)/openFPGALoader -b icepi-zero $<

clean:
	rm -rf build

.PHONY: build timing sim prog clean
