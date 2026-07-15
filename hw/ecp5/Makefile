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

prog: $(B).bit
	$(TOOLS)/openFPGALoader -b icepi-zero $<

clean:
	rm -rf build

.PHONY: build timing sim prog clean
