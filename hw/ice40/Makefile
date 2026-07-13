# ice40 build flow (oss-cad-suite). Usage:
#   make TOP=top_echo build      # synth + pnr + pack -> build/top_echo.bin
#   make TOP=top_echo prog       # flash via iceprog
#   make TOP=top_echo timing     # icetime report
TOOLS ?= $(HOME)/tools/oss-cad-suite/bin
TOP   ?= top_echo
RTL   ?= rtl/uart.v rtl/$(TOP).v
PCF   ?= icebreaker.pcf
B     := build/$(TOP)

build: $(B).bin

DSP ?= -dsp
$(B).json: $(RTL)
	$(TOOLS)/yosys -q -p "synth_ice40 $(DSP) -top top -json $@" $(RTL)

FREQ ?=
NPFREQ := $(if $(FREQ),--freq $(FREQ),)
$(B).asc: $(B).json $(PCF)
	$(TOOLS)/nextpnr-ice40 -q --up5k --package sg48 --pcf $(PCF) $(NPFREQ) \
	  --json $(B).json --asc $@ --report $(B)_pnr.json

$(B).bin: $(B).asc
	$(TOOLS)/icepack $< $@

prog: $(B).bin
	$(TOOLS)/iceprog $<

timing: $(B).asc
	$(TOOLS)/icetime -d up5k -mtr $(B)_timing.rpt $<
	@tail -3 $(B)_timing.rpt

clean:
	rm -f build/*

.PHONY: build prog timing clean
