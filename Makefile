# Regenerate the LUT-layer RTL from an exported model, then synthesise, place & route and flash the two DWN designs: WESAD on the ECP5 evaluation board (DESIGN=wesad, the default) and MNIST on the iCE40UP5K (DESIGN=mnist). Tools come from an OSS CAD Suite install on PATH.

SHELL  := /bin/bash

DESIGN ?= wesad
PYTHON ?= python3

ifeq ($(DESIGN),wesad)
  FAMILY ?= ecp5
  TOP    ?= wesad_i2c_top
  LINK   ?= i2c
  W      := wesad/rtl
  INC    := -I $(W)
  GEN    := $(W)/dwn
  MODEL  ?= verification/exported/dwn_wesad_hw115_100_51_tau3.5.json
  SRC    := $(W)/dwn/dwn_params.vh $(W)/dwn/*.v $(W)/dwn/*.sv $(W)/*.sv \
            $(W)/feature_engine/*.sv $(W)/reading/*.sv $(W)/frontend/*.sv \
            $(W)/frontend/SPI/*.sv $(W)/frontend/I2C/*.v $(W)/frontend/I2C/*.sv \
            $(W)/frontend/input_buffers/input_buffer_spram_$(FAMILY).sv
else
  FAMILY ?= ice40
  TOP    ?= dwn_uart_top
  LINK   ?= uart
  INC    :=
  GEN    := rtl/generated
  MODEL  ?= verification/exported/dwn_400_200_tau1.3.json
  SRC    := rtl/generated/dwn_params.vh rtl/*.v rtl/generated/*.v
endif

ifeq ($(FAMILY),ice40)
  DEVICE  ?= up5k
  PACKAGE ?= sg48
  CONSTR  ?= rtl/constraints/UP5K/dwn_uart_mnist.pcf
  OUT     := bin/$(DESIGN)_$(LINK)_$(DEVICE)
  SYNTH   := synth_ice40 -dsp
  PNR     := nextpnr-ice40 --$(DEVICE) --package $(PACKAGE) --pcf $(CONSTR) --freq 12 --seed 1
  ROUTE_OUT := --asc
  ROUTED  := $(OUT).asc
  IMAGE   := $(OUT).bin
  PACK     = icepack $< $@
  PROG    := iceprog
else
  DEVICE  ?= um5g-85k
  PACKAGE ?= CABGA381
  CONSTR  ?= rtl/constraints/ECP5/constraints_ecp5_wesad_i2c.lpf
  OUT     := bin/$(DESIGN)_$(LINK)_ecp5
  SYNTH   := synth_ecp5
  PNR     := nextpnr-ecp5 --$(DEVICE) --package $(PACKAGE) --lpf $(CONSTR)
  ROUTE_OUT := --textcfg
  ROUTED  := $(OUT).config
  IMAGE   := $(OUT).bit
  PACK     = ecppack --input $< --bit $@
  PROG    := openFPGALoader -b ecp5_evn
endif

SRC_FILES := $(wildcard $(SRC))
PNR_FLAGS ?=

.PHONY: model synth pnr flash

# The feature engines read thresholds.hex, twiddle.hex and hann.hex at
# elaboration; regenerate them under wesad/rtl/ before a wesad build.
model:
	$(PYTHON) rtl/generate_rtl.py $(MODEL) -o $(GEN)

synth: $(OUT).json
$(OUT).json: $(SRC_FILES)
	@mkdir -p $(dir $@)
	yosys -p "read_verilog -sv $(INC) $(SRC_FILES); $(SYNTH) -top $(TOP) -json $@"

pnr: $(ROUTED)
$(ROUTED): $(OUT).json $(CONSTR)
	$(PNR) --json $< $(ROUTE_OUT) $@ $(PNR_FLAGS)

$(IMAGE): $(ROUTED)
	$(PACK)

flash: $(IMAGE)
	$(PROG) $(IMAGE)
