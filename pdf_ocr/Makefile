# mec-pdf -- PDF -> rasterise -> deskew -> (later) Mistral OCR 4
#
# Conventions:
#   PDF_DIR   - where the source PDFs live
#   OUT_DIR   - where deskewed PNGs and skew_log.csv are written
#   SAMPLE_*  - parameters for the dry-run target
#
# Override on the command line, e.g.:
#   make sample PDF_DIR=/mnt/c/Users/Ice/Downloads
#   make all    PDF_DIR=/mnt/c/Users/Ice/Downloads OUT_DIR=./out

PDF_DIR      ?= /mnt/c/Users/Ice/Downloads
OUT_DIR      ?= ./out
SAMPLE_OUT   ?= ./sample_out
DPI          ?= 300
THRESHOLD    ?= 0.05
MAX_DEG      ?= 5.0
MAX_ITERS    ?= 1

# Glob expanded at use-time so new PDFs added later are picked up.
PDFS = $(wildcard $(PDF_DIR)/2026*.pdf)

UV = uv run --

.PHONY: help sync sample all probe diag clean clean-all check tree

SAMPLES ?= 20
DIAG_OUT ?= ./diagnostic

help:
	@echo "mec-pdf targets:"
	@echo "  make sync         - install/sync Python deps via uv"
	@echo "  make sample       - dry-run pipeline on the 8 known sample pages"
	@echo "  make all          - run pipeline on every PDF in PDF_DIR"
	@echo "  make probe IMG=.. - cross-check skew angle on one PNG with 4 methods"
	@echo "  make check        - assert system deps (pdftoppm) are present"
	@echo "  make clean        - remove $(OUT_DIR) and $(SAMPLE_OUT)"
	@echo "  make clean-all    - clean + remove .venv"
	@echo ""
	@echo "Vars (override with make VAR=value):"
	@echo "  PDF_DIR   = $(PDF_DIR)"
	@echo "  OUT_DIR   = $(OUT_DIR)"
	@echo "  DPI       = $(DPI)"
	@echo "  THRESHOLD = $(THRESHOLD) (degrees; smaller angles skipped)"

check:
	@command -v pdftoppm >/dev/null || { \
	  echo "ERROR: pdftoppm not found. Install: sudo apt-get install poppler-utils"; \
	  exit 1; }
	@command -v uv >/dev/null || { \
	  echo "ERROR: uv not found. See https://docs.astral.sh/uv/"; \
	  exit 1; }
	@echo "OK: pdftoppm and uv available"

sync:
	uv sync

PIPELINE_OPTS = --dpi $(DPI) \
  --threshold-deg $(THRESHOLD) --max-deg $(MAX_DEG) --max-iters $(MAX_ITERS)

sample: check sync
	@echo "Dry-run on the 8 known sample pages -> $(SAMPLE_OUT)"
	$(UV) python pipeline.py \
	  "$(PDF_DIR)/20260625092650.pdf" --pages 8,38 \
	  --out "$(SAMPLE_OUT)" $(PIPELINE_OPTS)
	$(UV) python pipeline.py \
	  "$(PDF_DIR)/20260625092932.pdf" --pages 2,3 \
	  --out "$(SAMPLE_OUT)" $(PIPELINE_OPTS)
	$(UV) python pipeline.py \
	  "$(PDF_DIR)/20260625093306.pdf" --pages 13,52 \
	  --out "$(SAMPLE_OUT)" $(PIPELINE_OPTS)
	$(UV) python pipeline.py \
	  "$(PDF_DIR)/20260625093803.pdf" --pages 15,24 \
	  --out "$(SAMPLE_OUT)" $(PIPELINE_OPTS)

all: check sync
	@test -n "$(PDFS)" || { echo "No PDFs found in $(PDF_DIR)"; exit 1; }
	@echo "Found $(words $(PDFS)) PDF(s) in $(PDF_DIR)"
	$(UV) python pipeline.py $(PDFS) \
	  --out "$(OUT_DIR)" $(PIPELINE_OPTS)

probe: sync
	@test -n "$(IMG)" || { echo "Usage: make probe IMG=/path/to/page.png"; exit 1; }
	$(UV) python skew_probe.py "$(IMG)"

diag: check sync
	@test -n "$(PDFS)" || { echo "No PDFs found in $(PDF_DIR)"; exit 1; }
	$(UV) python diagnostic.py $(PDFS) \
	  --out "$(DIAG_OUT)" --samples $(SAMPLES) --dpi $(DPI)

tree:
	@find . -maxdepth 2 -not -path './.git*' -not -path './.venv*' -not -path './__pycache__*' | sort

clean:
	rm -rf "$(OUT_DIR)" "$(SAMPLE_OUT)"

clean-all: clean
	rm -rf .venv uv.lock
