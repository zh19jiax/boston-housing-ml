# Boston Housing Violations — final project Makefile.
#
# Quick start:
#     make install   # install Python dependencies
#     make all       # install + regenerate dataset + train final model + run tests
#
# Or run any single target on its own (see below).

PYTHON ?= python3

.PHONY: help install data train test all clean

help:
	@echo "Available targets:"
	@echo "  make install   Install Python dependencies from requirements.txt"
	@echo "  make data      Re-run data_processing.ipynb to regenerate merged_violations.csv"
	@echo "  make train     Execute modeling_final.ipynb end-to-end (writes outputs in place)"
	@echo "  make test      Run the pytest suite under tests/"
	@echo "  make all       install + train + test (does NOT re-run data; we ship the CSV)"
	@echo "  make clean     Remove generated artifacts (predictions.csv, __pycache__, checkpoints)"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Re-runs the cleaning notebook. Note: requires the four raw Boston datasets in ./data/
# (BV.csv, SAM.csv, PA.csv, etc.). The repo already ships the cleaned output
# (merged_violations.csv), so you only need this if you want to regenerate from scratch.
data:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace data_processing.ipynb

# Executes the final modeling notebook in place — saves outputs (numbers, plots) into the
# notebook itself so a grader sees results immediately on GitHub.
train:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace modeling_final.ipynb

test:
	$(PYTHON) -m pytest tests/ -v

all: install train test

clean:
	rm -f predictions.csv
	rm -rf __pycache__ tests/__pycache__ .pytest_cache .ipynb_checkpoints
