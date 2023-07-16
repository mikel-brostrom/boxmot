###############################################################
# Makefile for yolo_tracking
# Install make, then run "make help" for usage
###############################################################

PROJECT_NAME = boxmot
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

COVERAGE_FAIL_UNDER = 80

.PHONY: run clean help setup test test-cov

help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done


setup: $(VENV)/bin/activate # Setup the virtual environment. Installs requirements.txt and boxmot.
	echo "Setup done!"


$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt
	$(PIP) install -v -e .


test: setup # Run tests found in the tests/ directory.
	$(PIP) install pytest pytest-cov
	$(PYTHON) -m pytest --cov=$(PROJECT_NAME) --cov-report=html -v tests


test-cov: test # Get a test coverage report with a PASS / FAIL verdict based on coverage percentage.
	$(PYTHON) -m coverage report --fail-under=$(COVERAGE_FAIL_UNDER)


check-formatting: setup # Check code formatting and fail if there are any formatting issues.
	$(PIP) install flake8 black flake8-pylint flake8-docstrings
	$(PYTHON) -m flake8 boxmot/*
	$(PYTHON) -m black --check boxmot/*

fix-formatting: setup # Fix code formatting using autopep8 and black.
	$(PIP) install black
	$(PYTHON) -m black boxmot/*

clean: # Removes the virtual environment and pycache directories.
	rm -rf __pycache__;
	rm -rf $(VENV)