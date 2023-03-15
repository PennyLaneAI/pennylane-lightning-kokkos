PYTHON3 := $(shell which python3 2>/dev/null)

PYTHON := python3
COVERAGE := --cov=pennylane_lightning_kokkos --cov-report term-missing --cov-report=html:coverage_html_report
TESTRUNNER := -m pytest tests --tb=short

LIGHTNING_KOKKOS_CPP_DIR := pennylane_lightning_kokkos/src/

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  test-python        to run the PennyLane-Lightning-Kokkos Python test suite"
	@echo "  test-cpp        	to run the PennyLane-Lightning-Kokkos C++ test suite"
	@echo "  docs               to build the documentation for Catalyst"
	@echo "  clean              to uninstall PennyLane-Lightning-Kokkos and delete all temporary, cache, and build files"
	@echo "  format [check=1]   to apply C++ and Python formatter; use with 'check=1' to check instead of modify (requires black and clang-format)"
	@echo "  format [version=?] to apply C++ and Python formatter; use with 'version={version}' to run clang-format-{version} instead of clang-format"

.PHONY: format format-cpp format-python clean test-builtin test-cpp
format: format-cpp format-python

format-cpp:
ifdef check
	./bin/format --check --cfversion $(if $(version:-=),$(version),0) ./pennylane_lightning_kokkos/src
else
	./bin/format --cfversion $(if $(version:-=),$(version),0) ./pennylane_lightning_kokkos/src
endif

format-python:
ifdef check
	black -l 100 ./pennylane_lightning_kokkos/ ./tests --check
else
	black -l 100 ./pennylane_lightning_kokkos/ ./tests
endif

test-python:
	$(PYTHON) -I $(TESTRUNNER)

test-cpp:
	# rm -rf ./BuildTests
	cmake . -BBuildTests -DPLKOKKOS_BUILD_TESTS=1
	cmake --build ./BuildTests
	./BuildTests/pennylane_lightning_kokkos/src/tests/runner_kokkos

clean:
	$(PYTHON) setup.py clean --all
	rm -rf pennylane_lightning_kokkos/__pycache__
	rm -rf pennylane_lightning_kokkos/src/__pycache__
	rm -rf tests/__pycache__
	rm -rf pennylane_lightning_kokkos/src/tests/__pycache__
	rm -rf dist
	rm -rf build
	rm -rf BuildTests Build
	rm -rf .coverage coverage_html_report/
	rm -rf tmp
	rm -rf *.dat
	rm -rf pennylane_lightning_kokkos/lightning_kokkos_qubit_ops*
