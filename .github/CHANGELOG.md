# Release 0.31.0

### Breaking changes

* Deprecate `kokkos_config_info`, replaced by Kokkos' `print_configuration`.
  [(#55)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/55)

* Update tests to be compliant with PennyLane v0.31.0 development changes and deprecations.
  [(#66)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/66)

### Improvements

* Upgrade Kokkos version to v4.0.01.
  [(#55)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/55)

* Remove logic from `setup.py` and transfer paths and definitions into workflow files.
  [(#58)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/58)

* Use `Operator.name` instead of `Operation.base_name`.
  [(#67)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/67)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee, Vincent Michaud-Rioux

---

# Release 0.30.0

### New features since last release

* Add native support for `expval` and `var` of generic observables. Refactor measurements support.
  [(#47)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/47)

### Breaking changes

* Provide support for PennyLane-Lightning-Kokkos to coexist with PennyLane-Lightning.
  [(#49)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/49)

### Improvements

* Replace deprecated InitArguments by InitializationSettings.
  [(#57)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/57)

* Remove deprecated `set-output` commands from workflow files.
  [(#56)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/56)

* `setup.py` works on MacOS without `brew` (which is required by Conda-Forge runners).
  [(#48)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/48)

* MacOS::Intel wheels are built for the SERIAL and THREADS Kokkos backends.
  [(#48)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/48)

* Wheels are now checked with `twine check` post-creation for PyPI compatibility.
  [(#50)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/50)

* Template n-qubit gate methods.
  [(#40)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/40)

### Bug fixes

* Updates to use the new call signature for `QuantumScript.get_operation`.
  [(#52)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/52)


### Contributors

This release contains contributions from (in alphabetical order):

Ali Asadi, Lee James O'Riordan, Vincent Michaud-Rioux, Romain Moyard

---

# Release 0.29.1

### Improvements

* Use CMake `find_package` to bind pre-installed Kokkos libraries.
  [(#43)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/43)

### Bug fixes

* Ensure Kokkos finalize is only called at the end of process execution.
  [(#45)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/45)

### Contributors

This release contains contributions from (in alphabetical order):

Vincent Michaud-Rioux, Lee James O'Riordan

---

# Release 0.29.0

### New features since last release

 * Add support for building X86-64 Linux wheels with OpenMP and SERIAL backends with Github Actions.
 [(#14)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/14)

* Add the `kokkos_config` class variable, which stores the kokkos build and runtime information such as `Backend`, `Architecture`, `Kokkos Version`, `Compiler`, to LightningKokkos for users' query purpose. Users can also access other information such as `Options`, `Memory`, `Atomics` and `Vectorization` from `kokkos_config`.
  The workflow for build and runtime information query is:

  ```python
  >>> import pennylane as qml
  >>> dev = qml.device('lightning.kokkos', wires=3)
  >>> dev.kokkos_config["Backend"]
  {'Parallel': 'OpenMP'}
  ```
  [(#17)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/17)


### Breaking changes

* Change LightningKokkos to inherit from QubitDevice instead of LightningQubit. Numpy data initialization is decoupled.
  [(#31)] (https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/31)

### Improvements

* Use CMake `find_package` to bind pre-installed Kokkos libraries.
  [(#43)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/43)

* Update `inv()` methods in Python unit tests with `qml.adjoint()`.
  [(#33)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/33)

* Remove explicit Numpy requirement.
  [(#35)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/35)

* Add Kokkos::InitArguments support.
  [(#17)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/17)

* Add Nvidia GPU support for CI checks.
  [(#37)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/37)

* Add VJP support.
  [(#32)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/32)

### Bug fixes

* Ensure early-failure rather than return of incorrect results from out of order probs wires.
  [(#41)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/41)

* Fix the CI environment variables for building wheels with the OpenMP backend.
  [(#36)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/36)

* Fix the failures of pl_device_test tests with shots set.
  [(#38)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/38)

### Contributors

This release contains contributions from (in alphabetical order):

Amintor Dusko, Vincent Michaud-Rioux, Lee James O'Riordan, Shuli Shu

---

# Release 0.28.0

### Breaking changes

* Drop python3.7 and deprecate the Python and C++ tests with threading backend in workflows.
Note this deprecation is based on the fact that Kokkos cannot promise that its Threads back-end will
not conflict with the application's direct use of operating system threads.
  [(#23)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/23)

* Remove the unused `supports_reversible_diff` device capability from `LightningKokkos`
  [(#20)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/20)

### Improvements

* Improve the stopping condition method.
  [(#25)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/25)

### Documentation

* Update version string in package for release.
  [(#27)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/27)

### Bug fixes

* Avoid integer overflow in middle value calculation of binary search in `Sampler`.
  [#18](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/18)

### Contributors

This release contains contributions from (in alphabetical order):

Amintor Dusko, Lee J. O'Riordan, Shuli Shu, Matthew Silverman

--

# Release 0.27.0

### New features since last release

* Add probability support.
  [(#11)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/11)

* Add sample generation support.
  [(#9)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/9)

### Breaking changes


### Improvements

* Add tests for MacOS.
  [(#3)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/3)

* Update `LightningKokkos` device following changes in `LightningQubit` inheritance from `DefaultQubit` to `QubitDevice`.
  [(#16)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/16)

### Documentation

### Bug fixes

### Contributors

This release contains contributions from (in alphabetical order):

Amintor Dusko, Lee J. O'Riordan, Shuli Shu

---
# Release 0.26.0

 * Initial release. The PennyLane-Lightning-Kokkos device adds support for AMD-capable GPU simulation through use of the Kokkos library.
This release supports all base operations, including the adjoint differentation method for expectation value calculations.

This release contains contributions from:

Trevor Vincent, Shuli Shu, Lee James O'Riordan, Isidor Schoch, Josh Izaac, Amintor Dusko and Chae-Yeun Park.
