# Release 0.29.0-dev

### New features since last release

 * Add X86-64 Linux wheels building with OpenMP backend for lightning.kokkos in workflows.
 [(#14)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/14)

 * Add the build information record support by adding the `record_build_info()` method in `setup.py`.
 With the `record_build_info()` method, running before the `setup()` method, a json file `build_info.json` will be created and stored with the build information of backend, device architecture and platform. This json file is then included into the installed package. 
 [(#14)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/14)

 * Add the build information query support with the `_build_info(keyname)` method.
 Supported `keyname`s are `Backend`, `Device_Arch` and `Platform`.
 The workflow for `_build_info(keyname)`is:
 ```python
 >>> import pennylane as qml
 >>> import pennylane_lightning_kokkos as plk
 >>> dev = qml.device('lightning.kokkos', wires=3)
 >>> dev._build_info('Backend')
 "OPENMP"
 ```
 [(#14)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/14)

### Breaking changes

### Improvements

### Documentation

### Bug fixes

### Contributors

Shuli Shu

This release contains contributions from (in alphabetical order):

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
[#18] (https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/18)

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
 * Update this device following changes in LightningQubit inheritance from DefaultQubit to QubitDevice.
 [(#16)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/16)

### Contributors

This release contains contributions from (in alphabetical order):

Amintor Dusko, Lee J. O'Riordan, Shuli Shu

---
# Release 0.26.0

 * Initial release. The PennyLane-Lightning-Kokkos device adds support for AMD-capable GPU simulation through use of the Kokkos library.
This release supports all base operations, including the adjoint differentation method for expectation value calculations.

This release contains contributions from:

Trevor Vincent, Shuli Shu, Lee James O'Riordan, Isidor Schoch, Josh Izaac, Amintor Dusko and Chae-Yeun Park.
