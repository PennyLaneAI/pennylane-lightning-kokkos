# Release 0.29.0-dev

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

### Contributors

This release contains contributions from (in alphabetical order):

Amintor Dusko, Lee J. O'Riordan, Shuli Shu

---
# Release 0.26.0

 * Initial release. The PennyLane-Lightning-Kokkos device adds support for AMD-capable GPU simulation through use of the Kokkos library.
This release supports all base operations, including the adjoint differentation method for expectation value calculations.

This release contains contributions from:

Trevor Vincent, Shuli Shu, Lee James O'Riordan, Isidor Schoch, Josh Izaac, Amintor Dusko and Chae-Yeun Park.
