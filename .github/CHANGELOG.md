# Release 0.28.0

 
### New features since last release


### Breaking changes

 * Drop python3.7 and deprecate the Python and C++ tests with threading backend in workflows.
 Note this deprecation is based on the fact that Kokkos cannot promise that its Threads back-end will 
 not conflict with the application's direct use of operating system threads. 
 [(#23)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/23)

  * Remove the unused `supports_reversible_diff` device capability from `LightningKokkos`
 [(#20)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/20)

### Improvements

### Documentation

### Bug fixes

* Avoid integer overflow in middle value calculation of binary search in `Sampler`.
[#18] (https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/18)

### Contributors

This release contains contributions from (in alphabetical order):

Shuli Shu, Matthew Silverman

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
