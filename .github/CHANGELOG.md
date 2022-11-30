# Release 0.28.0

 
### New features since last release

 * Add X86-64 Linux wheels for lightning.kokkos on CPU.
 [(#14)](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/pull/14)

### Breaking changes

### Improvements

### Documentation

### Bug fixes

### Contributors

This release contains contributions from (in alphabetical order):

Shuli Shu

---
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
 * Fix for errors caused by new lightning inheirtance implementation(DefaultQubit->QubitDevice).
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
