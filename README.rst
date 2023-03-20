PennyLane-Lightning-Kokkos Plugin
#################################

.. header-start-inclusion-marker-do-not-remove

The `PennyLane-Lightning-Kokkos <https://github.com/PennyLaneAI/pennylane-lightning-kokkos>`_ plugin extends the `Pennylane-Lightning <https://github.com/PennyLaneAI/pennylane-lightning>`_ state-vector simulator written in C++, and offloads to the `Kokkos library <https://github.com/kokkos/kokkos>`__ for accelerated circuit simulation.

`PennyLane <https://docs.pennylane.ai>`_ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

.. header-end-inclusion-marker-do-not-remove

.. installation-start-inclusion-marker-do-not-remove

Installation
============

.. code-block:: console

   cmake -Bbuild -DKokkos_ENABLE_OPENMP=On -DPLKOKKOS_BUILD_TESTS=On -G Ninja
   cmake --build ./build

You can install the python interface with:

.. code-block:: console

   python setup.py build_ext --backend="OPENMP"
   python setup.py bdist_wheel
   pip install ./dist/PennyLane*.whl --force-reinstall


or for an editable `pip` installation with:

.. code-block:: console

   BACKEND="OPENMP" python -m pip install -e .


Supported backend options are "SERIAL", "OPENMP", "THREADS", "HIP" and "CUDA". For "HIP" and "CUDA", the appropriate software stacks are required to enable compilation and subsequent use.
For explicit targeting of a given supported architecture, the environment variable `ARCH` can also be specified which directly sets the `-DKokkos_ARCH_{...}=ON` build option. Note that "THREADS"
backend is not recommended since `Kokkos <https://github.com/kokkos/kokkos-core-wiki/blob/17f08a6483937c26e14ec3c93a2aa40e4ce081ce/docs/source/ProgrammingGuide/Initialization.md?plain=1#L67>`_ does not guarantee its safety.

.. installation-end-inclusion-marker-do-not-remove

Testing
=======

To test with the ROCm stack using a manylinux2014 container we must first mount the repository into the container:

.. code-block:: console

    docker run -v `pwd`:/io -it quay.io/pypa/manylinux2014_x86_64 bash

Next, within the container, we install the ROCm software stack:

.. code-block:: console

    yum install -y https://repo.radeon.com/amdgpu-install/21.40.2/rhel/7.9/amdgpu-install-21.40.2.40502-1.el7.noarch.rpm
    amdgpu-install --usecase=hiplibsdk,rocm --no-dkms
    
We next build the test-suite, with a given AMD GPU target in mind, as listed `here <https://github.com/kokkos/kokkos/blob/master/Makefile.kokkos>`_.

.. code-block:: console

    cd /io
    export PATH=$PATH:/opt/rocm/bin/ 
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
    export CXX=/opt/rocm/hip/bin/hipcc 
    cmake -Bbuild -DCMAKE_CXX_COMPILER=/opt/rocm/hip/bin/hipcc -DKokkos_ENABLE_HIP=on -DPLKOKKOS_BUILD_TESTS=On -DKokkos_ARCH_VEGA90A=ON
    cmake --build ./build --verbose

We may now leave the container, and run the built test-suite on a machine with access to the targetted GPU.

For a system with access to the ROCm stack outside of a manylinux container, an editable `pip` installation can be built and installed as:

.. code-block:: console

   BACKEND="HIP" ARCH="VEGA90A" python -m pip install -e .


.. support-start-inclusion-marker-do-not-remove

Support
=======

- **Source Code:** https://github.com/PennyLaneAI/pennylane-lightning-kokkos
- **Issue Tracker:** https://github.com/PennyLaneAI/pennylane-lightning-kokkos/issues
- **PennyLane Forum:** https://discuss.pennylane.ai

If you are having issues, please let us know by posting the issue on our Github issue tracker, or
by asking a question in the forum.

.. support-end-inclusion-marker-do-not-remove
.. license-start-inclusion-marker-do-not-remove


License
=======

The PennyLane-Lightning-Kokkos plugin is **free** and **open source**, released under
the `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_. 
The PennyLane-Lightning-Kokkos plugin makes use of the `Kokkos <https://github.com/kokkos/kokkos>`__ library, which is held to their own respective licenses.

.. license-end-inclusion-marker-do-not-remove
.. acknowledgements-start-inclusion-marker-do-not-remove

Acknowledgements
================

The PennyLane Lightning Kokkos plugin makes use of the following libraries and tools, which are under their own respective licenses:

- **pybind11:** https://github.com/pybind/pybind11
- **Kokkos Core:** https://github.com/kokkos/kokkos

.. acknowledgements-end-inclusion-marker-do-not-remove
