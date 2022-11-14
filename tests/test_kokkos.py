# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for Kokkos bindings.
"""
import pytest

from pennylane_lightning_kokkos import InitArguments


class TestKokkos:
    """Tests that Kokkos bindings work."""

    def test_InitArguments_init(self):
        """Tests that InitArguments is properly initialized."""
        args = InitArguments(1)
        assert args.num_threads == 1
        args = InitArguments(2)
        assert args.num_threads == 2
        args = InitArguments(5)
        assert args.num_threads == 5

    def test_InitArguments_init(self):
        """Tests that InitArguments fields are properly accessed."""
        args = InitArguments()
        # num_threads
        assert args.num_threads == -1
        args.num_threads = 7
        assert args.num_threads == 7
        # num_numa
        assert args.num_numa == -1
        args.num_numa = 7
        assert args.num_numa == 7
        # device_id
        assert args.device_id == -1
        args.device_id = 7
        assert args.device_id == 7
        # ndevices
        assert args.ndevices == -1
        args.ndevices = 7
        assert args.ndevices == 7
        # skip_device
        assert args.skip_device == 9999
        args.skip_device = 7
        assert args.skip_device == 7
        # disable_warnings
        assert args.disable_warnings == False
        args.disable_warnings = False
        assert args.disable_warnings == False

    def test_InitArguments_repr(self):
        """Tests that InitArguments are properly initialized."""
        args = InitArguments(3)
        r0 = args.__repr__()
        r1 = """<example.InitArguments with
 num_threads = 3
 num_numa = -1
 device_id = -1
 ndevices = -1
 skip_device = 9999
 disable_warnings = 0>"""
        assert r0.strip() == r1.strip()
