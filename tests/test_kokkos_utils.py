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

from pennylane_lightning_kokkos.lightning_kokkos_qubit_ops import InitArguments


class TestKokkos:
    """Tests that Kokkos bindings work."""

    @pytest.mark.parametrize("num_threads", [None, 1, 2, 5])
    def test_InitArguments_init(self, num_threads):
        """Tests that InitArguments is properly initialized."""
        if num_threads is None:
            args = InitArguments()
            assert args.num_threads == -1
        else:
            args = InitArguments(num_threads)
            assert args.num_threads == num_threads

    @pytest.mark.parametrize(
        "fields",
        [
            ("num_threads", -1, 7),
            ("num_numa", -1, 7),
            ("device_id", -1, 7),
            ("ndevices", -1, 7),
            ("skip_device", 9999, 7),
            ("disable_warnings", False, True),
        ],
    )
    def test_InitArguments_readwrite_attrs(self, fields):
        """Tests that InitArguments fields are properly accessed."""
        args = InitArguments()
        field, default, value = fields
        assert getattr(args, field) == default
        setattr(args, field, value)
        assert getattr(args, field) == value

    def test_InitArguments_repr(self):
        """Tests that InitArguments are properly initialized."""
        r0 = """<InitArguments with
num_threads = 3
num_numa = -1
device_id = -1
ndevices = -1
skip_device = 9999
disable_warnings = 0>"""
        args = InitArguments()
        r1 = args.__repr__()
        assert r1.strip() != r0.strip()
        args = InitArguments(3)
        r1 = args.__repr__()
        assert r1.strip() == r0.strip()
