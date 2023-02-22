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

    @pytest.mark.parametrize("init_threads", [None, 1, 2, 5])
    @pytest.mark.parametrize(
        "fields",
        [
            ("num_threads", -1, 2),
            ("num_numa", -1, 3),
            ("device_id", -1, 4),
            ("ndevices", -1, 5),
            ("skip_device", 9999, 6),
            ("disable_warnings", False, True),
        ],
    )
    def test_InitArguments_readwrite_attrs(self, init_threads, fields):
        """Tests that InitArguments fields are properly accessed."""
        field, default, value = fields
        if init_threads is None:
            args = InitArguments()
        else:
            args = InitArguments(init_threads)
            default = init_threads if field == "num_threads" else default
        assert getattr(args, field) == default
        setattr(args, field, value)
        assert getattr(args, field) == value

    @pytest.mark.parametrize("init_threads", list(range(1, 5)))
    def test_InitArguments_repr(self, init_threads):
        """Tests that InitArguments are properly initialized."""
        r0 = f"""<InitArguments with
num_threads = {init_threads}
num_numa = -1
device_id = -1
ndevices = -1
skip_device = 9999
disable_warnings = 0>"""
        args = InitArguments()
        r1 = args.__repr__()
        assert r1.strip() != r0.strip()
        args = InitArguments(init_threads)
        r1 = args.__repr__()
        assert r1.strip() == r0.strip()
