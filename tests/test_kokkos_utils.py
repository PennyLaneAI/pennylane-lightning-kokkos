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

from pennylane_lightning_kokkos.lightning_kokkos_qubit_ops import InitializationSettings


class TestKokkos:
    """Tests that Kokkos bindings work."""

    @pytest.mark.parametrize(
        "fields",
        [
            ("get_num_threads", 0),
            ("get_device_id", 0),
            ("get_map_device_id_by", ""),
            ("get_disable_warnings", 0),
            ("get_print_configuration", 0),
            ("get_tune_internals", 0),
            ("get_tools_libs", ""),
            ("get_tools_help", 0),
            ("get_tools_args", ""),
        ],
    )
    def test_InitializationSettings_getters(self, fields):
        """Tests that InitializationSettings fields are properly accessed."""
        args = InitializationSettings()
        field, default = fields
        assert getattr(args, field)() == default

    @pytest.mark.parametrize(
        "fields",
        [
            ("num_threads", 4),
            ("device_id", 1),
            ("map_device_id_by", "mpi_rank"),
            ("disable_warnings", True),
            ("print_configuration", True),
            ("tune_internals", True),
            ("tools_libs", "LD_LIBRARY_PATH"),
            ("tools_help", True),
            ("tools_args", "TOOL_ARGS"),
        ],
    )
    def test_InitializationSettings_setters(self, fields):
        """Tests that InitializationSettings fields are properly accessed."""
        args = InitializationSettings()
        field, value = fields
        _ = getattr(args, f"set_{field}")(value)
        assert getattr(args, f"get_{field}")() == value
