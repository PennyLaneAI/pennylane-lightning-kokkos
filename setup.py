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

import os
import platform
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, find_packages

if not os.getenv("READTHEDOCS"):
    from setuptools import Extension
    from setuptools.command.build_ext import build_ext

    class CMakeExtension(Extension):
        def __init__(self, name, sourcedir=""):
            Extension.__init__(self, name, sources=[])
            self.sourcedir = Path(sourcedir).absolute()

    class CMakeBuild(build_ext):
        """
        This class is based upon the build infrastructure of Pennylane-Lightning-Kokkos.
        """

        user_options = build_ext.user_options + [
            ("define=", "D", "Define variables for CMake"),
            ("verbosity", "V", "Increase CMake build verbosity"),
        ]

        def initialize_options(self):
            super().initialize_options()
            self.define = None
            self.verbosity = ""

        def finalize_options(self):
            # Parse the custom CMake options and store them in a new attribute
            defines = [] if self.define is None else self.define.split(";")
            self.cmake_defines = [f"-D{define}" for define in defines]
            if self.verbosity != "":
                self.verbosity = "--verbose"

            super().finalize_options()

        def build_extension(self, ext: CMakeExtension):
            extdir = str(Path(self.get_ext_fullpath(ext.name)).parent.absolute())
            debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
            cfg = "Debug" if debug else "Release"
            ninja_path = str(shutil.which("ninja"))

            configure_args = [
                "-DCMAKE_CXX_FLAGS=-fno-lto",
                "-DKokkos_ENABLE_SERIAL=ON",  # always build serial backend
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
                f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
                *(self.cmake_defines),
            ]
            configure_args += (
                [f"-DPYTHON_EXECUTABLE={sys.executable}"]
                if platform.system() == "Linux"
                else [f"-DPython_EXECUTABLE={sys.executable}"]
            )

            if platform.system() == "Windows":
                configure_args += [
                    "-T clangcl",
                ]
            else:
                configure_args += [
                    "-GNinja",
                    f"-DCMAKE_MAKE_PROGRAM={ninja_path}",
                ]

            # Add more platform dependent options
            if platform.system() == "Windows":
                configure_args += [
                    "-DKokkos_ENABLE_OPENMP=OFF"
                ]  # only build with Clang under Windows
            elif platform.system() not in ["Darwin", "Linux"]:
                raise RuntimeError(f"Unsupported '{platform.system()}' platform")

            if not Path(self.build_temp).exists():
                os.makedirs(self.build_temp)

            if "CMAKE_ARGS" not in os.environ.keys():
                os.environ["CMAKE_ARGS"] = ""

            subprocess.check_call(
                ["cmake"]
                + os.environ["CMAKE_ARGS"].split(" ")
                + [str(ext.sourcedir)]
                + configure_args,
                cwd=self.build_temp,
                env=os.environ,
            )
            subprocess.check_call(
                ["cmake", "--build", ".", "--verbose"],
                cwd=self.build_temp,
                env=os.environ,
            )


with open("pennylane_lightning_kokkos/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "pennylane>=0.30",
]

info = {
    "name": "PennyLane-Lightning-Kokkos",
    "version": version,
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "https://github.com/PennyLaneAI/pennylane-lightning-kokkos",
    "license": "Apache License 2.0",
    "packages": find_packages(where="."),
    "package_data": {
        "pennylane_lightning_kokkos": [os.path.join("src", "*"), os.path.join("src", "**", "*")]
    },
    "entry_points": {
        "pennylane.plugins": [
            "lightning.kokkos = pennylane_lightning_kokkos:LightningKokkos",
        ],
    },
    "description": "PennyLane-Lightning-Kokkos plugin",
    "long_description": open("README.rst").read(),
    "long_description_content_type": "text/x-rst",
    "provides": ["pennylane_lightning_kokkos"],
    "install_requires": requirements,
    "ext_package": "pennylane_lightning_kokkos",
}

if not os.getenv("READTHEDOCS"):
    info["ext_modules"] = [CMakeExtension("lightning_kokkos_qubit_ops")]
    info["cmdclass"] = {"build_ext": CMakeBuild}

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
