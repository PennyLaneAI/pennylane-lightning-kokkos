import os
import sys
import platform
import subprocess
import json

def record_backend():
    backends = {"CUDA", "HIP", "OPENMP", "THREADS", "SERIAL"}
    backend = None
    arch = None
    if os.getenv("BACKEND") and not backend:
        backend = os.getenv("BACKEND")
    if os.getenv("BACKEND") and not arch:
        arch = os.getenv("ARCH")

    built_info = {
            "Backend":f"{backend}",
            "Device_Arch":f"{arch}",
            "Platform":f"{platform.system()}"
    }
    with open("./pennylane_lightning_kokkos/built_info.json", "w") as f:
        json.dump(built_info, f)
    f.close()

def local_installation():
    subprocess.run(["python3.10", "-m", "pip", "install", "-e", "."])

def local_wheels_build():
    backend = None
    arch = None
    if os.getenv("BACKEND") and not backend:
        backend = os.getenv("BACKEND")
    subprocess.run(["python3.10", "setup.py", "build_ext", "--backend={backend}"])
    subprocess.run(["python3.10", "setup.py", "bdist_wheel"])

record_backend()

if os.getenv("LOCAL_WHEELS_BUILD"):
    local_wheels_build()

if os.getenv("LOCAL_INSTALL"):
    local_installation()