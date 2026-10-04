"""Record the libraries actually loaded by Kit, not only Conda package metadata."""
import json
from pathlib import Path
import sys


def save_runtime_diagnostics(log_dir):
    import torch
    import warp

    info = {
        'python': sys.version,
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'warp': warp.__version__,
        'warp_path': warp.__file__,
    }
    maps = Path('/proc/self/maps')
    if maps.exists():
        info['cuda_library_paths'] = sorted({
            line.split()[-1] for line in maps.read_text().splitlines()
            if any(name in line.lower() for name in ('libcuda', 'libnvrtc', '/warp.so', 'physxgpu'))
        })
    driver = Path('/proc/driver/nvidia/version')
    if driver.exists():
        info['nvidia_kernel_module'] = driver.read_text().strip()
    destination = Path(log_dir) / 'runtime.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f"[Runtime] Torch {torch.__version__} / CUDA {torch.version.cuda}; Warp {warp.__version__}: {warp.__file__}")
    return info
