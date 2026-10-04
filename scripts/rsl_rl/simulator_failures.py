"""Fail promptly when a GPU simulator backend is no longer usable."""
import os
from pathlib import Path
import sys
import traceback


def exit_on_fatal_simulator_error(error, log_dir, device):
    text = str(error).lower()
    fatal = any(marker in text for marker in (
        'illegal memory access', 'unspecified launch failure',
        'failed to get dof velocities from backend',
        'physx internal cuda error',
    ))
    if not str(device).startswith('cuda') or not fatal:
        return
    report = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
    report += '\nFatal GPU simulator error. Restart the process using the last completed checkpoint.\n'
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    # Write through a fresh file descriptor before bypassing native destructors.
    with open(Path(log_dir) / 'fatal_error.log', 'a', encoding='utf-8') as stream:
        stream.write(report)
        stream.flush()
        os.fsync(stream.fileno())
    sys.stderr.write(report)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
