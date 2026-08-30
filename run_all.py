"""Run every replication script and report pass/fail.

Usage:  python run_all.py
Exits non-zero if any script fails.
"""
import pathlib
import subprocess
import sys

SCRIPTS = [
    "verify_appendix.py",
    "derive_model.py",
    "derive_bertrand.py",
    "derive_blp.py",
    "derive_uniqueness2.py",
    "derive_uniqueness3.py",
    "derive_soc2.py",
    "derive_twofuel.py",
    "derive_mandate_kkt.py",
    "derive_equivalence2.py",
    "derive_etsbase.py",
    "derive_fuelmarket2.py",
    "derive_welfare2.py",
    "derive_segments2.py",
    "passthrough.py",
    "calibrate_eu.py",
    "sweep.py",
]

here = pathlib.Path(__file__).parent / "scripts"
failed = []
for name in SCRIPTS:
    proc = subprocess.run([sys.executable, str(here / name)],
                          capture_output=True, text=True,
                          encoding='utf-8', errors='replace')
    status = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"{status}  {name}")
    if proc.returncode != 0:
        failed.append(name)
        print(proc.stderr.strip()[-500:])

print(f"\n{len(SCRIPTS) - len(failed)}/{len(SCRIPTS)} passed")
sys.exit(1 if failed else 0)
