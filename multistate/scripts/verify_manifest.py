#!/usr/bin/env python3
"""Check published SHA-256 hashes before regenerating derived files."""
from pathlib import Path
import hashlib

root = Path(__file__).resolve().parents[1]
count = 0
failures = []
for line in (root/'SHA256SUMS').read_text().splitlines():
    digest, name = line.split('  ',1)
    path = root/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        failures.append(name)
    count += 1
if failures:
    raise SystemExit('Integrity check failed: '+', '.join(failures))
print(f'PASS: {count} distributed file hashes.')
