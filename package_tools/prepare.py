"""Validate the source snapshot and link externally supplied pretrained assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared', type=Path, required=True)
    args = parser.parse_args()
    shared = args.shared.resolve()
    if not shared.is_dir():
        parser.error('The shared directory does not exist.')
    required = ['ViT-B-16.pt', 'region_teacher/C_RADIOv4_SO400M.safetensors']
    missing = [name for name in required if not (shared / name).is_file()]
    if missing:
        parser.error('Missing pretrained assets: ' + ', '.join(missing))
    manifest = json.loads((ROOT / 'configs/source_checksums.json').read_text())
    for name, metadata in manifest.items():
        path = ROOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['sha256']:
            raise RuntimeError('Source checksum mismatch: ' + name)
    link = ROOT / 'shared'
    if link.exists() or link.is_symlink():
        if link.resolve() != shared:
            raise RuntimeError('Existing shared path points elsewhere; not replacing it.')
    else:
        link.symlink_to(os.path.relpath(shared, ROOT), target_is_directory=True)
    print('Source verified and shared assets linked. No training or inference was run.')


if __name__ == '__main__':
    main()
