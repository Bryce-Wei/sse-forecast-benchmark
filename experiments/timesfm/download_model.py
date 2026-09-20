"""Download the fixed official checkpoint; verify every expected file hash.

This uses huggingface_hub's resumable download. Weights remain in the ignored
.cache directory. SSE_TIMESFM_CHECKPOINT can point to an existing local copy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
MODEL_ID = 'google/timesfm-2.5-200m-pytorch'
REVISION = '1d952420fba87f3c6dee4f240de0f1a0fbc790e3'
EXPECTED_SHA = '2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8'
EXPECTED_SIZE = 925181104
DEST = Path(os.environ.get('SSE_TIMESFM_CHECKPOINT', BASE / '.cache' / 'timesfm')).expanduser().resolve()
FILES = {
    'config.json': 'cd3315b760d5cc7e278d7afdf41b897031ced888fc6c115bd9b3ac0ea2c47408',
    'README.md': '4c3f16633a3d8b9324904bf004429aced2ee009b84e64effb59a902520cc7f2e',
    'model.safetensors': EXPECTED_SHA,
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    for name, expected in FILES.items():
        path = DEST / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Missing or invalid checkpoint file {name}; run download_model.py')
    assert (DEST / 'model.safetensors').stat().st_size == EXPECTED_SIZE
    return {'model_id': MODEL_ID, 'revision': REVISION, 'verified_files': FILES}


def main(verify_only=False):
    if not verify_only:
        from huggingface_hub import hf_hub_download
        DEST.mkdir(parents=True, exist_ok=True)
        for name, expected in FILES.items():
            path = DEST / name
            if path.is_file() and sha(path) == expected:
                continue
            hf_hub_download(repo_id=MODEL_ID, filename=name, revision=REVISION,
                            local_dir=str(DEST))
    print(json.dumps(verify(), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    main(parser.parse_args().verify_only)
