"""Pinned B75 release installation and integrity checks, outside the repository."""
import hashlib
from pathlib import Path
import urllib.request

from distill.common import DATA, atomic_write

RELEASE = 'base-nf4-b75-v1'
HASHES = {
    'manifest.json': '5fd0bd9d5d759674fa27df5d0375d7dcbf093a2a57cf80897816376087534b93',
    'weights.bin': '4f84df2853aecb830b7cf4bee60572e08e2503232a25d7ebaaee0231930d6dd2',
}


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verified(directory):
    directory = Path(directory)
    return all((directory/name).is_file() and digest(directory/name) == expected
               for name, expected in HASHES.items())


def install(root=DATA):
    directory = Path(root)/'releases'/RELEASE
    for name, expected in HASHES.items():
        target = directory/name
        if target.is_file() and digest(target) == expected:
            continue
        def download(handle):
            checksum = hashlib.sha256()
            url = f'https://github.com/dunkean/neural-earth/releases/download/{RELEASE}/{name}'
            with urllib.request.urlopen(url, timeout=60) as response:
                while chunk := response.read(1024*1024):
                    checksum.update(chunk)
                    handle.write(chunk)
            if checksum.hexdigest() != expected:
                raise ValueError(f'B75 release checksum mismatch: {name}')
        atomic_write(target, download)
    return directory


if __name__ == '__main__':
    print(install())
