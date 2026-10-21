"""Pinned native3 base release; installation stays outside the repository."""
import hashlib
import importlib.util
from pathlib import Path
import tarfile
import urllib.request

from distill.common import DATA, atomic_write

RELEASE = 'native3-base99m-v1'
ARCHIVE_SHA256 = '247d2f8f8424b11de1af3dd4be954ebeec4ec7b43b4dd2b00ec433fb88227f5c'
HASHES = {
    'manifest.json': '811a7d0661d188867a0caff39def0f032beb8cd6eb07da472e10b6869282e736',
    'load_base.py': '784c34e372161216447e33dea21afb425e826f6bce79a8f37c9bff61f1069d95',
    'weights.bin': 'e12fffc05ecfcdd389c08deada7f3a760769dcdc41829908725c362909671379',
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
    if verified(directory/'base'):
        return directory/'base'
    archive = directory/(RELEASE+'.tar')
    if not archive.is_file() or digest(archive) != ARCHIVE_SHA256:
        def download(handle):
            url = f'https://github.com/dunkean/neural-earth/releases/download/{RELEASE}/{RELEASE}.tar'
            with urllib.request.urlopen(url, timeout=60) as response:
                while chunk := response.read(1024*1024):
                    handle.write(chunk)
        atomic_write(archive, download)
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError('Native3 archive checksum mismatch.')
    with tarfile.open(archive) as source:
        source.extractall(directory, filter='data')
    if not verified(directory/'base'):
        raise ValueError('Native3 payload checksum mismatch.')
    return directory/'base'


def load(directory, device):
    directory = Path(directory)
    if not verified(directory):
        raise ValueError('Native3 release changed; reinstall it.')
    spec = importlib.util.spec_from_file_location('neural_earth_native3_loader', directory/'load_base.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    model, _ = loader.load_packed(directory, device=device)
    return model


if __name__ == '__main__':
    print(install())
