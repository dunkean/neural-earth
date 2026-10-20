"""Persistent, content-addressed recipes for compact same-server map sharing."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import zlib

from flask import Blueprint, jsonify, request

UID = re.compile(r'[gr][A-Za-z0-9_-]{18}\Z')
MAX_BYTES = 128 * 1024


class UnknownShare(LookupError):
    pass


class ShareStore:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.Lock()

    @staticmethod
    def encode(kind, recipe):
        data = json.dumps(dict(version=1, kind=kind, recipe=recipe), sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode('utf-8')
        if len(data) > MAX_BYTES:
            raise ValueError('Share recipe is too large')
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
        return kind + digest[:18], data

    def save(self, kind, recipe):
        uid, data = self.encode(kind, recipe)
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / (uid + '.json')
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError('Share UID collision or damaged saved recipe')
                return uid
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return uid

    def load(self, uid, kind):
        if not isinstance(uid, str) or not UID.fullmatch(uid) or uid[0] != kind:
            raise ValueError('Invalid share UID')
        try:
            data = (self.root / (uid + '.json')).read_bytes()
        except FileNotFoundError:
            raise UnknownShare('Unknown share UID. Open this code on the server where it was saved, or include its portable recipe.') from None
        saved = json.loads(data)
        canonical_uid, canonical = self.encode(kind, saved['recipe'])
        if canonical_uid != uid or canonical != data:
            raise ValueError('Saved share recipe is damaged')
        return saved['recipe']


def register_share_routes(app, root, restore_generation=None):
    store = ShareStore(root)
    routes = Blueprint('terrain_share', __name__)

    @routes.post('/api/share')
    def save_share():
        if request.content_length is None or request.content_length > MAX_BYTES:
            return jsonify(error='Share recipe is too large'), 413
        try:
            data = request.get_json()
            generation, rendering = data['generation'], data['rendering']
            seed = generation['seed']
            if (not isinstance(seed, str) or not re.fullmatch(r'[0-9]{1,20}', seed)
                    or not 0 <= int(seed) < 2**64
                    or not isinstance(generation['profile'], str)
                    or not isinstance(generation['settings'], dict)
                    or not isinstance(rendering, dict)):
                raise ValueError('Invalid share recipe')
            from terrain_share_stages import export_stages
            generation = dict(generation, stages=export_stages(generation['settings']))
            return share_response(generation, rendering)
        except (ValueError, KeyError, TypeError) as error:
            return jsonify(error=str(error)), 400
        except (RuntimeError, OSError) as error:
            return jsonify(error=str(error)), 503

    def share_response(generation, rendering):
        portable = json.dumps(dict(version=1, generation=generation, rendering=rendering),
                              sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if len(portable) > MAX_BYTES:
            raise ValueError('Portable recipe is too large')
        return jsonify(generation=store.save('g', generation), rendering=store.save('r', rendering),
                       recipe=base64.urlsafe_b64encode(zlib.compress(portable, 9)).decode().rstrip('='))

    @routes.post('/api/share/import')
    def import_share():
        if request.content_length is None or request.content_length > MAX_BYTES * 2:
            return jsonify(error='Portable recipe is too large'), 413
        try:
            data = request.get_json()
            token = data['recipe']
            if not isinstance(token, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', token):
                raise ValueError('Invalid portable recipe')
            inflater = zlib.decompressobj()
            raw = inflater.decompress(base64.urlsafe_b64decode(token + '=' * (-len(token) % 4)), MAX_BYTES + 1)
            if len(raw) > MAX_BYTES or not inflater.eof or inflater.unused_data:
                raise ValueError('Portable recipe is too large or damaged')
            saved = json.loads(raw)
            if saved['version'] != 1:
                raise ValueError('Unsupported portable recipe version')
            generation, rendering = saved['generation'], saved['rendering']
            if (store.encode('g', generation)[0] != data['generation']
                    or store.encode('r', rendering)[0] != data['rendering']):
                raise ValueError('Portable recipe does not match its UIDs')
            from terrain_share_stages import restore_stages
            local_generation = (restore_generation or restore_stages)(generation)
            store.save('g', generation)
            store.save('r', rendering)
            return jsonify(generation=local_generation, rendering=rendering)
        except (ValueError, KeyError, TypeError, zlib.error) as error:
            return jsonify(error=str(error)), 400
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            return jsonify(error=str(error)), 503

    @routes.get('/api/share/<generation>/<rendering>')
    def load_share(generation, rendering):
        try:
            return jsonify(generation=store.load(generation, 'g'), rendering=store.load(rendering, 'r'))
        except UnknownShare as error:
            return jsonify(error=str(error)), 404
        except (ValueError, KeyError, TypeError) as error:
            return jsonify(error=str(error)), 400

    app.register_blueprint(routes)
