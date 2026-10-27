"""Atomic columnar resource caches with lossless memory-mapped arrays.

The loader retains support for old record caches; compiler 13 keys its columnar
format separately. No pickle or new dependency. JSON completes publication.
"""
import json
from pathlib import Path
import uuid
import zipfile
import time
import struct
import numpy as np


def mapped_arrays(path,names=None):
    """NPY members in a stored ZIP can be read directly, without inflation/copy.

    Return None for old compressed caches; their normal NumPy reader remains
    available. Only requested members are inspected for immutable atlas inputs.
    """
    result={}
    with zipfile.ZipFile(path) as archive,open(path,'rb') as stream:
        entries=[info for info in archive.infolist() if names is None or info.filename.removesuffix('.npy') in names]
        if any(info.compress_type != zipfile.ZIP_STORED for info in entries):return None
        # One mapping owns every array view; avoid hundreds of Windows mapping
        # handles and file opens. ndarray.base keeps it alive in each consumer.
        block=np.memmap(path,mode='r',dtype=np.uint8)
        for info in entries:
            name=info.filename.removesuffix('.npy')
            stream.seek(info.header_offset)
            header=stream.read(30)
            filename_bytes,extra_bytes=struct.unpack_from('<HH',header,26)
            stream.seek(info.header_offset+30+filename_bytes+extra_bytes)
            version=np.lib.format.read_magic(stream)
            reader=np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
            shape,fortran,dtype=reader(stream)
            if dtype.hasobject:raise ValueError('Object arrays are not resource inputs')
            result[name]=np.ndarray(shape,dtype=dtype,buffer=block,order='F' if fortran else 'C',offset=stream.tell())
    return result


def _replace(source, target):
    for attempt in range(40):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.025)


def save_world(root, key, fields, deposits):
    from terrain_resource_deposits import DepositTable
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    arrays = root / f'{key}-{nonce}.tmp.npz'
    metadata = root / f'{key}-{nonce}.tmp.json'
    try:
        # Stored NPY members make a cold viewer load lazy and zero-copy. The
        # columnar representation omits objects/derived stocks; legacy DEFLATE
        # caches are still read, but inflating every monthly field contends with
        # terrain generation and costs >0.3 s under viewer load.
        with zipfile.ZipFile(arrays, 'w', compression=zipfile.ZIP_STORED) as archive:
            stored=dict(fields)
            if isinstance(deposits,DepositTable):
                stored.update({'deposit__'+k:v for k,v in deposits.arrays.items()})
            for name, value in stored.items():
                if not isinstance(value, np.ndarray) or name.startswith(('stock_', 'recoverable_', '_')):
                    continue
                with archive.open(name + '.npy', 'w', force_zip64=True) as stream:
                    np.lib.format.write_array(stream, np.ascontiguousarray(value), allow_pickle=False)
        info=dict(format='columnar-v1',families=deposits.families,metals=deposits.metals) if isinstance(deposits,DepositTable) else deposits
        metadata.write_text(json.dumps(info, separators=(',', ':')), encoding='utf-8')
        _replace(arrays, root / f'{key}.npz')
        _replace(metadata, root / f'{key}.json')
    finally:
        arrays.unlink(missing_ok=True)
        metadata.unlink(missing_ok=True)
