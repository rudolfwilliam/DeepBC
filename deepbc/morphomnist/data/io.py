"""This file is an exact copy of https://github.com/dccastro/Morpho-MNIST/morphomnist/io.py"""

import gzip
import struct
import numpy as np


def _load_uint8(f):
    idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]
    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


def _save_uint8(data, f):
    data = np.asarray(data, dtype=np.uint8)
    f.write(struct.pack('BBBB', 0, 0, 0x08, data.ndim))
    f.write(struct.pack('>' + 'I' * data.ndim, *data.shape))
    f.write(data.tobytes())


def save_idx(data: np.ndarray, path: str):
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'wb') as f:
        _save_uint8(data, f)


def load_idx(path: str) -> np.ndarray:
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'rb') as f:
        return _load_uint8(f)


if __name__ == '__main__': 
    MNIST_FILE = "/vol/biomedic/users/dc315/mnist/raw/train-images-idx3-ubyte.gz"
    TEST_FILE = "test"

    data = load_idx(MNIST_FILE)
    save_idx(data, TEST_FILE)
    data_ = load_idx(TEST_FILE)

    import os
    os.remove(TEST_FILE)

    assert np.alltrue(data == data_)
