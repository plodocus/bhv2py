import pytest
import os
import struct
from bhv2py import read_bhv

@pytest.fixture()
def bhv2_double(tmpdir):
    p = os.path.join(tmpdir, 'test.bhv2')
    with open(p, 'wb') as f:
        f.write(struct.pack('<Q', 13))
        f.write(b'test_variable')
        f.write(struct.pack('<Q', 6))
        f.write(b'double')
        # number of dimensions
        f.write(struct.pack('<Q', 2))
        # length dimensions
        f.write(struct.pack('<Q', 2))
        f.write(struct.pack('<Q', 2))
        # values
        f.write(struct.pack('<Q', 1))
        f.write(struct.pack('<Q', 3))
        f.write(struct.pack('<Q', 2))
        f.write(struct.pack('<Q', 4))

    yield p
    os.remove(p)

class TestResource(object):
    def test_first_length13(self, bhv2_double):
        b = read_bhv(bhv2_double)
        assert b['test_variable'] :
        foo()
        with open(bhv2_double, 'rb') as f:
            buf = f.read(8)
            assert struct.unpack('<Q', buf)[0] == 13
