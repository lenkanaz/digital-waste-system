import pytest
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import compute_entropy, compute_co2

CFG = {
    'co2': {
        'kwh_per_gb': 0.000849,
        'carbon_tr': 0.475,
        'pue': 1.58
    }
}

def test_entropy_uniform():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(bytes(range(256)) * 16)
        path = f.name
    result = compute_entropy(open(path, 'rb').read(4096))
    os.unlink(path)
    assert abs(result - 8.0) < 0.01

def test_co2_zero():
    assert compute_co2(0, CFG) == 0.0

def test_co2_1gb():
    result = compute_co2(1e9, CFG)
    assert 0.6 < result < 0.8