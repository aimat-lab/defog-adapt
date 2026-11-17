"""`morgan_condition`: the condition vector a fingerprint adapter is trained on and served with.

The encoding is a silent-failure surface. An adapter fed the encoding it was not trained on
receives out-of-distribution input, steers badly, and raises nothing; so the tests pin the exact
encoding (hashed counts through ``log1p``, or the plain bit vector) against RDKit's own older
fingerprint calls, which are what the recorded fingerprint results were produced with.
"""

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem, DataStructs  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from defog_adapt.domains import morgan_condition  # noqa: E402

SMILES = [
    "CC(=O)Nc1ccc(O)cc1",
    "CCOc1ccc(N)cc1",
    "O=C(N)c1cccnc1",
    "Clc1ccccc1CN1CCOCC1",
    "COc1ccc(CCN)cc1OC",
    "CN1CCCC1c1cccnc1",
    "CCCCCC",
    "c1ccc2c(c1)ccc1ccccc12",
]
legacy = pytest.mark.skipif(
    not (hasattr(AllChem, "GetHashedMorganFingerprint")
         and hasattr(AllChem, "GetMorganFingerprintAsBitVect")),
    reason="this RDKit has dropped the legacy fingerprint calls")


def _legacy_counts(mol, n_bits, radius):
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits), arr)
    return np.log1p(arr).astype(np.float32)


def _legacy_bits(mol, n_bits, radius):
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), arr)
    return arr


@legacy
@pytest.mark.parametrize("n_bits,radius", [(1024, 2), (512, 2), (2048, 3)])
def test_counts_equal_log1p_of_the_legacy_hashed_counts(n_bits, radius):
    for smi in SMILES:
        mol = Chem.MolFromSmiles(smi)
        got = morgan_condition(mol, n_bits=n_bits, radius=radius, counts=True)
        assert np.array_equal(got, _legacy_counts(mol, n_bits, radius)), smi


@legacy
@pytest.mark.parametrize("n_bits,radius", [(1024, 2), (512, 2), (2048, 3)])
def test_binary_equals_the_legacy_bit_vector(n_bits, radius):
    for smi in SMILES:
        mol = Chem.MolFromSmiles(smi)
        got = morgan_condition(mol, n_bits=n_bits, radius=radius, counts=False)
        assert np.array_equal(got, _legacy_bits(mol, n_bits, radius)), smi


def test_the_default_is_the_recorded_encoding():
    """1024 bits, radius 2, log1p counts."""
    mol = Chem.MolFromSmiles(SMILES[0])
    default = morgan_condition(mol)
    explicit = morgan_condition(mol, n_bits=1024, radius=2, counts=True)
    assert default.shape == (1024,) and np.array_equal(default, explicit)


def test_dtype_and_value_ranges():
    mol = Chem.MolFromSmiles(SMILES[1])
    counts = morgan_condition(mol, counts=True)
    bits = morgan_condition(mol, counts=False)
    assert counts.dtype == np.float32 and bits.dtype == np.float32
    assert (counts >= 0).all() and set(np.unique(bits)) <= {0.0, 1.0}
    assert bits.sum() > 0 and np.array_equal(counts > 0, bits > 0)


def test_counts_carry_repetition_that_the_bit_vector_loses():
    """Hexane repeats the same environment, so some count exceeds 1 (log1p > log 2 ~ 0.693),
    which a binary vector cannot express. This is what the count encoding is FOR."""
    mol = Chem.MolFromSmiles("CCCCCC")
    assert morgan_condition(mol, counts=True).max() > np.log1p(1.0) + 1e-6
    assert morgan_condition(mol, counts=False).max() == 1.0


def test_none_gives_zeros_of_the_requested_width():
    for n_bits in (256, 1024):
        out = morgan_condition(None, n_bits=n_bits)
        assert out.shape == (n_bits,) and out.dtype == np.float32 and not out.any()
