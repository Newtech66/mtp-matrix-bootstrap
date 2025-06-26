from collections import Counter
from pauli import PauliSum


def gen_periodic(op: str, L: int, weight: float) -> Counter:
    k = len(op)
    eop = op + 'I' * (L - k)
    terms = Counter()
    for j in range(L):
        rop = eop[j:] + eop[:j]
        if j > 0 and rop == eop:
            break
        terms[rop] = weight
    return terms


def transverse_ising_field_model(L: int, h: float) -> PauliSum:
    """Returns the Hamiltonian"""
    hamil = Counter()
    hamil.update(gen_periodic('XX', L, -1.0))
    hamil.update(gen_periodic('Z', L, -h))
    return PauliSum(hamil)


def xyz_model(L: int, h: float) -> PauliSum:
    hamil = Counter()
    for p in ['X', 'Y', 'Z']:
        hamil.update(gen_periodic(p * 2, L, -1.0))
        hamil.update(gen_periodic(p, L, -h))
    return PauliSum(hamil)
