from bisect import bisect_left
import itertools as it
import functools as ft
from typing import Self
from collections import Counter

import numpy as np
from bitarray import bitarray
from bitarray.util import count_and

class SiteBasis:
    def __init__(self, L, use_rot=False, use_reflect=False):
        self._N = L
        self._opset = set()
        exclude = set()
        for o in it.product('IXYZ', repeat=L):
            op = ''.join(o)
            if op in exclude:
                continue
            self._opset.add(op)
            if use_rot:
                exclude.update(self.gen_rotations(op))
            if use_reflect:
                exclude.update(self.gen_rotations(op[::-1]))
        self._ops = list(self._opset)
        self._ops.sort()
   
    def full_rank(self, word):
        r = 0
        for i, op in enumerate(reversed(word)):
            if op not in 'IXYZ':
                raise ValueError(f'{word} must have only IXYZ')
            r += (4 ** i) * ('IXYZ'.find(op))
        return r

    def full_unrank(self, pos):
        word = [None] * self._N
        for i in range(self._N):
            word[i] = 'IXYZ'[pos % 4]
            pos //= 4
        return ''.join(word[::-1])

    def gen_rotations(self, op):
        rots = [op]
        for j in range(1, self._N):
            op2 = op[j:] + op[:j]
            if op2 == op:
                break
            rots.append(op2)
        return rots
    
    def size(self):
        return len(self._ops)
    
    def rank(self, word: str):
        if len(word) != self._N:
            raise ValueError(f'{word} must be of length {self._N}')
        return bisect_left(self._ops, self.normalize(word))

    def unrank(self, pos: int):
        return self._ops[pos]

    def normalize(self, word: str):
        if len(word) != self._N:
            raise ValueError(f'{word} must be of length {self._N}')
        # Cycle through all rotations
        for j in range(self._N):
            word2 = word[j:] + word[:j]
            if j > 0 and word == word2:
                break
            if word2 in self._opset:
                return word2
        # Normalize the mirror, always works
        return self.normalize(word[::-1])
    
# An efficient implementation of Pauli string operations (arXiV:2405.19287)
# z is bits[::2], x is bits[1::2]

class PauliString:
    CODEC = {'I': bitarray('00'),
             'Z': bitarray('10'),
             'X': bitarray('01'),
             'Y': bitarray('11'),}
    MAT = {'I': np.diag([1, 1]),
           'Z': np.diag([1, -1]),
           'X': np.fliplr(np.diag([1, 1])),
           'Y': np.fliplr(np.diag([-1j, 1j]))}
    def __init__(self, label: str | Self = None, bits: bitarray = None):
        self.bits = None
        self._str = None
        if not ((label is None) ^ (bits is None)):
            raise ValueError("exactly one of label or bits must be provided")
        if bits is None:
            if isinstance(label, str):
                self.bits = bitarray()
                self.bits.encode(self.CODEC, label)
                self._str = label
            elif isinstance(label, PauliString):
                self.bits = label.bits.copy()
            else:
                raise TypeError(f"Unsupported type {type(label)}")
        else:
            if isinstance(bits, bitarray):
                self.bits = bits.copy()
            else:
                raise TypeError(f"Unsupported type {type(bits)}")
    
    def __hash__(self):
        return hash(str(self.bits))

    def __repr__(self):
        if self._str is None:
            self._str = ''.join(self.bits.decode(self.CODEC))
        return self._str
    
    def __eq__(self, other):
        """Equality of two PauliStrings."""
        if not isinstance(other, PauliString):
            raise TypeError("other is not a PauliString")
        return self.bits == other.bits

    def __mul__(self, other: Self):
        """Multiply two PauliStrings."""
        if not isinstance(other, PauliString):
            raise TypeError("other is not a PauliString")
        return self.multiply(other)

    def __or__(self, other: Self):
        """Commutator of two PauliStrings."""
        if not isinstance(other, PauliString):
            raise TypeError("other is not a PauliString")
        return self.multiply(other) if not self.commutes_with(other) else None
    
    def __and__(self, other: Self):
        """Anticommutator of two PauliStrings."""
        if not isinstance(other, PauliString):
            raise TypeError("other is not a PauliString")
        return self.multiply(other) if self.commutes_with(other) else None
    
    def commutes_with(self, other: Self):
        """Check if two Pauli bitarrays commute."""
        if not isinstance(other, PauliString):
            raise TypeError("other is not a PauliString")
        return count_and(self.bits[::2], other.bits[1::2]) % 2 == count_and(self.bits[1::2], other.bits[::2]) % 2
    
    def phase(self, other: Self):
        """Find the phase of the product of two Pauli bitarrays."""
        if not isinstance(other, PauliString):
            raise TypeError("other is not a PauliString")
        return (-1j) ** ((2 * count_and(self.bits[1::2], other.bits[::2])
                          + count_and(self.bits[::2], self.bits[1::2])
                          + count_and(other.bits[::2], other.bits[1::2])
                          - count_and(self.bits[::2] ^ other.bits[::2], self.bits[1::2] ^ other.bits[1::2])) % 4)
    
    def multiply(self, other: Self):
        """Return the Pauli bitarray corresponding to the product of two Pauli bitarrays."""
        if not isinstance(other, PauliString):
            raise TypeError("other is not a PauliString")
        return PauliString(bits=self.bits ^ other.bits)
    
    def to_matrix(self):
        """Return the matrix corresponding to this Pauli string."""
        return ft.reduce(np.kron, (self.MAT[p] for p in self.__repr__()))

class PauliSum:
    # Consider implementing __iter__
    def __init__(self, labels_and_weights: dict[str, complex] | str | PauliString | Self):
        self.terms = None
        if isinstance(labels_and_weights, dict):
            self.terms = self.clean(Counter({PauliString(label): weight for label, weight in labels_and_weights.items()}))
        elif isinstance(labels_and_weights, str):
            self.terms = Counter({PauliString(labels_and_weights): 1.0})
        elif isinstance(labels_and_weights, PauliString):
            self.terms = Counter({PauliString(bits=labels_and_weights.bits): 1.0})
        elif isinstance(labels_and_weights, PauliSum):
            self.terms = labels_and_weights.terms.copy()
        else:
            raise TypeError(f"Unsupported type {type(labels_and_weights)}")
    
    def __hash__(self):
        raise NotImplementedError
    
    def __eq__(self, other: Self):
        if not isinstance(other, PauliSum):
            raise TypeError("other is not a PauliSum")
        if self.terms.keys() != other.terms.keys():
            return False
        for pstr, weight in self.terms.items():
            if not np.isclose(other.terms[pstr], weight):
                return False
        return True
    
    def __repr__(self):
        if self.is_zero():
            return f'{0j:.2f}'
        return '; '.join(f'{weight:.2f}*{string}' for string, weight in self.terms.items())
    
    def __iadd__(self, other: Self):
        if not isinstance(other, PauliSum):
            raise TypeError("other is not a PauliSum")
        for pstr, weight in other.terms.items():
            self.terms[pstr] += weight
            if np.isclose(self.terms[pstr], 0):
                del self.terms[pstr]
        return self
    
    def __add__(self, other: Self):
        res = PauliSum(self)
        res += other
        return res

    def __imul__(self, other: complex):
        if not isinstance(other, complex):
            raise TypeError("other is not a complex")
        for pstr in self.terms:
            self.terms[pstr] *= other
        return self

    def __rmul__(self, other: complex):
        if not isinstance(other, complex):
            raise TypeError("other is not a complex")
        res = PauliSum(self)
        res *= other
        return res

    def __mul__(self, other: Self | complex):
        """Multiply two PauliSums."""
        if isinstance(other, PauliSum):
            res = Counter()
            for l1, w1 in self.terms.items():
                for l2, w2 in other.terms.items():
                    res[l1 * l2] += l1.phase(l2) * w1 * w2
            return PauliSum(res)
        elif isinstance(other, complex):
            res = PauliSum(self)
            res *= other
            return res
        raise TypeError(f"Unsupported type {type(other)}")

    def __or__(self, other: Self):
        """Commutator of two PauliSums."""
        if not isinstance(other, PauliSum):
            raise TypeError("other is not a PauliSum")
        res = Counter()
        for l1, w1 in self.terms.items():
            for l2, w2 in other.terms.items():
                if not l1.commutes_with(l2):
                    res[l1 * l2] += 2 * l1.phase(l2) * w1 * w2
        return PauliSum(res)

    def __and__(self, other: Self):
        """Anticommutator of two PauliSums."""
        if not isinstance(other, PauliSum):
            raise TypeError("other is not a PauliSum")
        res = Counter()
        for l1, w1 in self.terms.items():
            for l2, w2 in other.terms.items():
                if l1.commutes_with(l2):
                    res[l1 * l2] += 2 * l1.phase(l2) * w1 * w2
        return PauliSum(res)
    
    def is_zero(self):
        return not self.terms
    
    @staticmethod
    def _pauli_ord(row: np.ndarray, col: np.ndarray, n: int) -> None:
        if n == 1:
            row[0], col[0] = 0, 0
            row[1], col[1] = 1, 1
            row[2], col[2] = 0, 1
            row[3], col[3] = 1, 0
            return
        PauliSum._pauli_ord(row, col, n - 1)
        pw = 1 << (2 * (n - 1))
        row[1 * pw: 2 * pw] = row[:pw] + (1 << (n - 1))
        col[1 * pw: 2 * pw] = col[:pw] + (1 << (n - 1))
        row[2 * pw: 3 * pw] = row[:pw]
        col[2 * pw: 3 * pw] = col[:pw] + (1 << (n - 1))
        row[3 * pw: 4 * pw] = row[:pw] + (1 << (n - 1))
        col[3 * pw: 4 * pw] = col[:pw]

    @staticmethod
    def _mat_to_vec(matrix: np.ndarray) -> np.ndarray:
        """
        Vectorizes a matrix in Pauli order.
                                                              [vec(A)]
        Given a matrix M = [A B], we vectorize it as vec(M) = [vec(D)].
                           [C D]                              [vec(B)]
                                                              [vec(C)]

        Args:
            matrix: The matrix to vectorize.
        """
        log2n = int(matrix.shape[0]).bit_length() - 1
        row = np.zeros(4 ** log2n, dtype=np.int64)
        col = np.zeros(4 ** log2n, dtype=np.int64)
        PauliSum._pauli_ord(row, col, log2n)
        flat_index = (1 << log2n) * row + col
        return matrix.reshape(-1)[flat_index].astype(np.complex128)

    @staticmethod
    def matrix_decomposition(matrix: np.ndarray) -> np.ndarray:
        """
        Return the weight vector corresponding to the Pauli basis decomposition of a matrix.
        The ordering is IZXY.

        Args:
            matrix: The matrix to be decomposed.
        """
        if matrix.ndim != 2:
            raise ValueError("matrix must be a 2D ndarray")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"expected square matrix but matrix dimensions \
                            are ({matrix.shape[0]}, {matrix.shape[1]})")
        if matrix.shape[0] == 1:
            raise ValueError("input must be a matrix, not a scalar")
        if int(matrix.shape[0]).bit_count() != 1:
            raise ValueError(f"expected square matrix with power of two \
                            dimensions but matrix dimensions are \
                            ({matrix.shape[0]}, {matrix.shape[1]})")
        b = PauliSum._mat_to_vec(matrix)
        h = 1
        while h < b.shape[0]:
            for i in range(0, b.shape[0], 4 * h):
                x, y = b[i : i + h], b[i + h : i + 2 * h]
                b[i : i + h], b[i + h : i + 2 * h] = (x + y) / 2, (x - y) / 2
                z, w = b[i + 2 * h : i + 3 * h], b[i + 3 * h : i + 4 * h]
                b[i + 2 * h : i + 3 * h], b[i + 3 * h : i + 4 * h] = (z + w) / 2, 1j * (z - w) / 2
            h *= 4
        return b

    @classmethod
    def from_matrix(cls, mat: np.ndarray):
        L = int(mat.shape[0]).bit_length() - 1
        wvec = PauliSum.matrix_decomposition(mat)
        obj = Counter()
        for i, op in enumerate(it.product('IZXY', repeat=L)):
            sop = ''.join(op)
            obj[sop] += wvec[i]
        return PauliSum(obj)

    def to_matrix(self):
        return sum(weight * pstr.to_matrix() for pstr, weight in self.terms.items())

    def clean(self, dirty_counter: Counter):
        """Remove zero terms."""
        return Counter({label: weight for label, weight in dirty_counter.items() if not np.isclose(weight, 0)})
