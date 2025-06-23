from bisect import bisect_left
from itertools import product

class SiteBasis:
    def __init__(self, L, use_rot=False, use_reflect=False):
        self._N = L
        self._opset = set()
        exclude = set()
        for o in product('IXYZ', repeat=L):
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