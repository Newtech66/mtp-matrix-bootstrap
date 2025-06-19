class SiteBasis:
    def __init__(self, N):
        self._N = N
        self._ops = self.gen_basis()
        self._ops.sort()
        self._opset = set(self._ops)
   
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

    def enforce_rotation(self):
        exclude = set()
        L = self._N
        for i in range(4 ** L):
            if i in exclude:
                continue
            op = self.full_unrank(i)
            op2 = op
            for j in range(L - 1):
                op = op[1:] + op[0]
                if op == op2:
                    break
                exclude.add(self.full_rank(op))
        small_basis = []
        for i in range(4 ** L):
            if i in exclude:
                continue
            small_basis.append(self.full_unrank(i))
        return small_basis

    def enforce_reflection(self, basis):
        exclude = set()
        bset = set(basis)
        for word in bset:
            if word in exclude:
                continue
            rword = word[::-1]
            if word == rword:
                continue
            if rword not in bset:
                rword2 = rword[1:] + rword[0]
                while rword2 not in bset:
                    if rword2 == rword:
                        break
                    rword2 = rword2[1:] + rword2[0]
                if word != rword2:
                    exclude.add(rword2)
            else:
                exclude.add(rword)
        smaller_basis = []
        for word in basis:
            if word in exclude:
                continue
            smaller_basis.append(word)
        return smaller_basis

    def gen_basis(self):
        # small_basis = []
        # L = self._N
        # for i in range(4 ** L):
        #     small_basis.append(self.full_unrank(i))
        # return small_basis
        small_basis = self.enforce_rotation()
        return self.enforce_reflection(small_basis)
    
    def size(self):
        return len(self._ops)
    
    def rank(self, word: str):
        if len(word) != self._N:
            raise ValueError(f'{word} must be of length {self._N}')
        word = self.normalize(word)
        return self._ops.index(word)
        # return bisect.bisect_left(self._ops, word)

    def unrank(self, pos: int):
        # word = self.full_unrank(pos)
        # return self._ops[self.rank(word)]
        return self._ops[pos]

    def normalize(self, word):
        if len(word) != self._N:
            raise ValueError(f'{word} must be of length {self._N}')
        if word not in self._opset:
            word2 = word[1:] + word[0]
            while word2 not in self._opset:
                if word2 == word:
                    return self.normalize(word[::-1])
                word2 = word2[1:] + word2[0]
            return word2
        return word