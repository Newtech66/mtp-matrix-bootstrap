import numpy as np
from sympy import symbols
from scipy.linalg import eigvalsh_tridiagonal
from mpmath import mp
import datetime
import subprocess
from contextlib import redirect_stdout

def solve(problem, Es):
    vals = []
    for E in Es:
        vals.append(problem.solve(E))
    return np.array(vals)

def solveTISE(xmin, xmax, N, potential, Emax):
    """
    Solve H = p^2 + V(x).
    """
    p = np.linspace(xmin, xmax, num=N)
    step = p[1] - p[0]
    d = 2 * np.ones(N) / step ** 2 + potential(p)
    e = -np.ones(N - 1) / step ** 2
    spectrum = eigvalsh_tridiagonal(d, e, select='v', select_range=[-1, Emax])
    return spectrum

def solveSUSYTISE(xmin, xmax, N, W2, Wdiff, Emax):
    """
    Solve H = p^2 + W^2(x) - Wdiff \sigma_3.
    """
    p = np.linspace(xmin, xmax, num=N)
    Wsq = W2(p)
    Wd = Wdiff(p)
    step = p[1] - p[0]
    d = 2 * np.ones(2 * N) / step ** 2 + np.hstack([Wsq - Wd, Wsq + Wd])
    e = -np.ones(2 * N - 1) / step ** 2
    spectrum = eigvalsh_tridiagonal(d, e, select='v', select_range=[-1, Emax])
    return spectrum

class SDPAWrapper:
    def __init__(self):
        self.status = None
        self.oval = None
        self.xval = None
    
    def run(self, file='aho'):
        args = ['sdpa_gmp',f'{file}.dat',f'{file}.result']
        subprocess.run(args, stdout=subprocess.DEVNULL)
        with open(f'{file}.result','r') as f:
            lines = f.readlines()
            for lpos, line in enumerate(lines):
                if line.startswith('phase.value'):
                    self.status = line.split(' ')[2]
                if line.startswith('objValPrimal'):
                    self.oval = np.double(line.split(' ')[2])
                if line.startswith('xVec'):
                    self.xval = np.array(lines[lpos + 1][1:-2].split(','), dtype=np.double)

# You do not need to create any more instances of this class
# SDPARunObject runs the data file and stores the status, objective value, and the values of the optimization variables
SDPARunObject = SDPAWrapper()

class Problem:
    def __init__(self, K, problem_name):
        self._K = K
        self._name = problem_name
        self._basis = None
        self._freevars = None
        self._E = symbols('E')

    def _initialize_basis(self):
        raise NotImplementedError

    def _implement_recursion(self):
        raise NotImplementedError

    def _get_free_variables(self):
        raise NotImplementedError

    def initialize_problem(self):
        self._initialize_basis()
        self._implement_recursion()
        self._get_free_variables()

class HankelProblem(Problem):
    def __init__(self, K, problem_name):
        super().__init__(K, problem_name)
        self._internal_matrix = None
    
    def _get_free_variables(self):
        self._freevars = set()
        for v1 in self._basis:
            for v2 in v1:
                if hasattr(v2, 'free_symbols'):
                    self._freevars |= v2.free_symbols
        # The energy is not a free variable
        self._freevars.discard(self._E)
        self._freevars = list(self._freevars)
    
    def _generate_internal_matrix_structure(self):
        raise NotImplementedError

    def initialize_problem(self):
        super().initialize_problem()
        self._generate_internal_matrix_structure()
        self._generate_unsubstituted_sdpa_matrices()

    def _generate_unsubstituted_sdpa_matrices(self):
        # [F0] + [F1,...,Fc] + [t]
        M = self._internal_matrix
        self._Fs = [np.zeros_like(M)] + [np.zeros_like(M) for _ in range(len(self._freevars))] + [np.eye(M.shape[0])]
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                term = M[i, j]
                if hasattr(term, 'expand'):
                    term = term.expand()
                    for p, var in enumerate(self._freevars, 1):
                        self._Fs[p][i, j] += term.coeff(var)
                        term = term.subs({var: 0})
                self._Fs[0][i, j] -= term

    def _write_input(self, energy, file):
        with redirect_stdout(file):
            print(f'*Energy: {energy}')
            print('*Potential name: ' + self._name)
            print(1 + len(self._freevars)) # mDIM
            print(1) # nBLOCK
            print(self._internal_matrix.shape[0]) # bLOCKsTRUCT
            c = np.zeros(1 + len(self._freevars))
            c[-1] = 1
            print(*c)
            for F in self._Fs:
                for i in range(self._internal_matrix.shape[0]):
                    for j in range(self._internal_matrix.shape[1]):
                        if hasattr(F[i, j], 'subs'):
                            print(mp.mpf(F[i, j].subs({self._E: energy})), end =' ')
                        else:
                            print(mp.mpf(F[i, j]), end=' ')
                print()

    def solve(self, energy, write_to_log=True):
        with open('not_opt.log','a') as logfile:
            with open('aho.dat','w') as datafile:
                self._write_input(energy, datafile)
            SDPARunObject.run()
            if write_to_log == True and SDPARunObject.status != 'pdOPT':
                logfile.write(f'[{datetime.datetime.now()}]: name={self._name} | K={self._K} | status={SDPARunObject.status} | energy={energy}\n')
            return SDPARunObject.oval
        
class SUSYProblem(HankelProblem):
    """
    Solves H = p^2 + A(x) - B(x)[P, P\dag].
    A(x) and B(x) are polynomials provided as coefficient lists.
    """
    def __init__(self, K, problem_name, A, B):
        super().__init__(K, problem_name)
        self._A = A
        self._dA = A.shape[0] - 1
        self._B = B
        self._dB = B.shape[0] - 1

    def _initialize_basis(self):
        self._basis = [None] * 2
        self._basis[0] = [symbols(f'x^{i}') for i in range(2*self._K)]
        self._basis[1] = [symbols(f'x^{i}PP\dag') for i in range(2*self._K)]

    def _implement_recursion(self):
        # Normalization condition
        self._basis[0][0] = self._basis[0][0].subs({'x^0': mp.mpf('1')})

        for t in range(1, len(self._basis[0]) - self._dA + 1):
            ti = mp.mpf(t)
            # Recursion for <x>
            self._basis[0][t + self._dA - 1] = 4*ti*self._E*self._basis[0][t-1]
            if t >= 3:
                self._basis[0][t + self._dA - 1] += ti*(ti-1)*(ti-2)*self._basis[0][t-3]
            for n in range(self._dA):
                self._basis[0][t + self._dA - 1] -= (4*ti + 2*n)*self._A[n]*self._basis[0][t+n-1]
            for n in range(self._dB + 1):
                self._basis[0][t + self._dA - 1] += (4*ti + 2*n)*self._B[n]*self._basis[1][t+n-1]
            self._basis[0][t + self._dA - 1] /= 2*self._A[-1]*(2*ti+self._dA)

            # Recursion for <xsigma3>
            self._basis[1][t + self._dA - 1] = 4*ti*self._E*self._basis[1][t-1]
            if t >= 3:
                self._basis[1][t + self._dA - 1] += ti*(ti-1)*(ti-2)*self._basis[1][t-3]
            for n in range(self._dA):
                self._basis[1][t + self._dA - 1] -= (4*ti + 2*n)*self._A[n]*self._basis[1][t+n-1]
            for n in range(self._dB + 1):
                self._basis[1][t + self._dA - 1] += (4*ti + 2*n)*self._B[n]*self._basis[0][t+n-1]
            self._basis[1][t + self._dA - 1] /= 2*self._A[-1]*(2*ti+self._dA)
    
    def _generate_internal_matrix_structure(self):
        M0 = np.array([[None for _ in range(self._K)] for __ in range(self._K)]) #<x>
        M1 = np.array([[None for _ in range(self._K)] for __ in range(self._K)]) #<xsigma3>
        Z = np.zeros_like(M0)

        for i in range(self._K):
            for j in range(self._K):
                # M0
                M0[i, j] = self._basis[0][i + j]
                # M1
                M1[i, j] = self._basis[1][i + j]

        self._internal_matrix = np.block(
            [[M0 + M1, Z],
             [Z, M0 - M1]])

    def _write_input(self, energy, file):
        with redirect_stdout(file):
            print(f'*Energy: {energy}')
            print('*Potential name: ' + self._name)
            print(1 + len(self._freevars)) # mDIM
            print(2) # nBLOCK
            N = self._internal_matrix.shape[0] // 2
            print(N, N) # bLOCKsTRUCT
            c = np.zeros(1 + len(self._freevars))
            c[-1] = 1
            print(*c)
            for F in self._Fs:
                for i in range(N):
                    for j in range(N):
                        if hasattr(F[i, j], 'subs'):
                            print(mp.mpf(F[i, j].subs({self._E: energy})), end =' ')
                        else:
                            print(mp.mpf(F[i, j]), end=' ')
                for i in range(N, 2 * N):
                    for j in range(N, 2 * N):
                        if hasattr(F[i, j], 'subs'):
                            print(mp.mpf(F[i, j].subs({self._E: energy})), end =' ')
                        else:
                            print(mp.mpf(F[i, j]), end=' ')
                print()

class CosineProblem(HankelProblem):
    """
    Solves H = p^2 + 1 - cosx.
    """
    def _initialize_basis(self):
        self._basis = [None, None]
        self._basis[0] = [symbols(f'Ree^i{n}x') for n in range(2*self._K)]
        self._basis[1] = [symbols(f'Ime^i{n}x') for n in range(2*self._K)]

    def _implement_recursion(self):
        # Normalization condition
        self._basis[0][0] = self._basis[0][0].subs({'Ree^i0x': mp.mpf('1')})
        self._basis[1][0] = self._basis[1][0].subs({'Ime^i0x': mp.mpf('0')})
        # self._basis[1][1] = self._basis[1][1].subs({'Ime^i1x': mp.mpf('0')})

        # Recursion for <e^inx>
        for t in range(len(self._basis[0]) - 1):
            print(t)
            ti = mp.mpf(t)
            # ti = t
            # self._basis[0][t + 1] = -(ti/(ti + 2)) * self._basis[0][t-1] - (ti ** 3 /(ti + 2)) * self._basis[0][t] + 2 * self._E * (ti/(ti + 2)) * self._basis[0][t] + 2* self._E * self._basis[0][t]/(ti + 2)
            self._basis[0][t + 1] = (ti ** 3 + 4 * (self._E - 1) * ti)*self._basis[0][t]
            self._basis[0][t + 1] -= (2 * ti - 1) * self._basis[0][t - 1]
            self._basis[0][t + 1] /= (2 * ti + 1)
            # self._basis[1][t + 1] = -(ti/(ti + 2)) * self._basis[1][t-1] - (ti ** 3 /(ti + 2)) * self._basis[1][t] + 2 * self._E * (ti/(ti + 2)) * self._basis[1][t] + 2* self._E * self._basis[1][t]/(ti + 2)
            self._basis[1][t + 1] = (ti ** 3 + 4 * (self._E - 1) * ti)*self._basis[1][t]
            self._basis[1][t + 1] -= (2 * ti - 1) * self._basis[1][t - 1]
            self._basis[1][t + 1] /= (2 * ti + 1)
    
    def _generate_internal_matrix_structure(self):
        R = np.array([[None for _ in range(self._K)] for __ in range(self._K)]) #<Ree^inx>
        C = np.array([[None for _ in range(self._K)] for __ in range(self._K)]) #<Ime^inx>
        # C = np.zeros_like(R)
        # [[R, -C]
        # [C, R]]
        for i in range(self._K):
            for j in range(self._K):
                print(i, j)
                R[i, j] = self._basis[0][np.abs(j - i)]
                C[i, j] = np.sign(j - i) * self._basis[1][np.abs(j - i)]
        self._internal_matrix = np.block([[R, -C],
                                          [C, R]])
        
