from objfun import *
from heur_aux import *
import itertools

class SCP(ObjFun):

    def __init__(self, setCount, pointCount):
        np.random.seed(42)
        self.setCount = setCount
        self.pointCount = pointCount
        self.matrix = self.generate_problem(setCount, pointCount)
        self.fstar = self.find_fstar(setCount)
        ObjFun.__init__(self, self.fstar, np.zeros(setCount), np.ones(setCount))
        #self.elementWeights = np.array([ 2 ** (1 / (i + 1)) for i in np.arange(self.pointCount)])

    def find_fstar(self, n):
        fstar = np.inf
        for i in np.arange(2**n):
            s = np.binary_repr(i, width=n)
            x = np.array(list(s), dtype=int)
            f = self.evaluate(x)
            if f < fstar:
                fstar = f
        return fstar

    def get_matrix(self):
        return self.matrix

    def generate_problem(self, setCount, pointCount):
        return np.random.randint(low=0, high=2, size=(pointCount, setCount))

    def generate_point(self):
        return np.random.randint(low=0, high=2, size=self.setCount)

    def get_neighborhood(self, x, d):
        assert d == 1, "SCP supports neighbourhood with distance = 1 only"
        nd = []
        for i, xi in enumerate(x):
            xl = x.copy()
            xl[i] = 1 - x[i]
            nd.append(xl)
        return nd

    def evaluate(self, x):
        usedSetCount = np.sum(x)
        uncoveredElementCount = 0
        for j in range(self.pointCount):
            coveringSetCount = 0
            for k in range(self.setCount):
                coveringSetCount += self.matrix[j][k] * x[k]
            if coveringSetCount == 0:
                uncoveredElementCount += 1 #self.elementWeights[j]
        return usedSetCount + uncoveredElementCount

