from heur import *
from heur_aux import *
from heur import Heuristic, StopCriterion
from heur_fsa import *
from heur_go import *

class FsaGoHeuristic(Heuristic):

	def __init__(self, of, maxevalFsa, maxevalGo, N, M, Tsel1, Tsel2, mutation, crossover, T0, n0, alpha):
		Heuristic.__init__(self, of, maxevalGo)
		self.N = N
		self.M = M
		self.Tsel1 = Tsel1
		self.Tsel2 = Tsel2
		self.mutation = mutation
		self.crossover = crossover
		self.T0 = T0
		self.n0 = n0
		self.alpha = alpha
		self.geneLenth = np.size(self.of.a)
		self.maxevalFsa = maxevalFsa
		self.maxevalGo = maxevalGo

	def search(self):
		fsa_neval = 0
		best_fsa_y = np.inf
		initial_population = np.zeros([self.N, self.geneLenth], dtype=int)
		for i in np.arange(self.N):
			fsa = FastSimulatedAnnealing(self.of, self.maxevalFsa, self.T0, self.n0, self.alpha, self.mutation)
			fsa_result = fsa.search()
			if fsa_result['neval'] < np.inf:
				fsa_result['neval'] += fsa_neval
				fsa_result['GO_boost'] = False
				return fsa_result
			initial_population[i] = fsa_result['best_x']
			fsa_neval += self.maxevalFsa
			if fsa_result['best_y'] < best_fsa_y:
				best_fsa_y = fsa_result['best_y']

		go = GeneticOptimization(self.of, self.maxevalGo, self.N, self.M, self.Tsel1, self.Tsel2, self.mutation, self.crossover)
		go_result = go.search()
		#goNeval = go_result['neval'] if go_result['neval'] < np.inf else self.maxevalGo
		#go_result['neval'] = fsaNeval + goNeval
		go_result['neval'] += fsa_neval
		if go_result['best_y'] < best_fsa_y:
			go_result['GO_boost'] = True
		else:
			go_result['GO_boost'] = False
		return go_result