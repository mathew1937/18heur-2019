#!/usr/bin/env python
# coding: utf-8

# # Mixed heuristic approach
# 
# ## Fast Simulated Annealing (FSA) & Genetic Optimization (GO)
# 
# ### Author: Martin Zemko, HEUR, 2019

# ### Set Cover Problem
# What is the set cover problem?
# Idea:
# “You must select a minimum number [of any size set] of these sets so that the sets you have picked contain all the elements that are contained in any of the sets in the input (wikipedia).” 
# Additionally, you want to minimize the cost of the sets.
# 
# This task can be treated as a binary problem of linear programming, and its evaluation function is as follows:
# 
# $$f(x) = \sum_{i=1}^{m} x_i + \sum_{j=1}^{n} \lambda_i \cdot \mathrm{max} \left(1 - \sum_{k=1}^{m} S_{jk} \cdot x_k; 0 \right)$$
# 
# where
# 
# * m - set count
# * n - point count
# 
# <img src="img/scp.png">

# In[1]:


# Setting up enivironment
# Import path to source directory (bit of a hack in Jupyter)
import sys
import os
pwd = get_ipython().run_line_magic('pwd', '')
sys.path.append(os.path.join(pwd, os.path.join('..', 'src')))

# Ensure modules are reloaded on any change (very useful when developing code on the fly)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# Import external librarires
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

import matplotlib
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[3]:


# Inicialization of evaluating functions
def rel(x):
    return len([n for n in x if n < np.inf])/len(x)
def mne(x):
    return np.mean([n for n in x if n < np.inf])
def feo(x):
    return mne(x)/rel(x)
def go_boost(x):
    return np.sum(x) / len(x)


# Initialization of the **Set Cover Problem** task with paramters as follows:
# **``SCP(setCount=16, pointCount=16)``**
# 
# This task has 2^16 = 65 536 possible states

# In[4]:


# Import our code
from objfun_scp import SCP
scp = SCP(setCount=16, pointCount=16)


# ## Random Shooting and Steepest Descent heuristic

# In[5]:


from heur_sg import ShootAndGo


# In[7]:


NUM_RUNS = 100
maxeval = 1000


# In[8]:


def experiment_sg(of, maxeval, num_runs, hmax):
    results = []
    for i in tqdm_notebook(range(num_runs), 'Testing hmax={}'.format(hmax)):
        result = ShootAndGo(of, maxeval=maxeval, hmax=hmax).search() # dict with results of one run
        result['run'] = i
        result['heur'] = 'SG_{}'.format(hmax) # name of the heuristic
        result['hmax'] = hmax
        results.append(result)
    
    return pd.DataFrame(results, columns=['heur', 'run', 'hmax', 'best_x', 'best_y', 'neval'])


# In[9]:


results_sg = pd.DataFrame()
for hmax in [0, np.inf]:
    res = experiment_sg(of=scp, maxeval=maxeval, num_runs=NUM_RUNS, hmax=hmax)
    results_sg = pd.concat([results_sg, res], axis=0)


# In[10]:


stats_sg = results_sg.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_sg = stats_sg.reset_index()
stats_sg


# ## Fast Simulated Annealing heuristic

# In[11]:


from heur_fsa import FastSimulatedAnnealing
from heur_aux import BinaryMutation, Correction


# In[12]:


NUM_RUNS = 100
maxeval = 1000


# In[13]:


def experiment_fsa(of, maxeval, num_runs, T0, n0, alpha, p):
    results = []
    for i in tqdm_notebook(range(num_runs), 'Testing T0={}, n0={}, alpha={}, p={}'.format(T0, n0, alpha, p)):
        mut = BinaryMutation(p=p, correction=Correction(of))
        result = FastSimulatedAnnealing(of, maxeval=maxeval, T0=T0, n0=n0, alpha=alpha, mutation=mut).search()
        result['run'] = i
        result['heur'] = 'FSA_{}_{}_{}_{}'.format(T0, n0, alpha, p) # name of the heuristic
        result['T0'] = T0
        result['n0'] = n0
        result['alpha'] = alpha
        result['p'] = p
        results.append(result)
    
    return pd.DataFrame(results, columns=['heur', 'run', 'T0', 'n0', 'alpha', 'p', 'best_x', 'best_y', 'neval'])


# ### Optimization of the initial temperature T0

# In[14]:


results_fsa = pd.DataFrame()
for T0 in [1e-10, 1e-2, 1, np.inf]:
    res = experiment_fsa(of=scp, maxeval=maxeval, num_runs=NUM_RUNS, T0=T0, n0=1, alpha=1, p=0.10)
    results_fsa = pd.concat([results_fsa, res], axis=0)


# In[15]:


stats_fsa = results_fsa.pivot_table(
    index=['heur', 'T0'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_fsa = stats_fsa.reset_index()
stats_fsa.sort_values(by=['T0'])
stats_fsa


# ### Optimization of the mutation probability p

# In[16]:


results_fsa = pd.DataFrame()
for p in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
    res = experiment_fsa(of=scp, maxeval=maxeval, num_runs=NUM_RUNS, T0=1, n0=1, alpha=1, p=p)
    results_fsa = pd.concat([results_fsa, res], axis=0)


# In[17]:


stats_fsa = results_fsa.pivot_table(
    index=['heur', 'p'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_fsa = stats_fsa.reset_index()
stats_fsa.sort_values(by=['p'])
stats_fsa


# ### Optimization of the cooling parameters n0 and alpha

# In[20]:


results_fsa = pd.DataFrame()
for n0 in [1, 2, 5, 10, 100, np.inf]:
    res = experiment_fsa(of=scp, maxeval=maxeval, num_runs=NUM_RUNS, T0=1, n0=n0, alpha=1, p=0.10)
    results_fsa = pd.concat([results_fsa, res], axis=0)


# In[21]:


stats_fsa = results_fsa.pivot_table(
    index=['heur', 'n0'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_fsa = stats_fsa.reset_index()
stats_fsa.sort_values(by=['n0'])
stats_fsa


# In[26]:


results_fsa = pd.DataFrame()
for alpha in [1e-10, 1e-5, 1e-2, 1, 2, 5, 10]:
    res = experiment_fsa(of=scp, maxeval=maxeval, num_runs=NUM_RUNS, T0=1, n0=1, alpha=alpha, p=0.10)
    results_fsa = pd.concat([results_fsa, res], axis=0)


# In[27]:


stats_fsa = results_fsa.pivot_table(
    index=['heur', 'alpha'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_fsa = stats_fsa.reset_index()
stats_fsa.sort_values(by=['alpha'])
stats_fsa


# The optimal FSA parameters are following:
# * Initial temperature : T0 = 1
# * Mutation probability: p = 0.10
# * Cooling parameters  :
#     - n0 = 1
#     - alpha = 1

# In[28]:


mutation = BinaryMutation(p=0.10, correction=Correction(scp))
heur = FastSimulatedAnnealing(of=scp, maxeval=maxeval, T0=1, n0=1, alpha=1, mutation=mutation)
heur.reset()
result = heur.search()
print('neval = {}'.format(result['neval']))
print('best_x = {}'.format(result['best_x']))
print('best_y = {}'.format(result['best_y']))


# ## Genetic Optimization heuristic
# Let's optimize the GO heuristic in order to use the results in the mixed heuristic

# In[29]:


from heur_go import GeneticOptimization, UniformMultipoint
from heur_aux import BinaryMutation, Correction


# In[30]:


NUM_RUNS = 100
maxeval = 1000


# ### Optimization of the size of the population

# In[31]:


def experiment_go(of, maxeval, num_runs, N, M, Tsel1, Tsel2, mutation, crossover):
    results = []
    heur_name = 'GO_N={}'.format(N)
    for i in tqdm_notebook(range(num_runs), 'Testing {}'.format(heur_name)):
        result = GeneticOptimization(of, maxeval, N=N, M=M, Tsel1=Tsel1, Tsel2=Tsel2, 
                                     mutation=mutation, crossover=crossover).search()
        result['run'] = i
        result['heur'] = heur_name
        result['N'] = N
        results.append(result)
    return pd.DataFrame(results, columns=['heur', 'run', 'N', 'best_x', 'best_y', 'neval'])


# In[32]:


results_go = pd.DataFrame()
mutation = BinaryMutation(p=0.10, correction=Correction(scp))
crossover = UniformMultipoint(1)
for N in [1, 2, 3, 5, 10, 20, 30, 100]:
    res = experiment_go(of=scp, maxeval=maxeval, num_runs=NUM_RUNS, N=N, M=N*3, Tsel1=1e-10, Tsel2=1e-2, 
                        mutation=mutation, crossover=crossover)
    results_go = pd.concat([results_go, res], axis=0)


# In[33]:


stats_go = results_go.pivot_table(
    index=['heur', 'N'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_go = stats_go.reset_index()
stats_go.sort_values(by='N')
stats_go


# ### Optimization of the mutation probability
# Can we do better by changing the mutation probabilty?

# In[34]:


def experiment_go_2(of, maxeval, num_runs, p,N, M, Tsel1, Tsel2, mutation, crossover):
    results = []
    heur_name = 'GO_p={}'.format(p)
    for i in tqdm_notebook(range(num_runs), 'Testing {}'.format(heur_name)):
        result = GeneticOptimization(of, maxeval, N=N, M=M, Tsel1=Tsel1, Tsel2=Tsel2, 
                                     mutation=mutation, crossover=crossover).search()
        result['run'] = i
        result['heur'] = heur_name
        result['p'] = p
        results.append(result)
    return pd.DataFrame(results, columns=['heur', 'run', 'p', 'best_x', 'best_y', 'neval'])


# In[35]:


results_go_2 = pd.DataFrame()
crossover = UniformMultipoint(1)
N = 2
for p in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
    mutation = BinaryMutation(p=p, correction=Correction(scp))
    res = experiment_go_2(of=scp, maxeval=maxeval, num_runs=NUM_RUNS, p=p, N=N, M=N*3, Tsel1=1e-10, Tsel2=1e-2, 
                        mutation=mutation, crossover=crossover)
    results_go_2 = pd.concat([results_go_2, res], axis=0)


# In[36]:


stats_go_2 = results_go_2.pivot_table(
    index=['heur', 'p'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_go_2 = stats_go_2.reset_index()
stats_go_2.sort_values(by='p')
stats_go_2


# ### Optimization of the selection temperatures

# In[39]:


def experiment_go_3(of, maxeval, num_runs, p,N, M, Tsel1, Tsel2, mutation, crossover):
    results = []
    heur_name = 'GO_Tsel1={}_Tsel2={}'.format(Tsel1, Tsel2)
    for i in tqdm_notebook(range(num_runs), 'Testing {}'.format(heur_name)):
        result = GeneticOptimization(of, maxeval, N=N, M=M, Tsel1=Tsel1, Tsel2=Tsel2, 
                                     mutation=mutation, crossover=crossover).search()
        result['run'] = i
        result['heur'] = heur_name
        result['Tsel1'] = Tsel1
        result['Tsel2'] = Tsel2
        results.append(result)
    return pd.DataFrame(results, columns=['heur', 'run', 'Tsel1', 'Tsel2', 'best_x', 'best_y', 'neval'])


# In[43]:


results_go_3 = pd.DataFrame()
crossover = UniformMultipoint(1)
mutation = BinaryMutation(p=0.10, correction=Correction(scp))
N = 2
for Tsel in [1e-10, 1e-2, 1, np.inf]:
    res = experiment_go_3(of=scp, maxeval=maxeval, num_runs=NUM_RUNS, p=p, N=N, M=N*3, Tsel1=1e-10, Tsel2=Tsel, 
                        mutation=mutation, crossover=crossover)
    results_go_3 = pd.concat([results_go_3, res], axis=0)


# In[44]:


stats_go_3 = results_go_3.pivot_table(
    index=['heur', 'Tsel2'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_go_3 = stats_go_3.reset_index()
stats_go_3.sort_values(by='Tsel2')
stats_go_3


# The optimal GO parameters are following:
# * Size of the population: N = 2
# * Mutation probability  : p = 0.10
# * Selection temperatures:
#     - Tsel1 = 1e-10
#     - Tsel2 = 1e-2

# In[45]:


mutation = BinaryMutation(p=0.10, correction=Correction(scp))
heur = GeneticOptimization(of=scp, maxeval=maxeval, N=2, M=6, Tsel1=1e-10, Tsel2=1e-2, 
                        mutation=mutation, crossover=crossover)
result = heur.search()
print('neval = {}'.format(result['neval']))
print('best_x = {}'.format(result['best_x']))
print('best_y = {}'.format(result['best_y']))


# ## Fast Simulated Annealing + Genetic Optimization heursitic
# In this case, we will use FSA and GO parameters optimized separately in the previous section.

# In[46]:


from heur_go import UniformMultipoint
from heur_fsa_go import FsaGoHeuristic
from heur_aux import BinaryMutation, Correction


# ### Optimization of the size of the population
# We will find out whether the parameters from the GO heuristic are optimal in the combined heuristic as well

# In[47]:


NUM_RUNS = 100
maxevalFsa = 50
maxevalGo = 1000


# In[48]:


def experiment_fsa_go(of, maxevalFsa, maxevalGo, num_runs, N, M, Tsel1, Tsel2, mutation, crossover, T0, n0, alpha):
    results = []
    heur_name = 'FSA_GO_N={}'.format(N)
    for i in tqdm_notebook(range(num_runs), 'Testing {}'.format(heur_name)):
        result = FsaGoHeuristic(of=of, maxevalFsa=maxevalFsa, maxevalGo=maxevalGo, N=N, M=M, Tsel1=Tsel1, Tsel2=Tsel2, 
                                     mutation=mutation, crossover=crossover,
                                        T0=T0, n0=n0, alpha=alpha).search()
        result['run'] = i
        result['heur'] = heur_name
        result['N'] = N
        results.append(result)
    return pd.DataFrame(results, columns=['heur', 'run', 'N', 'best_x', 'best_y', 'neval', 'GO_boost'])


# In[49]:


results_fsa_go_1 = pd.DataFrame()
mutation = BinaryMutation(p=0.10, correction=Correction(scp))
crossover = UniformMultipoint(1)
for N in [1, 2, 3, 5, 10, 20, 50]:
    res = experiment_fsa_go(of=scp, maxevalFsa=maxevalFsa, maxevalGo=maxevalGo, num_runs=NUM_RUNS, N=N, M=N*3, Tsel1=1e-10, Tsel2=1e-2, 
                        mutation=mutation, crossover=crossover, T0=1, n0=1, alpha=1)
    results_fsa_go_1 = pd.concat([results_fsa_go_1, res], axis=0)


# In[50]:


stats_fsa_go_1A = results_fsa_go_1.pivot_table(
    index=['heur', 'N'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_fsa_go_1A = stats_fsa_go_1A.reset_index()
stats_fsa_go_1A.sort_values(by='N')
stats_fsa_go_1A


# In[51]:


stats_fsa_go_1B = results_fsa_go_1.pivot_table(
    index=['heur', 'N'],
    values=['GO_boost'],
    aggfunc=(go_boost)
)['GO_boost']
stats_fsa_go_1B = stats_fsa_go_1B.reset_index()
stats_fsa_go_1B.sort_values(by='N')
stats_fsa_go_1B


# The best combined heuristisic has only 2 elements in the population. Let's tune other parameters!

# ### Optimization of the FSA cooling factor

# In[52]:


def experiment_fsa_go_2(of, maxevalFsa, maxevalGo, num_runs, N, M, Tsel1, Tsel2, mutation, crossover, T0, n0, alpha):
    results = []
    heur_name = 'FSA_GO_n0={}'.format(n0)
    for i in tqdm_notebook(range(num_runs), 'Testing {}'.format(heur_name)):
        result = FsaGoHeuristic(of=of, maxevalFsa=maxevalFsa, maxevalGo=maxevalGo, N=N, M=M, Tsel1=Tsel1, Tsel2=Tsel2, 
                                     mutation=mutation, crossover=crossover,
                                        T0=T0, n0=n0, alpha=alpha).search()
        result['run'] = i
        result['heur'] = heur_name
        result['n0'] = n0
        results.append(result)
    return pd.DataFrame(results, columns=['heur', 'run', 'n0', 'best_x', 'best_y', 'neval', 'GO_boost'])


# In[53]:


results_fsa_go_2 = pd.DataFrame()
mutation = BinaryMutation(p=0.10, correction=Correction(scp))
crossover = UniformMultipoint(1)
for n0 in [1, 2, 5, 10]:
    res = experiment_fsa_go_2(of=scp, maxevalFsa=50, maxevalGo=maxevalGo, num_runs=NUM_RUNS, N=2, M=6, Tsel1=1e-10, Tsel2=1e-2,
                              mutation=mutation, crossover=crossover, T0=1, n0=n0, alpha=1)
    results_fsa_go_2 = pd.concat([results_fsa_go_2, res], axis=0)


# In[54]:


stats_fsa_go_2A = results_fsa_go_2.pivot_table(
    index=['heur', 'n0'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_fsa_go_2A = stats_fsa_go_2A.reset_index()
stats_fsa_go_2A.sort_values(by='n0')
stats_fsa_go_2A


# In[55]:


stats_fsa_go_2B = results_fsa_go_2.pivot_table(
    index=['heur', 'n0'],
    values=['GO_boost'],
    aggfunc=(go_boost)
)['GO_boost']
stats_fsa_go_2B = stats_fsa_go_2B.reset_index()
stats_fsa_go_2B.sort_values(by='n0')
stats_fsa_go_2B


# Optimization of the mixed heuristic provides the same results as optimization of the separated FSA and GO heuristics.

# ### Finding the FSA cut off

# In[56]:


def experiment_fsa_go_3(of, maxevalFsa, maxevalGo, num_runs, N, M, Tsel1, Tsel2, mutation, crossover, T0, n0, alpha):
    results = []
    heur_name = 'FSA_GO_{}'.format(maxevalFsa)
    for i in tqdm_notebook(range(num_runs), 'Testing {}'.format(heur_name)):
        result = FsaGoHeuristic(of=of, maxevalFsa=maxevalFsa, maxevalGo=maxevalGo, N=N, M=M, Tsel1=Tsel1, Tsel2=Tsel2, 
                                     mutation=mutation, crossover=crossover,
                                        T0=T0, n0=n0, alpha=alpha).search()
        result['run'] = i
        result['heur'] = heur_name
        result['maxevalFsa'] = maxevalFsa
        results.append(result)
    return pd.DataFrame(results, columns=['heur', 'run', 'maxevalFsa', 'best_x', 'best_y', 'neval', 'GO_boost'])


# In[57]:


results_fsa_go_3 = pd.DataFrame()
mutation = BinaryMutation(p=0.10, correction=Correction(scp))
crossover = UniformMultipoint(1)
for maxevalFsa in [2, 5, 10, 20, 30, 50, 100, 200, 500]:
    res = experiment_fsa_go_3(of=scp, maxevalFsa=maxevalFsa, maxevalGo=maxevalGo, num_runs=NUM_RUNS, N=2, M=N*3, Tsel1=1e-10, Tsel2=1e-2, 
                        mutation=mutation, crossover=crossover, T0=1, n0=1, alpha=1)
    results_fsa_go_3 = pd.concat([results_fsa_go_3, res], axis=0)


# In[58]:


stats_fsa_go_3A = results_fsa_go_3.pivot_table(
    index=['heur', 'maxevalFsa'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_fsa_go_3A = stats_fsa_go_3A.reset_index()
stats_fsa_go_3A.sort_values(by='maxevalFsa')
stats_fsa_go_3A


# In[59]:


stats_fsa_go_3B = results_fsa_go_3.pivot_table(
    index=['heur', 'maxevalFsa'],
    values=['GO_boost'],
    aggfunc=(go_boost)
)['GO_boost']
stats_fsa_go_3B = stats_fsa_go_3B.reset_index()
stats_fsa_go_3B.sort_values(by='maxevalFsa')
stats_fsa_go_3B


# With respect to the result, we should cut off the FSA initialization after 500 evaluations (or maybe more). However, it means that we should completely eliminate the Geneteric Optimization and let the FSA heuristic calculate the solution.

# The optimal parameters of the mixed heuristic are following:
# * FSA max evaluations    : >500
# * Initial temperature    : T0 = 1
# * Mutation probability   : p = 0.10
# * Size of the population : N = 2
# * Cooling parameters :
#     - n0 = 1
#     - alpha = 1
# * Selection temperatures :
#     - Tsel1 = 1e-10
#     - Tsel2 = 1e-2

# In[60]:


mutation = BinaryMutation(p=0.10, correction=Correction(scp))
heur = FsaGoHeuristic(of=scp, maxevalFsa=500, maxevalGo=maxevalGo, N=2, M=6, Tsel1=10e-10, Tsel2=1e-2, 
                                     mutation=mutation, crossover=crossover,
                                        T0=1, n0=1, alpha=1)
result = heur.search()
print('neval = {}'.format(result['neval']))
print('best_x = {}'.format(result['best_x']))
print('best_y = {}'.format(result['best_y']))


# ## Results

# Shoot and Go result:

# In[61]:


stats_sg


# Fast Simulated Annealing heuristic result:

# In[62]:


stats_fsa.sort_values(by=['feo']).head(1)


# Genetic Optimization heuristic result:

# In[63]:


stats_go_2.sort_values(by=['feo']).head(1)


# Mixed (GO + FSA) heuristic result:

# In[64]:


stats_fsa_go_3A.sort_values(by=['feo']).head(1)


# ## Conclusion

# 1. Based on the FEO values, the best heuristics solving the SCP problem is the FSA heuristics.
# 2. The mixed heuristics is the second best heuristics because it involves the FSA in the first phase. Moreover, its optimization process eliminates the second phase (i.e. the GO heuristics). Thus, the result is calculated only by the FSA.
# 3. Using the mixed approach, we can achieve better results than in the RandomShooting heuristics. However, it is because of the built-in FSA taking control over the problem.
# 4. The mixed approach is definitely not better than FSA in this particular case.
# 5. The mixed heuristics can be useful only for making decision about which heuristics should be used (FSA or GO). Besides that, this heuristics does not provide any additional benefits.

# In[ ]:




