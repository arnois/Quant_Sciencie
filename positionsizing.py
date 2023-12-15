# -*- coding: utf-8 -*-
"""
Position sizing analysis

@author: jquintero
"""

# In[Modules]
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
# In[Data]
strtmp = 'pab_rmults.xlsx'
data = pd.read_excel(strtmp)
# In[R Multiples Stats]
rhist = data['R'].plot.hist(title='R-Multiples', density=True)
cumr = data['R'].cumsum().plot(title='Cumulative R-Multiples', color='darkcyan')
plt.grid(alpha=0.5, linestyle='--')
recdf = ECDF(data['R'])
# In[R's Sampling]
#--- random samples generator from empirical distribution
def empirical_sample(ecdf, size):
    u = np.random.uniform(0,1,size)
    ecdf_x = np.delete(ecdf.x,0)
    sample = []
    for u_y in u:
        idx = np.argmax(ecdf.y >= u_y)-1
        e_x = ecdf_x[idx]
        sample.append(e_x)
    return pd.Series(sample,name='emp_sample')

#--- R paths generator from empirical distribution
def sim_path_R(ecdf, sample_size=1000, paths_size=1000):
    runs = []
    for n in range(paths_size):
        run = empirical_sample(ecdf,sample_size)
        run.name = run.name + f'{n+1}'
        runs.append(run.cumsum())
    df_runs = pd.concat(runs, axis = 1)
    df_runs.index = df_runs.index + 1
    df_runs = pd.concat([pd.DataFrame(np.zeros((1,paths_size)),
                                      columns=df_runs.columns),
               df_runs])
    return df_runs
# In[R-Paths Sampling]
#--- R-multiples simulation
N_sim = 1000
N_sample = 750
runs_myR = sim_path_R(recdf, sample_size=N_sample, paths_size=N_sim)
runs_myR.plot(title=f'Cumulative R Simulations\n'\
              f'N(paths)={N_sim}, N(sample) ={N_sample}',
              legend=False)
plt.grid(alpha=0.5, linestyle='--')
#--- simulation results
simmean = runs_myR.mean(axis=1)
simstd = runs_myR.std(axis=1)
simqtl = runs_myR.quantile(q=(0.025,1-0.025),axis=1).T
simres = [simmean, simmean-2*simstd, simmean+2*simstd, 
          simqtl, data['R'].cumsum()]
avgsim = pd.concat(simres,axis=1)
avgsim.columns = ['Avg Path','LB-2std','UB-2std',
                  f'{2.5}%-q',f'{97.5}%-q','Current']
avgsim.plot(title='Cumulative R Simulations\nMean path\n'\
            f'N(paths)={N_sim}, N(sample) ={N_sample}', 
            style=['-','--','--','--','--'])
plt.grid(alpha=0.5, linestyle='--')


