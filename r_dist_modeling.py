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
from scipy import stats
# In[Data distribuition]
in_data_count = [7,48,5,3,10,20,7]
in_data_R = [-2,-1,0,1,2,3,4]
data_freqt = pd.DataFrame([in_data_count,in_data_R]).T
data_freqt.columns = ['Trade_Count', 'R']
# In[DataGen]
lst = []
for idx, r in data_freqt.iterrows():
    f = r[0]
    v = r[1]
    lst += f*[v]
data = pd.DataFrame(lst, columns = ['R'])

# In[R Multiples Stats]
rhist = data['R'].plot.hist(title='R-Multiples', density=True)
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
N_sim = 10000
N_sample = np.sum(in_data_count)
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
          simqtl]
avgsim = pd.concat(simres,axis=1)
avgsim.columns = ['Avg Path','LB-2std','UB-2std',
                  f'{2.5}%-q',f'{97.5}%-q']
avgsim.plot(title='Cumulative R Simulations\nMean path\n'\
            f'N(paths)={N_sim}, N(sample) ={N_sample}', 
            style=['-','--','--','--','--'])
plt.grid(alpha=0.5, linestyle='--')

#--- simulation stats
(runs_myR.mean()/runs_myR.std()).mean()
(runs_myR.mean()/runs_myR.std()).std()
(runs_myR.mean()/runs_myR.std()).plot.hist(title =\
                                'Sharpe Ratios\nCum. R Sims', 
                                density = True)

# In[Position Sizing]
N_sim = 1000
N_sample = np.sum(in_data_count)
dic_bal = {}
for n in range(N_sim):
    sim_r = empirical_sample(recdf,N_sample)
    init_capital = 100 
    pos_size_pct = 0.002
    pnl = np.array([])
    balance = np.array([init_capital])
    for r in sim_r:
        trade_pos_size = pos_size_pct*balance[-1]
        trade_pnl = r*trade_pos_size
        pnl = np.append(pnl,trade_pnl)
        balance = np.append(balance,balance[-1]+trade_pnl)
    dic_bal[f'run{n+1}'] = balance

#--- equity curve sim results    
df_bal = pd.DataFrame(dic_bal)
df_bal.plot(title=f'Equity Curve Sim\n'\
              f'N(paths)={N_sim}, N(sample) ={N_sample}',
              legend=False)
plt.grid(alpha=0.5, linestyle='--')

# sharpe ratios dist
sharpes = np.array((df_bal.mean()-100)/df_bal.std())
plt.hist(sharpes, density = True)
plt.title('Sharpe Ratios\nEquity Curve Sims')
plt.show()

#--- sharpes stats
sharpes_mu = np.mean(sharpes)
sharpes_me = np.median(sharpes)
sharpes_mo = 3*sharpes_me - 2*sharpes_mu
(sharpes_mu, sharpes_me, sharpes_mo)
np.std(sharpes)
stats.mode(sharpes)

#--- original method for mode
sharpes_histogram = np.histogram(sharpes)
L = sharpes_histogram[1][np.argmax(sharpes_histogram[0])]
f1 = sharpes_histogram[0][np.argmax(sharpes_histogram[0])]
f0 = sharpes_histogram[0][np.argmax(sharpes_histogram[0])-1]
f2 = sharpes_histogram[0][np.argmax(sharpes_histogram[0])+1]
cls_i = np.diff(sharpes_histogram[1])[0]
mo = L + (f1-f0)/(2*f1-f0-f2)*cls_i
