"""
Objective: 
    Quant model research for trading/investment strategies over fixed
    income securities. Particularly, over US, MX and cross asset classes.

# Author: Arnulf (arnulf.q@gmail.com)
"""
###############################################################################
# System and path functions modules
###############################################################################
import os
import sys
str_cwd = r'H:\Python\\'
os.chdir(str_cwd)
sys.path.append(str_cwd)
###############################################################################
# Modules
###############################################################################
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import ticker as tkr
from matplotlib.style import available as plt_av
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")
import pickle
pd.set_option('display.float_format', lambda x: '%0.4f' % x)
pd.set_option('display.max_columns', 7)
plt_styles = [s for s in plt_av if 'seaborn' in s]
###############################################################################
# UDF
###############################################################################
from udf_trading_hyperdrive import *
###############################################################################
# Data
###############################################################################
# daily dataset import
str_path = r'H:\db'
str_file = r'\data_1D.csv'
data = csvImport(str_path, str_file)
updtc = ['TIIE1Y1Y','TIIE2Y1Y','TIIE2Y5Y','TIIE5Y5Y','TIIE3Y1Y',
         'TIIE2Y2Y','TIIE3Y2Y','TIIE4Y1Y','TIIE5Y2Y']
###############################################################################
# Features
###############################################################################
## spreads
data['T5s7s10s'] = (2*data['TIIE7Y']-data['TIIE5Y']-data['TIIE10Y'])*100
data['T5s10s'] = (data['TIIE10Y']-data['TIIE5Y'])*100
data['T4s5s'] = (data['TIIE5Y']-data['TIIE4Y'])*100
data['T4s7s'] = (data['TIIE7Y']-data['TIIE4Y'])*100
data['T2s5s'] = (data['TIIE5Y']-data['TIIE2Y'])*100
data['T2s10s'] = (data['TIIE10Y']-data['TIIE2Y'])*100
data['T3s7s'] = (data['TIIE7Y']-data['TIIE3Y'])*100
data['T2s3s4s'] = (2*data['TIIE3Y']-data['TIIE2Y']-data['TIIE4Y'])*100
data['T2s5s10s'] = (2*data['TIIE5Y']-data['TIIE2Y']-data['TIIE10Y'])*100
data['T2s3s5s'] = (2*data['TIIE3Y']-data['TIIE2Y']-data['TIIE5Y'])*100
data['T3s5s7s'] = (2*data['TIIE5Y']-data['TIIE3Y']-data['TIIE7Y'])*100
data['T3s5s10s'] = (2*data['TIIE5Y']-data['TIIE3Y']-data['TIIE10Y'])*100
data['T4s5s7s'] = (2*data['TIIE5Y']-data['TIIE4Y']-data['TIIE7Y'])*100
data['T3s4s5s'] = (2*data['TIIE4Y']-data['TIIE3Y']-data['TIIE5Y'])*100
data['T3s7s10s'] = (2*data['TIIE7Y']-data['TIIE3Y']-data['TIIE10Y'])*100
data['2Y1Y vs 3Y1Y'] = (data['TIIE3Y1Y']-data['TIIE2Y1Y'])*100
data['4Y1Y vs 5Y2Y'] = (data['TIIE5Y2Y']-data['TIIE4Y1Y'])*100
data['3Y1Y vs 4Y1Y'] = (data['TIIE4Y1Y']-data['TIIE3Y1Y'])*100
data['3Y2Y vs 5Y2Y'] = (data['TIIE5Y2Y']-data['TIIE3Y2Y'])*100
data['3Y2Y vs 5Y5Y'] = (data['TIIE5Y5Y']-data['TIIE3Y2Y'])*100 
data['1Y1Y vs 2Y1Y'] = (data['TIIE2Y1Y']-data['TIIE1Y1Y'])*100
data['2Y1Y vs 3Y2Y'] = (data['TIIE3Y2Y']-data['TIIE2Y1Y'])*100 
data['2Y2Y vs 3Y2Y'] = (data['TIIE3Y2Y']-data['TIIE2Y2Y'])*100
#%% VIZ
tmplst = ['TIIE1Y','TIIE2Y','TIIE3Y','TIIE4Y','TIIE5Y','TIIE7Y','TIIE10Y']
spreads = ['T4s5s7s']
normLinplots(data, '2023-07-01', dt_end='', 
             lst_securities=spreads , plt_size=(15,9), plt_cmap='tab10')
# level chges stats
statistics(data.loc['2016':,tmplst].diff().dropna()*100)
###############################################################################
# Multivariate analysis
###############################################################################
# dates
dt_start, dt_end = '2018-12-01', '2023-12-31'
# correlation matrix
plot_corrmatrix(data.diff(), dt_start=dt_start, dt_end=dt_end, 
                lst_securities=tmplst, plt_size=(13,10), txtCorr=True, 
                corrM='kendall')
# boxplots
from scipy.stats.mstats import winsorize
plt.style.use('ggplot')
boxplot_rets(data.diff().apply(lambda x: winsorize(x,limits=[0.03,0.12])), 
             dt_start=dt_start, dt_end=dt_end, lst_securities=tmplst, 
             str_ttl='Daily Rate Changes')
#%%############################################################################
# MODELING ASSOCIATION
###############################################################################
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
sclr = StandardScaler()
rgr = LinearRegression

# vars
y_name = ['T2s5s10s']
X1_name = ['TIIE5Y'] 
X2_name = ['T2s10s']
X_name = X1_name+X2_name

# Date Seg
tfmt = '%Y-%m-%d'
str_end_test = pd.Timestamp.strftime(data.index[-1], tfmt)
str_start_test = pd.Timestamp.strftime(
    data.index[-1] - pd.DateOffset(years=1), tfmt)
str_start_train = pd.Timestamp.strftime(
    pd.Timestamp(str_start_test) - pd.DateOffset(years=8), tfmt)

data_train, data_test = sliceDataFrame(data, 
                                       dt_start=str_start_train, 
                                       dt_end=str_start_test, 
                                       lst_securities=y_name+X_name),\
    sliceDataFrame(data, 
                   dt_start=str_start_test, 
                   dt_end=str_end_test,
                  lst_securities=y_name+X_name)

y_train, X_train = data_train[y_name], data_train[X_name]
y_test, X_test = data_test[y_name], data_test[X_name]

# Models
m1 = rgr().fit(X_train[X1_name], y_train)
m2 = rgr().fit(X_train[X2_name], y_train)

# Model betas
m1_coefs = m1.coef_.tolist()[0]+[m1.intercept_[0]]
m2_coefs = m2.coef_.tolist()[0]+[m2.intercept_[0]]

# Preds
y_train_pred1 = pd.DataFrame(m1.predict(X_train[X1_name]), 
                             index = y_train.index, columns=['Model1'])
y_train_pred2 = pd.DataFrame(m2.predict(X_train[X2_name]), 
                             index = y_train.index, columns=['Model2'])
# Model miss
m_levels_train = pd.concat([y_train, y_train_pred1, y_train_pred2], axis=1)

e_blly = m_levels_train.apply(lambda x: x[y_name] - x['Model1'], axis=1)
e_wngs = m_levels_train.apply(lambda x: x[y_name] - x['Model2'], axis=1)
## Error stats
e_blly_mu, e_blly_std = e_blly.mean(), e_blly.std()
e_wngs_mu, e_wngs_std = e_wngs.mean(), e_wngs.std()
## Zscores
e_blly_z = (e_blly - e_blly_mu)/e_blly_std
e_wngs_z = (e_wngs - e_wngs_mu)/e_wngs_std

# Model test
y_test_pred1 = pd.DataFrame(m1.predict(X_test[X1_name]), 
                           index=y_test.index, columns=['Model1'])
y_test_pred2 = pd.DataFrame(m2.predict(X_test[X2_name]), 
                           index=y_test.index, columns=['Model2'])
m_levels_test = pd.concat([y_test, y_test_pred1, y_test_pred2], axis=1)

# Spread vs Belly Model
plt.figure(figsize=(8,5))
plt.xlabel(f'{X1_name[0]}'); plt.ylabel(f'{y_name[0]}')
t5y = plt.scatter(X_test[X1_name], y_test, s=80, edgecolors='C0',
            facecolors='none')
t3m = plt.scatter(X_test[X1_name].iloc[-64:], y_test.iloc[-64:], 
            c='darkcyan')
t0 = plt.scatter(X_test[X1_name].iloc[-1], y_test.iloc[-1], s=80,
            c='orange', edgecolors='black')
plt.plot(X_test[X1_name], y_test_pred1, color='black', linestyle='--', 
         alpha=0.80, linewidth=1)
plt.legend((t5y, t3m, t0), ('T1Y', 'T3M', 'Current'))
plt.tight_layout(); plt.show()

# Spread vs Slope Model
plt.figure(figsize=(8,5))
plt.xlabel(f'{X2_name[0]}'); plt.ylabel(f'{y_name[0]}')
t5y = plt.scatter(X_test[X2_name], y_test, s=80, edgecolors='C0',
            facecolors='none')
t3m = plt.scatter(X_test[X2_name].iloc[-64:], y_test.iloc[-64:], 
            c='darkcyan')
t0 = plt.scatter(X_test[X2_name].iloc[-1], y_test.iloc[-1], s=80,
            c='orange', edgecolors='black')
plt.plot(X_test[X2_name], y_test_pred2, color='black', linestyle='--', 
         alpha=0.80, linewidth=1)
plt.legend((t5y, t3m, t0), ('T1Y', 'T3M', 'Current'))
plt.tight_layout(); plt.show()

# Test miss
e1 = m_levels_test.apply(lambda x: x[y_name] - x['Model1'], axis=1)
e2 = m_levels_test.apply(lambda x: x[y_name] - x['Model2'], axis=1)

# Normalized errors
z1 = (e1-e_blly_mu)/e_blly_std
z2 = (e2-e_wngs_mu)/e_wngs_std
norm_e = pd.concat([z1,z2], axis=1); norm_e.columns=['Belly','Slope']
## NormedEPlot
clrlst = [f'C{n+2}' for n in range(norm_e.shape[1])]
plt.style.use('ggplot')
plt.figure(figsize=(8,6)); norm_e.iloc[-64:].plot(color=clrlst)
plt.title('Normalized Errors')
plt.axhline(y=2, c='darkcyan', linestyle='--', linewidth=0.8)
plt.axhline(y=-2, c='orange', linestyle='--', linewidth=0.8)
plt.ylim(-4,4)
plt.tight_layout(); plt.show()











