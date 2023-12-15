# -*- coding: utf-8 -*-
"""
Quantitative Strategies
Trading with Copulas

@author: jquintero
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
# best-known
import numpy as np
import pandas as pd
import datetime
# plotting
from matplotlib import pyplot as plt
from matplotlib import ticker as tkr
import matplotlib.dates as mdates
# statistics tests
from statsmodels.tsa.stattools import adfuller
# error-warning mgmt
import warnings
warnings.filterwarnings("ignore")
# copulas
import copulas.bivariate as biCop
# combinatonics
from itertools import combinations
# saving/exporting data/models
import pickle

###############################################################################
# Data
###############################################################################
# daily dataset import
str_path = r'C:\Users\jquintero\db'
str_file = r'\data_5m.xlsx'

# database buildup: import data, export parquet
def set_data_5m(str_path, str_file, str_shtname):
    """
    Database buildup from excel file when non-existent.
    """
    # columns
    tmpdf_head =  pd.read_excel(str_path+str_file,
                                sheet_name = str_shtname,
                          header=None, nrows = 2, skiprows=2)
    assetnames = tmpdf_head.iloc[0,:].dropna()
    repls = {' Index' : '', ' Curncy' : '', ' Index':'', ' Comdty':''}
    names = np.array(
        [s.replace(s[s.find(' '):], 
               repls.get(s[s.find(' '):])) 
             for s in assetnames]
        )
    # data
    tmpdf = pd.read_excel(str_path+str_file, sheet_name = str_shtname, 
                          header=None, index_col=0, engine='openpyxl')
    tmpdf = tmpdf.iloc[4:,:].dropna(axis=1)
    tmpdf.columns = tmpdf_head.iloc[1,:].dropna().drop(0)
    tmpdf = tmpdf.drop('Dates',axis=1)
    df_cols = []
    for name in names:
        df_cols = df_cols + [name+'_'+t for t in ['O', 'H', 'L', 'C']]
    tmpdf.columns = df_cols
    tmpdf.index.name = 'date'
    tmpdf.to_parquet(r'H:\db\data_5m.parquet', compression=None)
    return None

# database update
def update_data_5m(str_path, str_file, str_shtname, n_skipRows, str_dbpath):
    data = pd.read_parquet(str_dbpath)
    tmpdata = readParseData(str_path, str_file, str_shtname, n_skipRows)
    tmpdata.iloc[~tmpdata.index.isin(data.index),:]
    updatedata = data.append(tmpdata.iloc[~tmpdata.index.isin(data.index),:])
    return updatedata

# import data
def readParseData(str_path, str_file, str_shtname, n_skipRows):
    """
    Import, parse and clean data from excel database file.
    """
    xlwb = pd.ExcelFile(str_path+str_file)
    xlwb_data = xlwb.book.get_sheet_by_name(str_shtname)
    xlwb_rows = xlwb_data.max_row
    tmpdf_head =  pd.read_excel(str_path+str_file,
                                sheet_name = str_shtname,
                          header=None, nrows = 2, skiprows=2)
    assetnames = tmpdf_head.iloc[0,:].dropna()
    repls = {' Index' : '', ' Curncy' : '', ' Index':'', ' Comdty':''}
    names = np.array(
        [s.replace(s[s.find(' '):], 
               repls.get(s[s.find(' '):])) 
             for s in assetnames]
        )
    
    tmpdf = pd.read_excel(str_path+str_file,
                          sheet_name = str_shtname,
                          header=None, index_col=0,
                          engine='openpyxl', 
                          skiprows=int(xlwb_rows-n_skipRows)).dropna(axis=1)
    tmpdf.columns = tmpdf_head.iloc[1,:].dropna().drop(0)
    tmpdf = tmpdf.drop('Dates',axis=1)
    
    df_cols = []
    for name in names:
        df_cols = df_cols + [name+'_'+t for t in ['O', 'H', 'L', 'C']]
    tmpdf.columns = df_cols
    tmpdf.index.name = 'date'
    xlwb.close()
    return(tmpdf)

# import realtime data
def readParseHFData(str_path, str_file, str_shtname):
    """
    Import, parse and clean real-time data from excel database file
    """
    tmpdf_head =  pd.read_excel(str_path+str_file,
                                sheet_name = str_shtname,
                          header=None, nrows = 2, skiprows=2)
    assetnames = tmpdf_head.iloc[0,:].dropna()
    repls = {' Index' : '', ' Curncy' : '', ' Index':'', ' Comdty':''}
    names = np.array(
        [s.replace(s[s.find(' '):], 
               repls.get(s[s.find(' '):])) 
             for s in assetnames]
        )
    
    tmpdf = pd.read_excel(str_path+str_file,
                          sheet_name = str_shtname,
                          header=None, index_col=0,
                          engine='openpyxl',
                          skiprows=4).dropna(axis=1)
    tmpdf.columns = tmpdf_head.iloc[1,:].dropna().drop(0)
    tmpdf = tmpdf.drop('Dates',axis=1)
    
    df_cols = []
    for name in names:
        df_cols = df_cols + [name+'_'+t for t in ['O', 'H', 'L', 'C']]
    tmpdf.columns = df_cols
    tmpdf.index.name = 'date'

    return(tmpdf)

## read whole excel-db
# tmpdata = readParseData(str_path, str_file, 'data', 1e4)
data = pd.read_parquet(r'H:\db\data_5m.parquet')
# tmpdata2 = data.loc[:str(data.index[-1]-datetime.timedelta(minutes=5*1)),:].append(
#    tmpdata.loc[str(data.index[-1]):,]
#    )
# tmpdata2.to_parquet(r'H:\db\data_5m.parquet')


# # train-validation dates
today = datetime.date.today()
lastFriday = today - datetime.timedelta(days=today.weekday()) +\
                    datetime.timedelta(days=4, weeks=-1)
last2Friday = lastFriday - datetime.timedelta(days=7)
tEnd_valid = datetime.datetime(lastFriday.year, lastFriday.month, 
                               lastFriday.day, 16, 0, 0)
tStart_valid = datetime.datetime(last2Friday.year, last2Friday.month, 
                               last2Friday.day, 16, 0, 0)
tStart_train = tStart_valid - datetime.timedelta(days=27, hours=16)

# train-validation data split: train(28d)-valid(7d)-test(1d)
df_train = data.loc[str(tStart_train):str(tStart_valid),]
df_valid = data.loc[str(tStart_valid):str(tEnd_valid),]

# OC returns
def get_df_ret(df_px):
    """
    Open2Close Log-Returns
    """
    idx_cols = np.concatenate(
        (np.where(np.array([s.find('_O') for s in df_px.columns])
                            >=0 )[0], 
                   np.where(np.array([s.find('_C') for s in df_px.columns])
                            >=0 )[0])
                   )
    idx_cols.sort()
    df_logRet = df_px.iloc[:,idx_cols].\
        apply(np.log).diff(axis=1).dropna(axis=1)
    idx_cols_logret = np.where(
        np.array([s.find('_C') for s in df_logRet.columns])>=0 )[0]
    df_logRet.columns = [s.replace('_C','') for s in df_logRet.columns]
    return(df_logRet.iloc[:,idx_cols_logret])

# log-returns
df_train_r = get_df_ret(df_train)
df_valid_r = get_df_ret(df_valid)

# returns' ECDF
def get_df_ret_ecdf(df_ret):
    from statsmodels.distributions.empirical_distribution import ECDF
    df_ret_ecdf = df_ret.apply(ECDF)
    df_ret_ecdf.index = [s.replace('_C','') for s in df_ret.columns]
    return(df_ret_ecdf)

# train-set returns' ECDF
df_train_r_ecdf = get_df_ret_ecdf(df_train_r)

# trian-valid-sets uniform data
df_train_r_u = df_train_r.apply(lambda y: df_train_r_ecdf[y.name](y))
df_valid_r_u = df_valid_r.apply(lambda y: df_train_r_ecdf[y.name](y))
##############################################################################
# Best Copula Model
def get_bestBivarCop(tgt_y, df_train_r_u, df_valid_r_u):
    names_posspairs = df_train_r_u.columns.drop(tgt_y)
    n_posspairs = len(names_posspairs)
    df_tgtPairs = pd.DataFrame(names_posspairs, 
                               index=[tgt_y]*n_posspairs,
                               columns=['y2'])
    lst_tgtPairs = [[tgt_y,c] for c in df_tgtPairs['y2']]
    # possible best bivariate copulas
    tmpdic_copPairs = dict(
        [(p[1],biCop.select_copula(df_train_r_u[p].to_numpy())) 
         for p in lst_tgtPairs]
        )
    # possible mispricing indexes
    tmpdf_pairsM = pd.DataFrame(
            [
                v.partial_derivative(
                        df_valid_r_u[[tgt_y,k]].to_numpy()
                        ) - 0.5
                for k,v in tmpdic_copPairs.items()
             ]
        ).T.fillna(method = 'ffill').cumsum().\
        rename(columns=dict(
            [(n,name) 
             for n,name in zip(range(n_posspairs), 
                               names_posspairs)
            ]
        ))
    # best mispricing index
    tmp_adfstatsPairs = tmpdf_pairsM.apply(lambda y: adfuller(y)[0])
    # best paired mean-reverting-M
    bestpair = names_posspairs[tmp_adfstatsPairs.argmin()]
    # best bivariate copula update
    bestcopair = [tgt_y, bestpair]
    try:
        # update params
        tmpdic_copPairs[bestpair].\
            fit(df_train_r_u.append(df_valid_r_u)[bestcopair].to_numpy())
    except ValueError:
        # update copula model
        tmpdic_copPairs[bestpair] = \
            biCop.\
                select_copula(
                    df_train_r_u.append(df_valid_r_u)[bestcopair].to_numpy())
    # save updated model
    selCop_dic = tmpdic_copPairs[bestpair].to_dict()
    # save updated mispricing index
    selCop_valid_M = pd.DataFrame(
        tmpdic_copPairs[bestpair].partial_derivative(
            df_valid_r_u[bestcopair].to_numpy())-0.5,
        index = df_valid_r_u.index,
            ).cumsum().rename(columns={0:'M'})
    res = {'covar': bestpair, 'cop': selCop_dic, 'M': selCop_valid_M}
    return(res)
###############################################################################

##############################################################################
# Asset Copula Models , 52min
## Best Copula Model Fit For Possible Pairs
itime = datetime.datetime.now()
assetnames = df_train_r_u.columns.tolist()
models = {}
for name in assetnames:
    models[name] = get_bestBivarCop(name, df_train_r_u, df_valid_r_u)

models = dict((name, get_bestBivarCop(name, df_train_r_u, df_valid_r_u))
     for name in assetnames)
ftime = datetime.datetime.now()
elapsedtime = ftime-itime
print(f"\nElapsed time fitting models: {elapsedtime.seconds/60} minutes\n")
##############################################################################

# ecdf update
df_trva_r_ecdf = get_df_ret_ecdf(df_train_r.append(df_valid_r))
# export models
str_week_no = str(today.isocalendar().week)
str_modelname = r'models\copmodels_w'+str_week_no+r'.pickle'
with open(str_modelname, 'wb') as f:
    pickle.dump(models, f)
df_trva_r_ecdf.to_pickle(r'models\ecdfs.pkl')
# import models
models = pd.read_pickle(str_modelname)

# technical indicators module
import pandas_ta as ta

# model data specs
tgt_y = 'USDMXN'
cop = biCop.Bivariate.from_dict(models[tgt_y]['cop'])
y2 = models[tgt_y]['covar'] 

# test data
hfdata = readParseHFData(str_path, r'\fut_copmod.xlsx', 'test')

# test-set uniform data
df_test_r = get_df_ret(hfdata)
df_test_r_u = df_test_r.apply(lambda y: df_trva_r_ecdf[y.name](y))

# model's cond. cdf
mpxidx_test = pd.DataFrame(cop.partial_derivative(
    df_test_r_u[[tgt_y, y2]].to_numpy()),
    index = df_test_r_u.index, columns = ['h'])
mpxidx_test['m'] = mpxidx_test.h-0.5
mpxidx_test['M'] = mpxidx_test.m.cumsum()

df_M = models[tgt_y]['M'].append(mpxidx_test[['M']])
df_M.ta.rsi(close='M', length=9, suffix='M', append=True)
df_run = df_M.copy()
df_run = df_run.merge(hfdata[[tgt_y+'_C']], 
                      left_index = True, right_index = True)
# timehandle
dt_now_st = datetime.datetime(datetime.datetime.today().year, 
                           datetime.datetime.today().month,
                           datetime.datetime.today().day,
                           6,0,0)
dt_now = datetime.datetime(datetime.datetime.today().year, 
                           datetime.datetime.today().month,
                           datetime.datetime.today().day, 
                           datetime.datetime.today().hour, 
                           np.max(
                               ((int(datetime.datetime.today().minute/5)-1)*5,
                                0)
                               ), 0)
dt_now_0 = dt_now - datetime.timedelta(minutes=5*10)
# display
print(df_run.loc[str(dt_now_0):str(dt_now),])

ax = df_run.loc[str(dt_now_st):,'RSI_9_M'].plot(title=r'RSI($M_t$)')
ax.axhline(y=30, color='g', linestyle='--', lw=1.5, alpha=0.5)
ax.axhline(y=70, color='r', linestyle='--', lw=1.5, alpha=0.5)
ax.axhline(y=50, color='gray', linestyle='-', lw=1, alpha=0.4)
plt.tight_layout()
plt.show()


