"""
Objective: Quant model research for generating global macro trading/investment 
strategies in fixed income markets. Particularly, over US, MX and cross assets.
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
# best-known
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy import stats
from sklearn.preprocessing import StandardScaler, Normalizer
# plotting
from matplotlib import pyplot as plt
from matplotlib import ticker as tkr
import matplotlib.dates as mdates
# data pulling
import fredapi as fed
fred_api_key = '7d87ebd4394dba092e1df378caac8885'
fred = fed.Fred(api_key=fred_api_key)
# statistic models
#import statsmodels.api as sm
# statistics tests
from statsmodels.tsa.stattools import adfuller
# error-warning mgmt
import warnings
warnings.filterwarnings("ignore")
# object import-export
import pickle
# Display options
pd.set_option('display.float_format', lambda x: '%0.4f' % x)
pd.set_option('display.max_columns', 7)
###############################################################################
# UDF
###############################################################################
from udf_trading_eco_fi import *
# In[]
###############################################################################
# Securities Data
###############################################################################
# daily dataset import
str_path = r'H:\db'
str_file = r'\data_1D.csv'
data = csvImport(str_path, str_file) #data.columns[:63]
# data cleanup
# sc
tmp = (data.diff().fillna(0))/data.rolling(21*3).std().fillna(method='bfill')
for asset in data.columns[:63]:
    try:
        bool_idx = tmp[asset][tmp[asset]<-2].\
            append(tmp[asset][tmp[asset]>2]).\
        sort_index().to_frame().apply(np.sign).diff().shift(-1).\
            merge(tmp[asset][tmp[asset]<-2].\
                  append(tmp[asset][tmp[asset]>2]).sort_index().\
                      index.to_frame().diff().shift(-1).\
                          apply(lambda x: x[0].days,axis=1).to_frame(),
                          left_index=True,
                          right_index=True).\
                apply(lambda r: abs(r.iloc[0])==2 and r.iloc[1]==1,axis=1)
    
        selidx = bool_idx.index[bool_idx].tolist()
        if len(selidx)>0:
            data.loc[selidx,asset] = np.nan
            data[asset] = data[asset].interpolate()
    except:
        print(f'DataFrame for {asset}:\n')
        print(tmp[asset][tmp[asset]<-2].\
            append(tmp[asset][tmp[asset]>2]).\
        sort_index().to_frame())
# spreads
data['T5s7s10s'] = (2*data['TIIE7Y']-data['TIIE5Y']-data['TIIE10Y'])*100
data['T5s10s'] = (data['TIIE10Y']-data['TIIE5Y'])*100
data['T2s5s'] = (data['TIIE5Y']-data['TIIE2Y'])*100
data['T3s7s'] = (data['TIIE7Y']-data['TIIE3Y'])*100
data['T2s10s'] = (data['TIIE10Y']-data['TIIE2Y'])*100
data['T2s5s10s'] = (2*data['TIIE5Y']-data['TIIE2Y']-data['TIIE10Y'])*100
data['MXUS2Y'] = (data['TIIE2Y']-data['USGG2'])*100
data['MXUS5Y'] = (data['TIIE5Y']-data['USGG5'])*100
data['MXUS10Y'] = (data['TIIE10Y']-data['USGG10'])*100
data['MXEU10Y'] = (data['TIIE10Y']-data['EUSW10'])*100
# In[]
###############################################################################
# Macro Data
###############################################################################
"""
TOP 10 Economic indicators (AAII Journal)
i) GDP; ii) M2; iii) CPI; iv) PPI; v) ConsumerConfidence
vi) Employment; vii) RetailSales; viii) HousingStarts; ix) Manuf+Inv; x) EqMkt
"""
# data files path
str_ecopath = r'H:\Python\algot_eco_fi'
# files name
str_retail = r'\RSXFSN_retail.csv'
str_jobsop = r'\JTUJOL.csv'
str_ip = r'\IPMANSICN.csv'
str_buss = r'\TOTBUSSMNSA.csv'
str_m2 = r'\M2NS.csv'
str_ppi = r'\PPIACO.csv'
str_loans = r'\H8B1023NCBCMG_commindloans.csv'
# data pull
lst_ecoind = [str_retail,str_jobsop,str_ip,str_buss,str_m2,str_ppi,str_loans]
lst_econames = ['ret_sales','jobs','ind_prod','bus_sales','m2','ppi','loans']
df_eco = pd.DataFrame()
for indicator in lst_ecoind:
    filepath = str_ecopath + indicator
    tmpdf = pd.read_csv(str_ecopath+indicator,index_col=0, parse_dates=True)
    df_eco = pd.concat([df_eco, tmpdf], axis=1)

df_eco = df_eco.rename(columns=dict(zip(df_eco.columns.tolist(),lst_econames)))
df_eco_yoy = pd.concat(
    [
     (100*(df_eco.dropna()[lst_econames[:-1]]\
           /df_eco.dropna()[lst_econames[:-1]].shift(12)-1)).dropna(),
         df_eco['loans']],axis=1).dropna()
# In[]
# data pull via API
fred_retail = fred.get_series('RSXFSN')
fred_jobs = fred.get_series('JTUJOL')
fred_ip = fred.get_series('IPB50001N')
fred_nwods = fred.get_series('UMTMNO')
fred_nwodsdi = fred.get_series('NOCDINA066MNFRBNY')
fred_buss = fred.get_series('TOTBUSSMNSA')
fred_m2 = fred.get_series('M2NS')
fred_ppi = fred.get_series('PPIACO')
fred_loans = fred.get_series('H8B1023NCBCMG')
fred_nfp = fred.get_series('PAYNSA')
fred_nfpsa = fred.get_series('PAYEMS')
fred_avgearn = fred.get_series('CEU0500000003')
fred_avgearn_leis = fred.get_series('CEU7000000003')
fred_avgearn_retail = fred.get_series('CEU4200000003')
fred_avgearn_serv = fred.get_series('CEU6000000003')
fred_avgearn_transp = fred.get_series('CEU4300000003')
fred_cpi = fred.get_series('CPIAUCNS')
# list group
lst_eco = [fred_m2, fred_ip, fred_nwods, fred_buss, fred_loans, fred_retail, 
           fred_jobs, fred_nfp, fred_avgearn_serv, fred_ppi, fred_cpi]
lst_eco_name = ['m2','ip', 'nwods','buss','loans','retail',
                'jobops','nfp','awe_serv','ppi','cpi']
for serie,name in zip(lst_eco,lst_eco_name): serie.name=name
# merge
df_eco = pd.concat(lst_eco, axis=1)
df_eco_yoy = ((df_eco.iloc[:,np.where(df_eco.columns != 'loans')[0]]/\
               df_eco.iloc[:,np.where(df_eco.columns != 'loans')[0]].\
                   shift(12)-1)*100).merge(df_eco.loans, how='left',
                                           left_index=True,right_index=True)
# series corr viz
plot_corrmatrix(
    pd.DataFrame(
        Normalizer().fit_transform(
            StandardScaler().fit_transform(
                df_eco_yoy.dropna())), columns=df_eco_yoy.columns))
# In[Case on PPI leading CPI]
df_avgearn = pd.concat([fred_avgearn,fred_avgearn_leis,fred_avgearn_retail,
                        fred_avgearn_serv,fred_avgearn_transp],
                       axis=1).rename(columns={0:'avgearn',
                                               1:'avgearn_leis',
                                               2:'avgearn_retail',
                                               3:'avgearn_serv',
                                               4:'avgearn_transp'})
(100*(df_avgearn/df_avgearn.shift(12)-1)).rolling(3).mean().\
    loc['2018':,['avgearn','avgearn_serv']].plot.scatter('avgearn_serv','avgearn')
(df_avgearn/df_avgearn.loc['2018-01-01']*100).loc['2018':].plot()
# In[Case on Diffusion Idx vs Level Idx]
runcase = False
if runcase:
    fred_nwods = fred.get_series('UMTMNO')
    fred_nwodsdi = fred.get_series('NOCDINA066MNFRBNY')
    fred_nwods.name = 'new_ords'
    fred_nwodsdi.name = 'new_ords_didx'
    
    df_ords = pd.merge(fred_nwods, fred_nwodsdi, how='right', 
                       left_index=True, right_index=True)
    df_ords_nz = pd.DataFrame(Normalizer().fit_transform(StandardScaler().\
            fit_transform(df_ords.diff().dropna())),
                              columns=df_ords.columns,
                              index=df_ords.diff().dropna().index)
    df_ords_nz.rolling(12).mean().plot()
    df_ords_nz['new_ords'] = df_ords_nz['new_ords'].shift(-9)
    df_ords_nz.rolling(12).mean().loc['2015':].corr('spearman')
    df_ords_nz.rolling(12).mean().loc['2015':].plot()
    
    df_ords_d1 = df_ords.copy()
    df_ords_d1['new_ords_yoy'] = df_ords_d1['new_ords']/\
        df_ords_d1['new_ords'].shift(12)-1
    df_ords_d1['new_ords_didx_l3'] = df_ords_d1['new_ords_didx'].shift(3)
    df_ords_d1['new_ords_didx_l6'] = df_ords_d1['new_ords_didx'].shift(6)
    df_ords_d1['new_ords_didx_l9'] = df_ords_d1['new_ords_didx'].shift(9)
    X = df_ords_d1.dropna().\
        loc['2015':][['new_ords_didx', 'new_ords_didx_l3', 
                      'new_ords_didx_l6', 'new_ords_didx_l9']]
    y = df_ords_d1.dropna().loc['2015':]['new_ords_yoy']
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X.to_numpy(),y.to_numpy().reshape(-1,1))
    y_pred = pd.DataFrame(lr.predict(X.to_numpy()),
                          index=y.index,columns=['Lr'])
    lr_res = pd.concat([y_pred,y],axis=1)
    lr_res.plot()
    X_test = df_ords_d1.\
        loc['2023-04':,['new_ords_didx', 'new_ords_didx_l3', 
                        'new_ords_didx_l6', 'new_ords_didx_l9']]
    y_new = pd.DataFrame(lr.predict(X_test.to_numpy()),
                         index=X_test.index,columns=['Model'])
# In[]
###############################################################################
# Data Merge
###############################################################################
data_M = data.resample('M').last()
data_M.index = [dt.replace(day=1) for dt in data.resample('M').last().index]
df_0 = pd.concat([df_eco_yoy, data_M], axis=1).loc[data_M.index[0]:,:]
df_1 = pd.concat([df_eco_yoy.rolling(3).mean(), 
                  data_M], axis=1).loc[data_M.index[0]:,:]
df_d1 = pd.concat([df_eco_yoy.rolling(3).mean().diff(), 
                  data_M.diff()], axis=1).loc[data_M.index[0]:,:]
# In[]
###############################################################################
# Visualization
###############################################################################
# spreads
normLinplots(data.loc['2019':'2023',['T2s5s','T5s10s','T3s7s','T2s10s']], plt_cmap='copper')
# asw
normLinplots(data.loc['2019':'2023',['MXUS2Y','MXUS5Y','MXUS10Y','MXEU10Y']], plt_cmap='coolwarm')
# selected securities
tmplst = ['TIIE1Y1Y', 'TIIE2Y5Y','TIIE10Y', 'USSW2', 'USSW5', 'USSW10']
normLinplots(data, '2021-12-31', dt_end='', lst_securities=tmplst , plt_cmap='viridis')
# eco indicators
normLinplots(df_eco_yoy.loc['2010':,['ret_sales','ppi','jobs','bus_sales']].dropna(axis=1))
###############################################################################
# Stats
###############################################################################
# eco data pct chgs
statistics(df_eco_yoy.loc['2009':])
###############################################################################
# Multivariate analysis
###############################################################################
# eco data correlation matrix
plot_corrmatrix(df_eco_yoy.loc['2009':])
# eco data boxplots
boxplot_rets(df_eco_yoy.loc['2009':], str_ttl='Macro Data (1M)')
# pairwise returns scatterplots
scatterplot(df_eco_yoy)
###############################################################################
# Clustering analysis
###############################################################################
# timeframe dates
n_yr_train = 8
tmpdtf = '2023-03-01'
tmpdti = (pd.to_datetime(
    np.datetime64(tmpdtf) - np.timedelta64(365*n_yr_train,'D')).replace(day=1)
    ).strftime('%Y-%m-%d')
# eco data clustering assessment
preclustering(df_0, dt_start = tmpdti, dt_end = tmpdtf, 
              lst_securities=lst_eco_name,  plt_size = (15,5), str_mode='sclr')
# k-means clustering of eco data
dic_cltr = cluster_kmeans(df_0, dt_start = tmpdti, dt_end = tmpdtf, 
                          lst_securities=lst_eco_name,
                          n_clusters=3, iclrmap=True, str_mode='sclr')
dic_cltr['cluster_set']

###############################################################################
# PCA
###############################################################################
# eco data PCA
pca_assess(df_0, lst_securities=lst_eco_name, str_mode='', 
           dt_start = tmpdti, dt_end = tmpdtf)
# PCA
tmplst_byClust = [item for sublist in dic_cltr['cluster_set'].values() 
                  for item in sublist]
pca_fit, loadings_df, loadings, pca_scores = pca_(df_0, n_comps=3, 
                                                  dt_start = tmpdti, 
                                                  dt_end = tmpdtf, 
                                                  lst_securities = tmplst_byClust, 
                                                  plt_pcmap=True,
                                                  str_mode='')
# first 3 PC amongst prev clusters
#tmpclrmap = {1:'red', 2:'blue', 3:'cyan', 4:'orange', 5:'purple', 
#                           6:'green', 7:'gray', 8:'magenta', 9:'brown', 
#                           10:'yellow'}
tmpclrmap = {1:'red', 2:'blue', 3:'cyan'}#, 4:'orange',
             #5:'purple', 6:'green'} 
             #7:'gray'}
plt_pcal_cltr(pca_fit, loadings_df, loadings, dic_cltr, d_cltr_cmap=tmpclrmap)
# first 3 PC biplots
plt_pca_biplots(pca_scores, loadings_df, loadings, dic_cltr)
# In[Factor Model]
###############################################################################
# MODELING ASSOCIATION
###############################################################################
"""
Problem: Are macro data signals profitable within mexican FI instruments.
Macro data signals are going to be defined by the errors coming between
a simple model and the observed value of a given target. Testing for z-score 
threshold over the model's error whould trigger a trade signal for the given
target variable (TIIE 1y1y, for instance). Successful models should provide 
timely signals that translate to consistent and stable profitable trade ideas.

The problem will be solved by PCR model fitting over an 5-year trainning period.
Over a 1-month period, daily, the model is tested for trading signals. Every
time the errors exceed 2-sigma levels a long/short position is accounted and 
closed whenever convergence is reached, when the errors return to zero from
below/above.
"""
sclr = StandardScaler()
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Proc to generate PCR model testing runs
# hyper params
n_rsi_l = 5
# res vars
df_pcr_coefs = pd.DataFrame(columns=['PC1','PC2','PC3','b'])
df_pcr_test_perf = pd.DataFrame(columns=['Exp.Var','Max Err','MAE',
                                         'MSE','RMSE','R2Score','MAPE'])
dic_pcr_run = {}
# response variable
y_name = 'USGG2'
# Dates delimiters for train-test data split
df_dates_train_test = get_train_test_split_dates(data_M,8,3,3)
tmplst = tmplst_byClust + [y_name]
# Loop through study periods
for i,r in df_dates_train_test.iterrows(): # i,r=list(df_dates_train_test.iterrows())[0]
    #i = 1
    str_start_train, str_end_train, str_start_test, str_end_test = r
    # data split
    data_train = df_1.loc[str_start_train:str_end_train,tmplst]
    data_test = df_1.loc[str_start_test:str_end_test,tmplst]

    # data lags
    y_l1 = pd.concat([data_train,data_test])[y_name].shift().rename(y_name+'_l1').diff()
    y_l2 = pd.concat([data_train,data_test])[y_name].shift(2).rename(y_name+'_l2')
    y_l3 = pd.concat([data_train,data_test])[y_name].shift(3).rename(y_name+'_l3')
    y_ma_f = pd.concat([data_train,data_test])[y_name].rolling(3).mean().rename('SMA3')
    y_ma_s = pd.concat([data_train,data_test])[y_name].rolling(6).mean().rename('SMA6')
    y_mad = (y_ma_f - y_ma_s).rename('MAD').rolling(3).mean()
    y_techs = pd.concat([y_l1,y_mad],axis=1)
    #
    data_train = data_train.merge(y_techs, how='left', left_index=True, right_index=True).dropna()
    data_test = data_test.merge(y_techs, how='left', left_index=True, right_index=True)
    # explanatory lags
    x_l3 = pd.concat([data_train,data_test])[['nwods','buss','ppi','awe_serv']].\
        shift(3).rename(columns=\
            dict(zip(['nwods','buss','ppi','awe_serv'],
                     [s+'_l3' for s in ['nwods','buss','ppi','awe_serv']]))
            )
    x_l6 = pd.concat([data_train,data_test])[['nwods','buss','ppi','awe_serv']].\
        shift(6).rename(columns=\
            dict(zip(['nwods','buss','ppi','awe_serv'],
                     [s+'_l6' for s in ['nwods','buss','ppi','awe_serv']]))
            )
    x_lags = x_l3.merge(x_l6,how='left',left_index=True, right_index=True)
    data_train = data_train.merge(x_lags, how='left', left_index=True, right_index=True).dropna()
    data_test = data_test.merge(x_lags, how='left', left_index=True, right_index=True)
    
    # explanatory split
    X_train, X_test = data_train.drop(y_name, axis=1),\
        data_test.drop(y_name, axis=1)
    # response split
    y_train, y_test = data_train[y_name], data_test[y_name]
    # feature normalization
    sclr = StandardScaler()
    sclr.fit(X_train)
    nrm = Normalizer()
    nrm.fit(sclr.transform(X_train))
    # feature dim redux
    n_comps = 8
    pca = PCA(n_components = n_comps)
    # Model train
    X_train_T = nrm.transform(sclr.transform(X_train))
    pca = pca.fit(X_train_T)
    pca_scores_train = pca.transform(X_train_T) 
    #pca_scores_train = pca.fit_transform(X_train)
    rgr = LinearRegression()
    pcr = rgr.fit(pca_scores_train, y_train) # y-var vs the PCA scores linear regression
    # save coeffs
    df_pcr_coefs.loc[i] = pcr.coef_.tolist()+[pcr.intercept_]
    # Trainning perf metrics
    y_train_pred = pd.Series(pcr.predict(pca_scores_train), 
                             index = y_train.index)
    y_train_pred.rename('Model')
    train_perf_metrics = rgr_perf_metrics(y_train, y_train_pred, 
                                          'Train')
    # Trainning errors
    pcr_err = pd.DataFrame(pd.concat([y_train, y_train_pred.rename('Model')], 
                                     axis=1)).diff(axis=1).dropna(axis=1)*-1
    pcr_err_std = pcr_err.std()
    pcr_err_mu = pcr_err.mean()
    pcr_err_z = (pcr_err - pcr_err_mu)/pcr_err_std
    ## ADF test
    pcr_err_z_adft = adfuller(pcr_err_z)
    
    # Model test
    pca_scores_test = pca.transform(nrm.transform(sclr.transform(X_test)))
    #pca_scores_test = pca.transform(X_test)
    y_test_pred = pd.Series(pcr.predict(pca_scores_test), index=y_test.index)
    # Test performance metrics
    test_perf_metrics = rgr_perf_metrics(y_test, y_test_pred, 'Test')
    # Run performance met
    df_pcr_test_perf.loc[i] = test_perf_metrics.to_numpy().reshape(-1,)
    # Testing errors
    model_err = pd.DataFrame(
        pd.concat([y_test, y_test_pred.rename('Model')], axis=1)
        ).diff(axis=1).dropna(axis=1)*-1
    model_err_z  = (model_err - pcr_err_mu)/pcr_err_std
    
    ## Errors transform
    model_err_rsi = pcr_err_z.append(model_err_z).\
        rename(columns={'Model':'close'}).ta.rsi(length=n_rsi_l).\
            loc[model_err_z.index].to_frame()
    model_err_rsi = model_err_rsi[~model_err_rsi.index.duplicated(keep='first')]
    ## Testing model signals
    tmp_merrs = model_err_rsi.shift().merge(model_err_rsi,how='left',
                                          left_index=True, 
                                          right_index=True).fillna(method='bfill')
    model_err_signals_ub = tmp_merrs.apply(lambda y: 
                                           y[0]>70 and (y[1]<70 and y[1]>50),
                                           axis=1)
    model_err_signals_lb = tmp_merrs.apply(lambda y: 
                                           y[0]<30 and (y[1]>30 and y[1]<50),
                                           axis=1)
    ## Testing run
    model_err_rsi['Signal'] = 0
    model_err_rsi.loc[model_err_signals_ub[model_err_signals_ub].index,
                      'Signal'] = -1
    model_err_rsi.loc[model_err_signals_lb[model_err_signals_lb].index,
                      'Signal'] = 1
    model_err_test_run = model_err_rsi.merge(y_test,
                                             left_index=True,right_index=True)
    test_run_sl = np.max([np.round(pcr_err_std/2,2).values[0],
                         np.round(y_train.diff().dropna().std(),2)])
    model_err_test_run['SL'] = model_err_test_run['Signal']*test_run_sl*-1 +\
        model_err_test_run[y_name]
    dic_pcr_run[i] = model_err_test_run
###############################################################################
# MODEL EXPORT
features = ['TIIE1Y', 'TIIE2Y', 'TIIE3Y', 'TIIE4Y', 'TIIE5Y', 'TIIE7Y', 'TIIE10Y',
       'USSW1', 'USSW2', 'USSW3', 'USSW5', 'USSW7', 'USSW10', 'USGG2', 'USGG3',
       'USGG5', 'USGG7', 'USGG10', 'USGG30', 'EUSW2', 'EUSW3', 'EUSW5',
       'EUSW10', 'EUSW15', 'EUSW20', 'EUSW30', 'MXCDS', 'MXIVOL1M', 'TIIE1Y1Y',
       'TIIE1Y2Y', 'TIIE2Y1Y', 'TIIE1Y3Y', 'TIIE2Y2Y', 'TIIE3Y1Y', 'TIIE1Y4Y',
       'TIIE2Y3Y', 'TIIE3Y2Y', 'TIIE4Y1Y', 'TIIE3Y4Y', 'TIIE4Y3Y', 'TIIE5Y2Y',
       'TIIE3Y7Y', 'TIIE5Y5Y', 'TIIE7Y3Y']
model_pcr = {
    'n_train_yr': 2,
    'n_test_month': 1,
    'n_roll_month': 1,
    'response': y_name,
    'features': features,
    'feat_transform': sclr,
    'feat_redux': pca,
    'model':pcr,
    'model_err_rsi': 5,
    'hist_model_studyperiods': df_dates_train_test,
    'hist_model_coeff': df_pcr_coefs,
    'hist_model_testruns': dic_pcr_run}
## save model
str_model_path = 'H:\Python\hyperdrive_models'+rf'\pcr_{y_name}.pickle'
with open(str_model_path, 'wb') as handle:
    pickle.dump(model_pcr, handle, protocol=pickle.HIGHEST_PROTOCOL)
## load model
#with open(str_model_path, 'rb') as handle:
#    model_dic = pickle.load(handle)
###############################################################################
# In[USE MODEL]
def run_model_pcr(y_name):
    #y_name = 'TIIE2Y5Y' #y_name = 'TIIE1Y1Y'
    # model path
    str_model_path = 'H:\Python\hyperdrive_models'+rf'\pcr_{y_name}.pickle'
    # file verif
    modelExists = os.path.isfile(str_model_path) 
    if not modelExists:
        print(f'\nCould not find model-file: pcr_{y_name}.pickle')
        return None
    # load
    with open(str_model_path, 'rb') as handle:
        model_dic = pickle.load(handle)
    # study periods
    m_h_studyperiods = model_dic['hist_model_studyperiods']
    n_last_studyperiod = m_h_studyperiods.index[-1]
    # last data split timestamps
    train_st, train_ed, test_st, test_ed =\
        m_h_studyperiods.loc[n_last_studyperiod]
    # last data date
    data_last_date = data.index[-1]
    # new model run verif
    isNewModRun = np.datetime64(data_last_date)>np.datetime64(test_ed)
    ###########################################################################
    # NEW MODEL RUN
    if isNewModRun:
        n_new = n_last_studyperiod+1
        # new data split
        new_train_st, new_train_ed, new_test_st, new_test_ed = \
            dt_delims_train_test(np.datetime64(test_ed) +\
                                 np.timedelta64(1,'D'),2,1)
        # data split
        new_data_train, new_data_test = sliceDataFrame(data, 
                                               dt_start=new_train_st, 
                                               dt_end=new_train_ed),\
            sliceDataFrame(data, 
                           dt_start=new_test_st, 
                           dt_end=new_test_ed)
        # explanatory-response variables
        X_train, X_test = new_data_train[model_dic['features']], \
            new_data_test[model_dic['features']]
        y_train, y_test = new_data_train[model_dic['response']], \
            new_data_test[model_dic['response']]
        # data transformers
        m_feat_transform = model_dic['feat_transform'].fit(X_train)
        m_feat_redux = model_dic['feat_redux'].\
            fit(m_feat_transform.transform(X_train))
        # model
        X_train_pca_scores = m_feat_redux.\
            transform(m_feat_transform.transform(X_train)) 
        m_new = model_dic['model'].fit(X_train_pca_scores, y_train)
        # model pred
        ## train
        y_train_pred = m_new.predict(X_train_pca_scores)
        ## test
        X_test_pca_scores = m_feat_redux.\
            transform(m_feat_transform.transform(X_test))
        y_test_pred = m_new.predict(X_test_pca_scores)
        ## errors
        m_new_err = y_train - y_train_pred
        m_new_err_mu = m_new_err.mean()
        m_new_err_std = m_new_err.std()
        m_new_err_z = (m_new_err - m_new_err_mu)/m_new_err_std
        m_new_err_test = y_test-y_test_pred
        m_new_err=m_new_err.append(m_new_err_test)
        m_new_err=m_new_err[~m_new_err.index.duplicated(keep='first')]
        ## rsi
        m_new_err_z = (m_new_err-m_new_err_mu)/m_new_err_std
        m_new_rsi = m_new_err_z.rename('close').to_frame().ta.\
            rsi(length=model_dic['model_err_rsi']).to_frame()#.\
                #merge(data[model_dic['response']],
                #      how='left',left_index=True,
                #      right_index=True)
        ## plot
        m_new_rsi[f'RSI_{model_dic["model_err_rsi"]}'].\
            rename(f'Model({model_dic["response"]})').iloc[-22:].\
                plot(color='C0',xlabel='');plt.\
                axhline(y=70,color='g',alpha=0.5,linestyle='--');plt.\
                    axhline(y=30,color='r',alpha=0.5,linestyle='--');plt.\
                        legend();plt.\
                    tight_layout();plt.show()
        ## sl
        test_run_sl = np.max([np.round(m_new_err.std()/2,2),
                             np.round(y_train.diff().dropna().std(),2)])
        ## signals
        tmp_rsi = m_new_rsi.shift().merge(m_new_rsi,how='left',
                                              left_index=True, 
                                              right_index=True).\
                                                fillna(method='bfill')
        tmp_rsi_signals_ub = tmp_rsi.apply(lambda y: 
                                               y[0]>70 and \
                                                   (y[1]<70 and y[1]>50),
                                               axis=1)
        tmp_rsi_signals_lb = tmp_rsi.apply(lambda y: 
                                               y[0]<30 and \
                                                   (y[1]>30 and y[1]<50),
                                               axis=1)
        m_new_rsi['Signal'] = 0
        m_new_rsi.loc[tmp_rsi_signals_ub[tmp_rsi_signals_ub].index,
                          'Signal'] = -1
        m_new_rsi.loc[tmp_rsi_signals_lb[tmp_rsi_signals_lb].index,
                          'Signal'] = 1
        m_curr_err_run = m_new_rsi.merge(y_test, 
                                         left_index=True, right_index=True)
        m_curr_err_run['SL'] = m_curr_err_run['Signal']*test_run_sl*-1 +\
            m_curr_err_run[y_name]
        m_curr_err_run = \
            m_curr_err_run[~m_curr_err_run.index.duplicated(keep='first')]
        # add run to dictionary
        model_dic['hist_model_testruns'][n_new] = m_curr_err_run
        model_dic['feat_transform'] = m_feat_transform
        model_dic['feat_redux'] = m_feat_redux
        model_dic['model'] = m_new
        model_dic['hist_model_studyperiods'] = m_h_studyperiods.append(
            pd.Series(
                [new_train_st, np.datetime_as_string(new_train_ed), 
                 new_test_st, new_test_ed], 
                name=n_new, 
                index=m_h_studyperiods.columns))
        model_dic['hist_model_coeff'] = model_dic['hist_model_coeff'].append(
            pd.Series(np.insert(m_new.coef_,len(m_new.coef_),m_new.intercept_), 
                      name=n_new, index=model_dic['hist_model_coeff'].columns))
        # save new run
        with open(str_model_path, 'wb') as handle:
            pickle.dump(model_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # update testruns
    m_curr_run_day = model_dic['hist_model_testruns'][n_last_studyperiod]
    m_last_busdayofmonth = np.busday_offset(
        last_day_of_month(m_curr_run_day.index[-1]),-1, roll='forward')
    # new observation verif
    isCurrRunFin = \
        np.datetime64(m_curr_run_day.index[-1]) >= m_last_busdayofmonth
    ###########################################################################
    # UPDATE MODEL RUN
    if not isCurrRunFin:
        # train data
        data_train = data.loc[train_st:train_ed,model_dic['features']]
        y_train = data.loc[train_st:train_ed,model_dic['response']]
        # test data
        data_test = data.loc[test_st:test_ed,model_dic['features']]
        y_test = data.loc[test_st:test_ed,model_dic['response']]
        # data transform
        m_curr_feat_transform = model_dic['feat_transform']
        m_curr_feat_redux = model_dic['feat_redux']
        X_train = m_curr_feat_redux.\
            transform(m_curr_feat_transform.transform(data_train))
        X_test = m_curr_feat_redux.\
            transform(m_curr_feat_transform.transform(data_test))
        # model
        m_curr = model_dic['model']
        # model pred
        y_pred_train = m_curr.predict(X_train)
        y_pred_test = m_curr.predict(X_test)
        # model err
        m_curr_err_train = y_train - y_pred_train
        m_curr_err_test = y_test - y_pred_test
        m_curr_err = m_curr_err_train.append(m_curr_err_test)
        m_curr_err_z = \
            (m_curr_err - m_curr_err_train.mean())/m_curr_err_train.std()
        m_curr_err_rsi = m_curr_err_z.to_frame().\
            rename(columns={y_name:'close'}).\
                ta.rsi(length=model_dic['model_err_rsi']).to_frame()
        ## plot
        plt.style.use('seaborn')
        m_curr_err_rsi[f'RSI_{model_dic["model_err_rsi"]}'].\
            rename(f'Model({model_dic["response"]})').iloc[-22:].\
                plot(color='C0',xlabel='');plt.\
                axhline(y=70,color='g',alpha=0.5,linestyle='--');plt.\
                    axhline(y=30,color='r',alpha=0.5,linestyle='--');plt.\
                        legend();plt.\
                    tight_layout();plt.show()
        # model signals
        tmp_rsi = m_curr_err_rsi.shift().merge(m_curr_err_rsi,how='left',
                                              left_index=True, 
                                              right_index=True).\
                                                fillna(method='bfill')
        tmp_rsi_signals_ub = tmp_rsi.apply(lambda y: 
                                               y[0]>70 and \
                                                   (y[1]<70 and y[1]>50),
                                               axis=1)
        tmp_rsi_signals_lb = tmp_rsi.apply(lambda y: 
                                               y[0]<30 and \
                                                   (y[1]>30 and y[1]<50),
                                               axis=1)
        m_curr_err_rsi['Signal'] = 0
        m_curr_err_rsi.loc[tmp_rsi_signals_ub[tmp_rsi_signals_ub].index,
                          'Signal'] = -1
        m_curr_err_rsi.loc[tmp_rsi_signals_lb[tmp_rsi_signals_lb].index,
                          'Signal'] = 1
        m_curr_err_run = m_curr_err_rsi.merge(y_test,
                                                 left_index=True,
                                                 right_index=True)
        test_run_sl = np.max([np.round(m_curr_err_train.std()/2,2),
                             np.round(y_train.diff().dropna().std(),2)])
        m_curr_err_run['SL'] = m_curr_err_run['Signal']*test_run_sl*-1 +\
            m_curr_err_run[y_name]
        m_curr_err_run = \
            m_curr_err_run[~m_curr_err_run.index.duplicated(keep='first')]
        # update dictionary
        model_dic['hist_model_testruns'][n_last_studyperiod] = m_curr_err_run
        ## save update
        with open(str_model_path, 'wb') as handle:
            pickle.dump(model_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return m_curr_err_run

# In[Model]
# Run Model
y_name = 'T5s10s'
run_model_pcr(y_name)
###############################################################################
# Loadings visualization given new observations
# Model Factors Attribution
str_model_path = 'H:\Python\hyperdrive_models'+rf'\pcr_{y_name}.pickle'
with open(str_model_path, 'rb') as handle:
    model_dic = pickle.load(handle)

m_curr_feat_redux = model_dic['feat_redux']
m_curr_feat_transform = model_dic['feat_transform']
m_curr_feats = model_dic['features']
n_last_studyperiod = model_dic['hist_model_studyperiods'].index[-1]
train_st, train_ed, test_st, test_ed =\
    model_dic['hist_model_studyperiods'].loc[n_last_studyperiod]
m_curr = model_dic['model']
data_test = data.loc[test_st:test_ed,m_curr_feats]
m_curr_factors = m_curr_feat_redux.\
                transform(m_curr_feat_transform.\
                          transform(data_test))[-1,]
m_curr_coef_attb = m_curr_factors*m_curr.coef_ 
for i in range(3):
    (m_curr.coef_[0]*(pd.DataFrame(m_curr_feat_redux.components_.T,
                  columns=[f'PC{n}' for n in range(1,4)],
                  index=m_curr_feats)[f'PC{i+1}']\
         *pd.DataFrame(m_curr_feat_transform.transform(data_test),
                          columns=m_curr_feats).iloc[-1,:].T)).\
        plot.barh(title=f'Factor {i+1} ({m_curr_coef_attb[i]:.2f})', 
                  figsize=(10,9)).invert_yaxis();plt.\
            show()
# Current Model Run
df_m_run = pd.concat([data.loc[test_st:test_ed,model_dic['response']], 
           pd.DataFrame(m_curr.predict(m_curr_feat_redux.\
                transform(m_curr_feat_transform.\
                          transform(data_test))),
                        index=data.loc[test_st:test_ed,
                                       model_dic['response']].index,
                        columns=['Model'])], axis=1)
# Viusalize current model run
rgr_perf_metrics(df_m_run[y_name], df_m_run['Model'], 'PCA Rgr Test Model')
df_m_run.plot(style=['-','--'],color=['C0','orange']);plt.tight_layout();plt.show()
###############################################################################
# In[PCR Test]
# Proc to analyse PCR model testing runs
trading_res_cols = ['signal','duration','pnl','R']
df_pcr_test_trading_res = pd.DataFrame(columns=trading_res_cols)
# pcr runs concat
df_pcr_bt = pd.concat(dic_pcr_run.values())
v = df_pcr_bt[~df_pcr_bt.index.duplicated(keep='first')]

xb_50 = (v['RSI_'+str(n_rsi_l)].shift()>50).\
    to_frame().merge((v['RSI_'+str(n_rsi_l)]<50),
                     left_index=True,
                     right_index=True).all(axis=1)
xa_50 = (v['RSI_'+str(n_rsi_l)].shift()<50).\
    to_frame().merge((v['RSI_'+str(n_rsi_l)]>50),
                     left_index=True,
                     right_index=True).all(axis=1)
    
for q in v.index[v['Signal'] < 0]: # receiver signals
    rsilev,sgnl,entry,sl = v.loc[q,:]
    try:
        q_end = xb_50[q:].index[xb_50[q:]][0]
    except: # Close at last month day if no closing signal
        q_end = xb_50[q:].index[-1]
    tmpsgnlrun = v.loc[q:q_end,:]
    stopped = (tmpsgnlrun.loc[:,y_name] > sl).any()
    if not stopped:
        exit_level = v.loc[q_end,y_name]
    else:
        exit_level = sl
    pnl = (exit_level - entry)*sgnl
    duration = (q_end-q).days
    r_unit = pnl/abs((sl-entry))
    df_pcr_test_trading_res.loc[q] = sgnl,duration,pnl,r_unit
    
for q in v.index[v['Signal'] > 0]: # payer signals
    rsilev,sgnl,entry,sl = v.loc[q,:]
    try:
        q_end = xa_50[q:].index[xa_50[q:]][0]
    except: # Close at last month day if no closing signal
        q_end = xa_50[q:].index[-1]
    tmpsgnlrun = v.loc[q:q_end,:]
    stopped = (tmpsgnlrun.loc[:,y_name] < sl).any()
    if not stopped:
        exit_level = v.loc[q_end,y_name]
    else:
        exit_level = sl
    pnl = (exit_level - entry)*sgnl
    duration = (q_end-q).days
    r_unit = pnl/abs((sl-entry))
    df_pcr_test_trading_res.loc[q] = sgnl,duration,pnl,r_unit



for k,v in dic_pcr_run.items():
    xb_50 = (v['RSI_'+str(n_rsi_l)].shift()>50).\
        to_frame().merge((v['RSI_'+str(n_rsi_l)]<50),
                         left_index=True,
                         right_index=True).all(axis=1)
    xa_50 = (v['RSI_'+str(n_rsi_l)].shift()<50).\
        to_frame().merge((v['RSI_'+str(n_rsi_l)]>50),
                         left_index=True,
                         right_index=True).all(axis=1)
    
    for q in v.index[v['Signal'] < 0]: # receiver signals
        rsilev,sgnl,entry,sl = v.loc[q,:]
        try:
            q_end = xb_50[q:].index[xb_50[q:]][0]
        except: # Close at last month day if no closing signal
            q_end = xb_50[q:].index[-1]
        tmpsgnlrun = v.loc[q:q_end,:]
        stopped = (tmpsgnlrun.loc[:,y_name] > sl).any()
        if not stopped:
            exit_level = v.loc[q_end,y_name]
        else:
            exit_level = sl
        pnl = (exit_level - entry)*sgnl
        duration = (q_end-q).days
        r_unit = pnl/abs((sl-entry))
        df_pcr_test_trading_res.loc[q] = k,sgnl,duration,pnl,r_unit
        
    for q in v.index[v['Signal'] > 0]: # payer signals
        rsilev,sgnl,entry,sl = v.loc[q,:]
        try:
            q_end = xa_50[q:].index[xa_50[q:]][0]
        except: # Close at last month day if no closing signal
            q_end = xa_50[q:].index[-1]
        tmpsgnlrun = v.loc[q:q_end,:]
        stopped = (tmpsgnlrun.loc[:,y_name] < sl).any()
        if not stopped:
            exit_level = v.loc[q_end,y_name]
        else:
            exit_level = sl
        pnl = (exit_level - entry)*sgnl
        duration = (q_end-q).days
        r_unit = pnl/abs((sl-entry))
        df_pcr_test_trading_res.loc[q] = k,sgnl,duration,pnl,r_unit
###############################################################################
# In[R-Multiples]
# PCR Model R-Multiples Analysis
from statsmodels.distributions.empirical_distribution import ECDF
## Plot R-Mult Histogram
df_pcr_test_trading_res['R'].\
    plot.hist(title='R-Multiples', density=True);plt.tight_layout();plt.show()
## R-Multiples ECDF
recdf = ECDF(df_pcr_test_trading_res['R'])
recdf = ECDF(100*df_pcr_test_trading_res['pnl'])
## R-Multiples simulation
N_sim = 10000
total_years = (df_pcr_test_trading_res.index[-1] - \
               df_pcr_test_trading_res.index[0]).days/364
n_trades_yr = int(df_pcr_test_trading_res['R'].shape[0]/total_years)
N_sample = n_trades_yr
runs_myR = sim_path_R(recdf, sample_size=N_sample, paths_size=N_sim)
### plot
runs_myR.plot(title=f'Cumulative R Simulations\n'\
              f'N(paths)={N_sim}, N(sample) ={N_sample}',
              legend=False)
plt.grid(alpha=0.5, linestyle='--')
plt.tight_layout()
plt.show()
## Simulation results
simmean = runs_myR.mean(axis=1)
simstd = runs_myR.std(axis=1)
simqtl = runs_myR.quantile(q=(0.025,1-0.025),axis=1).T
#simres = [simmean, simmean-2*simstd, simmean+2*simstd, 
#          simqtl]
res_2021 = (pd.Series(0).append(100*df_pcr_test_trading_res.\
                                loc['2021']['pnl'].iloc[:n_trades_yr])).\
    cumsum().reset_index(drop=True)
res_2013 = (pd.Series(0).append(100*df_pcr_test_trading_res.\
                                loc['2013']['pnl'].iloc[:n_trades_yr])).\
    cumsum().reset_index(drop=True)
res_2022 = (pd.Series(0).append(100*df_pcr_test_trading_res.\
                                loc['2022']['pnl'].iloc[:n_trades_yr])).\
    cumsum().reset_index(drop=True)
res_2016 = (pd.Series(0).append(100*df_pcr_test_trading_res.\
                                loc['2016']['pnl'].iloc[:n_trades_yr])).\
    cumsum().reset_index(drop=True)
res_2023 = (pd.Series(0).append(100*df_pcr_test_trading_res.\
                                loc['2023']['pnl'].iloc[:n_trades_yr])).\
    cumsum().reset_index(drop=True)
simres = [simmean, simmean-2*simstd, simmean+2*simstd, 
          res_2021,res_2013,res_2022,res_2016,res_2023]
avgsim = pd.concat(simres,axis=1)
avgsim.columns = ['Exp. Path','LB(2std)','UB(2std)',
                  '2021','2013','2022','2016','2023']
plt_ttl = 'Cumulative PnL Simulations\nMean path\n'\
            f'N(paths)={N_sim}, N(sample) ={N_sample}'
plt_ttl_pnl = 'Cumulative PnL (bp)'
plt_colors = ['darkcyan','lightsalmon','yellowgreen', 'thistle',
              'wheat', 'lightsteelblue','silver','navy']
(avgsim/1).plot(title=plt_ttl_pnl, style=[':','--','--','-','-','-','-'], 
            color=plt_colors)
plt.grid(alpha=0.5, linestyle='--')
plt.tight_layout()
plt.show()

## Simulation stats
(runs_myR.mean()/runs_myR.std()).mean()
(runs_myR.mean()/runs_myR.std()).std()
(runs_myR.mean()/runs_myR.std()).plot.hist(title =\
                                'Sharpe Ratios\nCum. R Sims', 
                                density = True)
# In[Position Sizing]
N_sim = 1000
N_sample = n_trades_yr
dic_bal = {}
for n in range(N_sim):
    sim_r = empirical_sample(recdf,N_sample)
    init_capital = 100 
    pos_size_pct = 0.009
    pnl = np.array([])
    balance = np.array([init_capital])
    for r in sim_r:
        trade_pos_size = pos_size_pct*balance[-1]
        trade_pnl = r*trade_pos_size
        pnl = np.append(pnl,trade_pnl)
        balance = np.append(balance,balance[-1]+trade_pnl)
    dic_bal[f'run{n+1}'] = balance
    
## Equity curve sim results    
df_bal = pd.DataFrame(dic_bal)
df_bal.plot(title=f'Equity Curve Sim\n'\
              f'N(paths)={N_sim}, N(sample)={N_sample}',
              legend=False)
plt.grid(alpha=0.5, linestyle='--')
plt.tight_layout()
plt.show()
### Most probable outcomes
pdf_bal_mpo = pd.DataFrame([np.quantile(df_bal,.25,axis=1),
                           df_bal.mean(axis=1),
                           np.quantile(df_bal,.5,axis=1),
                           np.quantile(df_bal,.75,axis=1)]).T.\
    rename(columns=dict(zip(range(4),['q25%','Mean','Med','q75%'])))
pdf_bal_mpo.plot(title=f'Expected Equity Curve\n 5k DV01 {y_name}',
                 style=['--','-','-','--'])
plt.tight_layout()
plt.show()

## Sharpe ratios dist
sharpes = np.array((df_bal.mean()-100)/df_bal.std())
plt.hist(sharpes, density = True)
plt.title('Sharpe Ratios\nEquity Curve Sims')
plt.tight_layout()
plt.show()

## Sharpes stats
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

# In[Reg Betas Research]
for n_years in range(2,11):
    ## REGRESSION
    n_yr_train = n_years
    # data split dates
    str_end_train = '2022-12-31'
    str_start_train = str(np.datetime64(str_end_train) -\
                          np.timedelta64(365*n_yr_train,'D'))
    str_start_test = str(np.datetime64(str_end_train) - np.timedelta64(1,'D'))
    str_end_test = str(np.datetime64(str_start_test) + np.timedelta64(4*3,'W'))
    # data split
    data_train, data_test = sliceDataFrame(data, 
                                           dt_start=str_start_train, 
                                           dt_end=str_end_train, 
                                           lst_securities=tmplst),\
        sliceDataFrame(data, 
                       dt_start=str_start_test, 
                       dt_end=str_end_test,
                      lst_securities=tmplst)
    
    # response variable
    y_name = 'TIIE1Y1Y'
    #y_name = 'T5s7s10s'
    y_train, y_test = data_train[y_name], data_test[y_name]
    
    # features set
    try:
        X_train, X_test = data_train.drop(y_name, axis=1),\
            data_test.drop(y_name, axis=1)
    except KeyError:
        #raise KeyError('target var not in data')
        X_train, X_test = data_train, data_test
    #else:
    #    X_train, X_test = data_train.drop(y_name, axis=1),\
    #        data_test.drop(y_name, axis=1)
            
    # data normalization
    sclr = StandardScaler()
    sclr.fit(X_train)

    # PC Regression
    # PCA train
    n_comps = 3
    pca = PCA(n_components = n_comps)
    pca_scores_train = pca.fit_transform(sclr.transform(X_train))

    # y var vs the PCA scores linear regression
    rgr = LinearRegression()
    pcr = rgr.fit(pca_scores_train, y_train)
    
    # save coefs
    df_pcr_coefs = df_pcr_coefs.append(
        pd.DataFrame(pcr.coef_, index=['PC1','PC2','PC3']).T)

df_pcr_coefs.index = range(2,11)
df_pcr_coefs.plot();plt.tight_layout();plt.show()

# In[Cointegration]
###############################################################################
## COINTEGRATION
###############################################################################
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.tsa.vector_ar.vecm as ts_vecm

# data split sizes for train/test
test_pct = 0.20
n_test = 21*1
n_train = int(n_test*(1-test_pct)/test_pct)
assert((1-test_pct) == n_train/(n_train+n_test))

# indices for train/test backtesting windows
idx_df = getBacktest_idx(data.shape[0], n_train, n_test)

# subset of eleigible universe
lstcmb_bonos = ['M 10% D24', 'M 5.75% M26',
       'M 7.5% J27', 'M 8.5% M29', 'M 7.75% M31', 'M 7.75% N34',
       'M 8.5% N38', 'M 7.75% N42', 'M 8% N47']
lstcmb_tiies = ['TIIE1Y', 'TIIE2Y', 'TIIE3Y', 
                'TIIE4Y', 'TIIE5Y', 'TIIE7Y', 'TIIE10Y']
lstcmb_bonos + lstcmb_tiies

# all possible pairs
comb = list(combinations(lstcmb_bonos + lstcmb_tiies, 3))
data.columns
data.columns[:62]

# 3-asset pfolio
tmplst2 = ['M 8.5% M29','M 7.75% M31', 'M 7.75% N34']
#tmplst2 = ['M 5.75% M26', 'M 8.5% M29', 'M 8% N47']
df_cv = pd.DataFrame(columns=tmplst2)
dic_test = {}
# coint backtest runs
for i in range(idx_df.shape[0]-1):
    # backtest window indices
    idx_run = idx_df.iloc[i,]
    tri, trf, tei, tef = data.index[idx_run]
    # price data check
    noPriceData = data[tmplst2].isnull().any(axis=1).loc[tri:trf,].any()
    if noPriceData:
        continue
    # train/test data split
    df_train, df_test = sliceDataFrame(data, dt_start=tri, dt_end=trf, lst_securities=tmplst2),\
                        sliceDataFrame(data, dt_start=tei,  dt_end=tef, lst_securities=tmplst2)
    # johansentest
    try:
        jt = ts_vecm.coint_johansen(df_train,0,1)
    except:
        pass
    #jt_rejH0 = np.any(jt.trace_stat >= jt.trace_stat_crit_vals[:,2])
    jt_rejH0 = (jt.trace_stat >= jt.trace_stat_crit_vals[:,2])[2]
    # cointegrating vector
    cvector = pd.DataFrame([jt.eig.copy(), jt.evec.T.copy()], 
                            index = ['eigenvalue', 'coint_vector']).T.\
                                sort_values('eigenvalue', ascending=False).loc[0,'coint_vector']
    # normalized cointegrating vector
    norm_cv = cvector*100/np.linalg.norm(cvector*100)
    pctnorm_cv = norm_cv / norm_cv[1]
    cz = df_train.apply(lambda c: c*pctnorm_cv*1, axis=1).sum(axis=1)
    df_cv.loc[i, tmplst2] = pctnorm_cv
    df_cv.loc[i,'mu'] = cz.mean()
    df_cv.loc[i,'ub'] = cz.mean() + 2*cz.std()
    df_cv.loc[i,'lb'] = cz.mean() - 2*cz.std()
    df_cv.loc[i,'pval'] = adfuller(cz)[1]
    df_cv.loc[i,'rejH0'] = jt_rejH0
    # coint serie out-of-sample
    cz_test = df_test.apply(lambda c: c*pctnorm_cv*1, axis=1).sum(axis=1)
    df_cz_test = pd.concat([cz_test, pd.Series(cz.mean(), index=cz_test.index), 
               pd.Series(cz.mean()+2*cz.std(), index=cz_test.index), 
               pd.Series(cz.mean()-2*cz.std(), index=cz_test.index)], 
              axis=1)
    df_cz_test.columns = ['z','z_mu','ub','lb']
    df_cz_test = maskSpread(df_cz_test)
    dic_test[i] = df_cz_test

# long-term coint weights
pctnorm_cv_longterm = np.array(df_cv[(df_cv[['TIIE3Y','TIIE7Y']].abs() < 1).all(axis=1)].mean()[tmplst2])
cz_lt = df_train.apply(lambda c: c*pctnorm_cv_longterm*1, axis=1).sum(axis=1)
cz_test_lt = df_test.apply(lambda c: c*pctnorm_cv_longterm*1, axis=1).sum(axis=1)

# 2-to-1 vs beta-weighted spread
nvbw = data.loc['2016':'2023',tmplst2].apply(lambda c: c*np.array([-57.05,100,-39.23])*1, axis=1).sum(axis=1)
nvbw_ub = nvbw.mean()+3*nvbw.std()
nvbw_lb = nvbw.mean()-3*nvbw.std()
ax = pd.DataFrame(nvbw, columns=['BWS']).plot(title='TIIE 3s5s7s')
ax.axhline(y=nvbw_ub, c='g', linestyle='--')
ax.axhline(y=nvbw_lb, c='r', linestyle='--')
plt.show()

# significant coint vectors (I(0)-spread in-sample) runs
df_cv_sel = df_cv[df_cv['rejH0']]
df_cv_adf = df_cv[df_cv['pval']<0.05]
# out-of-sample coint series runs
dic_test_sel = {key: dic_test[key] for key in df_cv_sel.index}
len(dic_test_sel)

### MEAN-REVERTING PFOLIO
# cointegration datarun
tmplst2 = ['TIIE3Y','TIIE5Y', 'TIIE7Y']
cj_train = data_train[tmplst2]
cj_test = data_test[tmplst2]
cjt = ts_vecm.coint_johansen(cj_train,0,1)

# coint test stats results
cjt_idx = [f'r<={n}' for n in range(cj_train.shape[1])]
pvth = ['90%','95%','99%']
cjt_trace_stat = pd.DataFrame(cjt.trace_stat, index = cjt_idx, columns=['tstat'])
cjt_trace_cv = pd.DataFrame(cjt.trace_stat_crit_vals, index = cjt_idx, columns = pvth)
cjt_egn_stat = pd.DataFrame(cjt.max_eig_stat, index = cjt_idx, columns=['tstat'])
cjt_egn_cv = pd.DataFrame(cjt.max_eig_stat_crit_vals, index = cjt_idx, columns = pvth)

# df summaries
dfcj_trace = pd.concat([cjt_trace_stat,cjt_trace_cv], axis=1)
dfcj_eign = pd.concat([cjt_egn_stat,cjt_egn_cv], axis=1)

# H0 rejection df
dfcj_trace_test = dfcj_trace.drop('tstat', axis=1).apply(lambda c: c < dfcj_trace['tstat'])
dfcj_eign_test = dfcj_eign.drop('tstat', axis=1).apply(lambda c: c < dfcj_eign['tstat'])

print('Trace Statistics\nReject H0\n', dfcj_trace_test)
print('\nEigenvalue Statistics\nReject H0\n', dfcj_eign_test)

# cointegration relationship
coint_vector = pd.DataFrame([cjt.eig.copy(), cjt.evec.T.copy()], 
                            index = ['eigenvalue', 'coint_vector']).T.\
                                sort_values('eigenvalue', ascending=False).loc[0,'coint_vector']

# normalized cointegrating vector
norm_cv = coint_vector*100/np.linalg.norm(coint_vector*100)
pctnorm_cv = norm_cv / norm_cv[1]

# coint run df
coint_z = cj_train.apply(lambda c: c*pctnorm_cv*100, axis=1).sum(axis=1)
df_crun_train = pd.concat([coint_z, pd.Series(coint_z.mean(), index=coint_z.index), 
           pd.Series(coint_z.mean()+2*coint_z.std(), index=coint_z.index), 
           pd.Series(coint_z.mean()-2*coint_z.std(), index=coint_z.index)], 
          axis=1)
df_crun_train.columns = ['z','z_mu','ub','lb']
df_crun_train['set'] = 'train'

# coint train run
#coint_z = cj_train.apply(lambda c: c*coint_vector, axis=1).sum(axis=1)
#coint_z = cj_train.apply(lambda c: c*np.round(coint_vector*100,0), axis=1).sum(axis=1)
coint_z = cj_train.apply(lambda c: c*pctnorm_cv*100, axis=1).sum(axis=1)

# plot specs
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (9,6))
fig.suptitle('Mean-reverting Portfolio', size = 26)
ax.set_xlabel('Date')
ax.set_ylabel('Cointegrated Price (z)')
# spread serie
ax.plot(coint_z, color='darkcyan', label='Cointegrated serie')
# spread limits
ax.axhline(y=coint_z.mean(), color='darkcyan', linestyle='--', alpha=0.5)
ax.axhline(y=np.max([coint_z.mean()+3*coint_z.std(), np.quantile(coint_z,0.95)]), 
           color='r', linestyle='--', alpha=0.5)
ax.axhline(y=np.min([coint_z.mean()-3*coint_z.std(), np.quantile(coint_z,0.05)]), 
           color='g', linestyle='--', alpha=0.5)
# plot formatting
ax.set_title(f'Cointegrating Relationship (p-val: {adfuller(coint_z)[1]:.2%})')
yaxis_format = tkr.FuncFormatter(lambda x, p: format(int(x), ','))
xaxis_format = mdates.DateFormatter('%Y-%b-%d')
xaxis_loc = mdates.DayLocator(interval=8)
ax.get_yaxis().set_major_formatter(yaxis_format)
ax.get_xaxis().set_major_formatter(xaxis_format)
ax.get_xaxis().set_major_locator(xaxis_loc)
plt.xticks(rotation=45, fontsize='x-small')

# plot output
fig.tight_layout()
plt.show()

# coint-model equation
str_coint_ = f'z ~ '
for n in range(pctnorm_cv.shape[0]):
    tmpsgn = np.sign(pctnorm_cv[n])
    if tmpsgn > 0:
        tmpopp = '+'
    else:
        tmpopp = '-'
    #str_coint_ += f' {tmpopp} {np.abs(coint_vector[n]):.2f}({cj_data.columns[n]})'
    #str_coint_ += f' {tmpopp} {np.abs(np.round(coint_vector*100,0)[n]):.0f}({cj_data.columns[n]})'
    str_coint_ += f' {tmpopp} {np.abs(np.round(pctnorm_cv*100,0)[n]):.0f}({cj_train.columns[n]})'
    
    
print(str_coint_)


# coint test run df
coint_z_test = cj_test.apply(lambda c: c*pctnorm_cv*100, axis=1).sum(axis=1)
df_crun_test = pd.concat([coint_z_test, pd.Series(coint_z.mean(), index=coint_z_test.index), 
           pd.Series(coint_z.mean()+3*coint_z.std(), index=coint_z_test.index), 
           pd.Series(coint_z.mean()-3*coint_z.std(), index=coint_z_test.index)], 
          axis=1)
df_crun_test.columns = ['z','z_mu','ub','lb']
df_crun_test['set'] = 'test'

# plot specs
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (9,6))
fig.suptitle('Mean-reverting Portfolio', size = 26)
ax.set_xlabel('Date')
ax.set_ylabel('Cointegrated Price (z)')

# spread series
ax.plot(coint_z_test, color='darkcyan', label='Cointegrated serie')

# spread limits
ax.axhline(y=coint_z.mean(), color='darkcyan', linestyle='--', alpha=0.5)
ax.axhline(y=np.max([coint_z.mean()+3*coint_z.std(), np.quantile(coint_z,0.95)]), 
           color='r', linestyle='--', alpha=0.5)
ax.axhline(y=np.min([coint_z.mean()-3*coint_z.std(), np.quantile(coint_z,0.05)]), 
           color='g', linestyle='--', alpha=0.5)
# plot formatting
ax.set_title(f'Test (p-val: {adfuller(coint_z)[1]:.2%})')
yaxis_format = tkr.FuncFormatter(lambda x, p: format(int(x), ','))
xaxis_format = mdates.DateFormatter('%Y-%b-%d')
xaxis_loc = mdates.DayLocator(interval=1)
ax.get_yaxis().set_major_formatter(yaxis_format)
ax.get_xaxis().set_major_formatter(xaxis_format)
ax.get_xaxis().set_major_locator(xaxis_loc)
plt.xticks(rotation=45, fontsize='x-small')
# plot output
fig.tight_layout()
plt.show()

# coint-model equation
str_coint_ = f'z ~ '
for n in range(pctnorm_cv.shape[0]):
    tmpsgn = np.sign(pctnorm_cv[n])
    if tmpsgn > 0:
        tmpopp = '+'
    else:
        tmpopp = '-'

    str_coint_ += f' {tmpopp} {np.abs(np.round(pctnorm_cv*100,0)[n]):.0f}({cj_test.columns[n]})'
    
print(str_coint_)
























