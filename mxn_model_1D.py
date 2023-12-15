# -*- coding: utf-8 -*-
"""
MXN Analysis
Frequency: Daily

@author: jquintero
"""
import pandas as pd
import yfinance as yf
from yahoo_fin import stock_info as si
import datetime
import pandas_datareader as pdr
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

# In[UDF]:
##############################################################################
### UDFs
##############################################################################

def plot_rollcorr(df, yvar, lbp = 64):
    """
    Rolling Correlations

    Parameters
    ----------
    df : pandas.DataFrame
        Data to use for computing correlations.
    yvar : str
        Variable name being the dependent variable in the correlation coef.
    lbp : Lookback period, optional
        Rolling window. The default is 64.

    Returns
    -------
    None. Prints pairwise correlations of yvar between the rest ones.

    """
    xvars = df.columns.drop(yvar)
    rollcorr = df.rolling(lbp).corr(pairwise=True)[yvar].dropna()
    df_rc = pd.DataFrame()
    for x in xvars:
        df_rc[x] = rollcorr.loc[slice(None),x]
    sns.lineplot(data=df_rc)
    
    return None

def test_ADF(ts, rgrs = 'c', ic_lag = 'AIC'):
    """
    Augmented Dickey-Fuller test for unit root in the timeseries process (H0)
    against the alternative for no unit root presence.

    Parameters
    ----------
    ts : pandas.Series
        Timeseries data to test for unit root.
    rgrs : str, optional, default: 'c'
        Whether to consider or not additional terms in the regression:
            nc: no constant, no trend
            c: constant
            ct: constant + linear trend
            ctt: constant + linear trend + quadratic trend
    ic_lag : str, optional, default: 'AIC'
        Method to use when automatically determining the lag length    
        
    Returns
    -------
    None. Prints ADF test statistic with its p-value. Besides, critical values
    are showed up for 1, 5 and 10% levels.

    """
    # modules
    from statsmodels.tsa.stattools import adfuller
    # test
    result = adfuller(ts)
    # output
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
        
        
def plot_ACF_PACF(ts, fsz = (17,11)):
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    """ACF and PACF
    Plots a timeseries ACF and PACF for its price and returns levels.
    
    Parameters
    ----------
    ts : pandas.DataFrame
        The data with price, logreturn and non-relative returns (1st-diff).
    fsz : tuple, optional, default: (17,11)
        Plot's figure size (height,width).
    
    """
    # data proc
    ts_px = ts.iloc[:,0]
    ts_ret = ts.iloc[:,1]
    ts_ret_sqrd = ts_ret**2
    df_ttl1 = ts_px.name
    df_ttl2 = ts_ret.name
    df_ttl3 = ts_ret.name+'_SQRD'
    df = ts
    df_lags = min(int((df.shape[0]-3)/2)-1,20)
    # plot specs
    fig, axes = plt.subplots(3, 3, sharex=False, figsize = fsz)
    # plot levels
    ## price
    axes[0, 0].plot(ts_px.values)
    axes[0, 0].set_title(df_ttl1)
    ## logreturns
    axes[1, 0].plot(ts_ret.values)
    axes[1, 0].set_title(df_ttl2)
    ## non-relative returns
    axes[2, 0].plot(ts_ret_sqrd.values)
    axes[2, 0].set_title(df_ttl3)
    # plot acf and pacf
    ## price
    plot_acf(ts_px.values, ax=axes[0, 1], 
             zero = False, title='ACF', lags = df_lags)
    plot_pacf(ts_px.values, ax=axes[0, 2], 
              zero = False, title='PACF', lags = df_lags)
    ## logreturns
    plot_acf(ts_ret.values, ax=axes[1, 1], 
             zero = False, title='ACF')
    plot_pacf(ts_ret.values, ax=axes[1, 2], 
              zero = False, title='PACF', lags = df_lags)
    ## non-relative returns
    plot_acf(ts_ret_sqrd.values, ax=axes[2, 1], 
             zero = False, title='ACF')
    plot_pacf(ts_ret_sqrd.values, ax=axes[2, 2], 
              zero = False, title='PACF', lags = df_lags)

# In[Code]:
yf.pdr_override()

# Lookback period (in days)
lbp = 545

# Variables
tickers = si.get_currencies().Symbol
str_tickers = [item.replace(".", "-") for item in tickers]
start_date = datetime.datetime.now() - datetime.timedelta(days=lbp)
end_date = datetime.date.today()

# Other market variables
tkr_X_names = ['dji','VWO','UST5','gaso','copper']
tkr_X = ['^DJI','VWO','ZF=F','RB=F','HG=F']

# Download fx rates and markets historical data 
str_rate = 'MXN'
str_tkr = str_rate+'=X'
str_tkr = [str_tkr] + tkr_X
df = pdr.get_data_yahoo(str_tkr, start_date, end_date)

# Data mgmt
df = df['Adj Close']
df.columns = ['MXN']+tkr_X_names
df.fillna(method='ffill', inplace=True)
dfd1 = df.diff().dropna()

# Plot
sns.set_theme(style='ticks')
sns.pairplot(dfd1)

# In[Model]:
##############################################################################
'ADF test rejects unit root for every var in dfd1'
##############################################################################

# ARMA-GARCH
tmpY = pd.concat([df['MXN'],df['MXN'].diff()],axis=1).dropna()
tmpY.columns = ['MXN','Diff']
plot_ACF_PACF(tmpY)


# RollRgrss - using statsmodels
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
str_Y = 'MXN'
str_X = dfd1.columns.to_list()
str_X.pop(dfd1.columns.to_list().index('MXN'))
X =sm.add_constant(dfd1[str_X])
y = dfd1[str_Y]
rwndw = 64
lm_rol = RollingOLS(y,X,window=rwndw)
lm_rolf = lm_rol.fit()
lm_rolf_params = lm_rolf.params.copy()
fig = lm_rolf.plot_recursive_coefficient(variables=str_X, figsize=(14,18))

# Roll4cast
betas = lm_rolf_params.dropna()
X_=X.shift().loc[lm_rolf_params.dropna().index]
y_hat = pd.DataFrame(index = betas.index, columns = ['y_hat'])
for t in betas.index:
    tmpb = betas.loc[t]
    tmpX = X_.loc[t]
    tmpbX = tmpb*tmpX
    y4cast = tmpbX.sum()
    y_hat.loc[t] = y4cast

mxn = pd.DataFrame(df[str_Y].shift().loc[y_hat.index].dropna())
mxn_hat = y_hat['y_hat'] + mxn['MXN']
r4casts = pd.DataFrame([df[str_Y].loc[mxn_hat.index],mxn_hat])
r4casts = r4casts.T
r4casts.columns = ['MXN','Model']
r4casts.tail(128).plot(title='Daily USDMXN Rate',
                       color=['darkcyan','darkblue'],
                       style=['-',':'])
plt.grid(alpha=0.5, linestyle='--')

# Residuals
r4casts['resid'] = r4casts.MXN - r4casts.Model
r4casts.resid.tail(128).plot(title=f'MXN Model\nLast {128} Residuals')
plt.grid(alpha=0.5, linestyle='--')
test_ADF(r4casts.resid.tail(128))
plot_ACF_PACF(r4casts[['MXN','resid']])
r4casts['std_resid'] = (r4casts.resid - r4casts.resid.mean())/r4casts.resid.std()

# Err Measures
mae = (abs(r4casts.resid)).mean()
mae
##############################################################################
'Entry signal with Model stdz. residuals'
##############################################################################
# Trading Setups
zlim = 2
zlimu_dt = r4casts.std_resid >= zlim
zlimd_dt = r4casts.std_resid <= -1*zlim
r4casts['oob'] = np.NaN
r4casts['oob'][zlimu_dt] = -1
r4casts['oob'][zlimd_dt] = 1

# Trading Simulation
pos = None
pos_entry = None
pos_dt = None
pos_rtpnl = pd.DataFrame(index = r4casts.index, columns = ['rtpnl'])
pos_pnl = pd.DataFrame(index = r4casts.index, columns = ['pnl'])
for row in r4casts.iterrows():
    dt = row[0]         # date
    s = row[1]['oob']   # signal
    # address current pos
    if pos is not None:
        pos_val = (row[1]['MXN'] - pos_entry)*pos
        pos_rtpnl.loc[dt] = pos_val
        # check for exit signal
        stde = row[1]['std_resid']
        if pos == 1:
            pos_isexit = stde >= 0
        else:
            pos_isexit = stde <= 0
        if pos_isexit:
            pos_pnl.loc[dt] = pos_val
            pos = None
            pos_entry = None
            pos_dt = None
    # check for entry signal
    if not np.isnan(s):
        entry = row[1]['MXN']   # entry price
        if pos is None:                 # open trade
            pos = s.copy()
            pos_entry = entry.copy()
            pos_dt = dt
        elif pos == 1 and s == -1:      # close long to open sell
            pos_pnl.loc[dt] = (entry - pos_entry)*pos
            pos_rtpnl.loc[dt] = pos_pnl.loc[dt].copy()
            pos == s.copy()
            pos_entry = entry.copy()
            pos_dt = dt
        elif pos == -1 and s == 1:      # close short to open buy
            pos_pnl.loc[dt] = (entry - pos_entry)*pos
            pos_rtpnl.loc[dt] = pos_pnl.loc[dt].copy()
            pos == s.copy()
            pos_entry = entry.copy()
            pos_dt = dt

# results            
sim_trade = pd.concat([r4casts[['MXN','std_resid','oob']],
                       pos_pnl,pos_rtpnl],
                      axis=1)
sim_trade.loc[sim_trade['pnl'].dropna().index]

# stats
sim_trade.rtpnl.dropna().plot.hist()
np.quantile(sim_trade.rtpnl.dropna(),0.20)
sim_trade.pnl.dropna().cumsum().plot()
signal_n = sim_trade.shape[0]-sim_trade.oob.isnull().sum()
signal_acc = sim_trade.pnl[sim_trade.pnl>0].count() / signal_n        




# In[ModelExt]:

df['MXN'] = 1/df['Adj Close']
df['MXN_RET'] = df['MXN'].pct_change()

# Download pandemic data
url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
df_owid = pd.read_csv(url)

# Filter data by location
df_owid_MX = df_owid[df_owid.location == 'Mexico']
str_selCols = ['date','total_cases','new_cases','population',
               'total_vaccinations','people_vaccinated']
df_owid_MX = df_owid_MX[str_selCols]
df_owid_MX.set_index(pd.to_datetime(df_owid_MX['date']),inplace=True)
df_owid_MX.drop(['date'], axis=1, inplace=True)

# Filter out non valids
idx_fvi = df_owid_MX.total_cases.first_valid_index()
df_covid = df_owid_MX[idx_fvi:]
df_covid = df_covid.fillna(0)

# Merge
dff = pd.concat([df_covid,df], axis=1)
dff = dff.dropna()
dff.index.set_names('date', inplace=True)

# Vars
dff['case_rate'] = dff['total_cases']/dff['population']
dff['vacc_rate'] = dff['people_vaccinated']/dff['population']
dff['new_vacc'] = dff['total_vaccinations'].diff()

# Plot
dff.plot.scatter(x = 'total_cases', y = 'MXN')
dff.plot.scatter(x = 'total_cases', y = 'Adj Close')


