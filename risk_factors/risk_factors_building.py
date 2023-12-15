# -*- coding: utf-8 -*-
"""
Factor Building

The purpose of this code is to determine a standard process for ubilding risk
factors that provide the basis for efficient exposure to different sources
of market regimes.

@author: arnulf
"""
#%% MODULES
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from seaborn import heatmap
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.mixture import GaussianMixture as GM

#%% DATA
data = pd.read_excel(r'H:\db\RiskFactors.xlsx', sheet_name='data', skiprows=4)
lst_cols = data.columns.tolist()
lst_cols[0] = 'Date'
data.columns = lst_cols
data = data.drop(0).set_index('Date', drop=True)
# Custom USD factor
dxy = pd.read_excel(r'H:\db\RiskFactors_DXY.xlsx', sheet_name='index')
dxy = dxy.rename(columns={'t':'Date'}).set_index('Date', drop=True)
# dataset
df = data.merge(dxy[['USDFX']], left_index=True, right_index=True)
df = df.astype(float)
df_ret = df.apply(np.log).diff()
#%% UDF
def roll_resid(y:pd.DataFrame, X:pd.DataFrame, w: int=252) -> pd.Series:
    """
    Calculates rolling residualization of time series given a set of exogenous
    variables. For a linear regression setting, the resulting factor will be 
    the estimated regression constant plus the residual.
    -------
    Args:
    - y (pd.DataFrame): The time series to residualize.
    - X (pd.DataFrame): The variables to factor out.
    - w (int): Window size for rolling linear regressions.
    
    Returns:
    -------
    pd.Series: Residualized time series.
    """
    # Exogenous variables for linear model
    exog = sm.add_constant(X)
    # Linear model
    rols = RollingOLS(y, exog, window=w)
    m = rols.fit()
    # Rolling coefficients
    rol_coef = m.params.dropna()
    rol_a = rol_coef[['const']]
    rol_beta = rol_coef.drop('const', axis=1)
    # Exog vars rolling preds
    pred_beta = (rol_coef*exog.loc[rol_beta.index, rol_beta.columns]).\
        sum(axis=1).rename(y.columns[0]).to_frame()
    # Exog vars rolling residuals
    pred_beta_err = y.loc[rol_beta.index] - pred_beta
    # Factor
    rol_f = rol_a.rename(columns={'const': y.columns[0]}) + pred_beta_err
    
    return rol_f.rename(columns={rol_f.columns[0]: rol_f.columns[0]+'_F'})

# function to plot correlation matrix among series
def plot_corrmatrix(df: pd.DataFrame, plt_size: tuple = (10,8), 
                    strttl: str = '') -> None:
    """
    Args:
        df: the data to plot
        plt_size: plot figure size
    Returns: 
        Correlation matrix among series.
    """
    # transform
    #df_ = pd.DataFrame(StandardScaler().fit_transform(df))
    #df_ = pd.DataFrame(Normalizer().fit_transform(df))
    df_ = df.copy()
    # corrmatrix
    plt.figure(figsize=plt_size)
    heatmap(
        df_.corr(method='spearman'), # 'pearson', 'spearman', 'kendall'        
        cmap='RdBu', 
        annot=False, 
        vmin=-1, vmax=1,
        fmt='.2f')
    plt.title(strttl, size=20)
    plt.tight_layout()
    plt.show()
    return None

    
#%% COMMODITIES MARGINAL RISK FACTOR
# Rolling window size
rws = 252*3
# 3y-rolling residualization of commodities returns
factor_Comm_ret = roll_resid(df_ret[['Commodities']], 
                             df_ret[['Equity', 'Rates']], rws)
# Commodities Risk Factor
factor_Comm = 100*np.exp(factor_Comm_ret.cumsum())

#%% CREDIT MARGINAL RISK FACTOR
# Rolling window size
rws = 252*3
# 3y-rolling resid of US Credit returns
factor_USIG_ret = roll_resid(df_ret[['US IG']], 
                             df_ret[['Equity', 'Rates']], rws)
factor_USHY_ret = roll_resid(df_ret[['US HY']], 
                             df_ret[['Equity', 'Rates']], rws)
# US Credit Risk Factors
factor_USIG = 100*np.exp(factor_USIG_ret.cumsum())
factor_USHY = 100*np.exp(factor_USHY_ret.cumsum())

# 3y-rolling resid of EU Credit returns
factor_EUIG_ret = roll_resid(df_ret[['Euro IG']], 
                             df_ret[['Equity', 'Rates']], rws)
factor_EUHY_ret = roll_resid(df_ret[['Euro HY']], 
                             df_ret[['Equity', 'Rates']], rws)
# EU Credit Risk Factors
factor_EUIG = 100*np.exp(factor_EUIG_ret.cumsum())
factor_EUHY = 100*np.exp(factor_EUHY_ret.cumsum())

# Credit Risk Factor
tmplst = [factor_USIG_ret, factor_USHY_ret, factor_EUIG_ret, factor_EUHY_ret]
factor_Credit_ret = pd.concat(tmplst, axis=1).mean(axis=1).rename('Credit_F')
factor_Credit = 100*np.exp(factor_Credit_ret.cumsum())

#%% EM MARGINAL RISK FACTOR
# Rolling window size
rws = 252*3
# 3y-rolling resid of EM Credit returns
factor_EMC_ret = roll_resid(df_ret[['EM Credit']], 
                             df_ret[['Rates']], rws)
# 3y-rolling resid of EM Equity returns
factor_EME_ret = roll_resid(df_ret[['EM Equity']], 
                             df_ret[['Equity']], rws)
# EM Risk Factor
tmplst = [factor_EMC_ret, factor_EME_ret]
factor_EM_ret = pd.concat(tmplst, axis=1).mean(axis=1).rename('EM_F')
factor_EM = 100*np.exp(factor_EM_ret.cumsum())

#%% RISK FACTORS CATEGORIES
df_CoreMacro = df[['Rates','Equity']].merge(factor_Credit, 
                                            how='right', 
                                            left_index=True, 
                                            right_index=True).\
    merge(factor_Comm, left_index=True, right_index=True)

df_SecondMacro = df[['USDFX', 'Local Inflation', 'Local Eqty']].\
    merge(factor_EM, how='right', left_index=True, right_index=True)

df_MacroStyle = df[['Equity Short Vol', 'Fixed Income Carry', 
                    'FX Carry', 'Trend Following']].loc[factor_EM.index]

df_EqtyStyle = df[['Low Risk', 'Momentum', 
                    'Quality', 'Value', 'Small Cap']].loc[factor_EM.index]

df_F = pd.concat(
    [df_CoreMacro, df_SecondMacro, df_MacroStyle, df_EqtyStyle], axis=1)
df_F_ret = df_F.apply(np.log).diff().dropna()
#%% RISK FACTORS CORR
plot_corrmatrix(df_F.apply(np.log).diff().dropna(), strttl='Factor Correlation')

#%% FACTOR MEAN RETURNS & VOLATILITIES
df_F_ret.mean()*252
df_F_ret.std()*np.sqrt(252)

#%% RISK FACTORS CLUSTERS

# Cluster number lookout
n_comp = np.arange(2, 21)
lst_gmm = [GM(n, random_state=13).fit(df_F_ret) for n in n_comp]
# GMM Models Comparison
gmm_model_comparisons = pd.DataFrame({
    "BIC" : [m.bic(df_F_ret) for m in lst_gmm],
    "AIC" : [m.aic(df_F_ret) for m in lst_gmm]},
    index=n_comp)

# Optimal components
n_opt = gmm_model_comparisons.\
    index[gmm_model_comparisons.apply(np.argmin)['BIC']]

plt.figure(figsize=(8,6))
ax = gmm_model_comparisons[["BIC","AIC"]].\
    plot(color=['darkcyan', 'b'], linestyle=':', marker='o', 
         mfc='w', xticks=gmm_model_comparisons.index)
plt.axvline(x=n_opt, color='orange', alpha=0.25)
plt.xlabel("Number of Clusters")
plt.ylabel("Score"); plt.show()

#%% FACTORS MARKET REGIMES
# NOTE: Remember that GMMs are generative models and the number of components
# thus measures how well the GMM works as a density estimator, not as how well
# suits a clustering algo.
gmm = GM(n_opt, random_state=13).fit(df_F_ret)
pd.DataFrame(gmm.means_)
labels = gmm.predict(df_F_ret)
data_wclust = df_F.drop(df_F.index[0]).copy()
data_wclust['c'] = labels
# Regime labels
df_F_ret['c'] = labels
# Regime probas
regime_prob = pd.DataFrame(gmm.predict_proba(df_F_ret.drop('c', axis=1)),
                           index=df_F_ret.index)

# Factor Regime Display
colors = ['red', 'blue', 'green', 'yellow', 'darkcyan', 'orange', 'purple']
markers = ['.', '+', '<', 's']
alphas = np.round(data_wclust['c'].value_counts().to_numpy()/data_wclust.shape[0],2)
s_name = 'Trend Following'
for n in range(n_opt):
    tmpdata = data_wclust[data_wclust['c'] == n]#.loc['2021-06':'2023-12']
    plt.scatter(tmpdata.index, tmpdata[s_name], s=20, alpha=alphas[n],
                c=colors[n], marker=markers[n])
plt.tight_layout()
plt.show()

# Factors Moments Between Regimes
import seaborn as sns
df_F_regime = df_F_ret.groupby('c')
## Mean Returns
fig, ax = plt.subplots(figsize=(6, 6))
axsns = sns.heatmap(df_F_regime.mean().T*252, cmap ='RdYlGn', 
                    linewidths = 0.21, annot = True, fmt='0.1%', 
                    xticklabels=range(1,n_opt+1))
axsns.set_title('Mean Returns'); axsns.set_xlabel('Market Regime')
plt.tight_layout(); plt.show()

## Volatility
fig, ax = plt.subplots(figsize=(6, 6))
axsns = sns.heatmap(df_F_regime.std().T*np.sqrt(252), cmap ='crest', 
                    linewidths = 0.21, annot = True, fmt='0.1%', 
                    xticklabels=range(1,n_opt+1))
axsns.set_title('Volatility'); axsns.set_xlabel('Market Regime')
plt.tight_layout(); plt.show()

## Skewness
fig, ax = plt.subplots(figsize=(6, 6))
axsns = sns.heatmap(df_F_regime.skew().T, cmap ='RdYlGn', linewidths = 0.21, 
            annot = True, fmt='0.1g', xticklabels=range(1,n_opt+1))
axsns.set_title('Daily Returns Skewness'); axsns.set_xlabel('Market Regime')
plt.tight_layout(); plt.show()

## Kurtosis
from scipy.stats import kurtosis
df_F_regime_kurt = pd.concat(
    [df_F_ret[df_F_ret['c']==n].apply(kurtosis).drop('c') 
     for n in range(n_opt)],
    axis=1)

### threshold
from scipy.stats import norm
tmpdata = norm.rvs(size=(df_F_ret.shape[0],32), random_state=13)
tmpdata_muk = kurtosis(tmpdata).mean()
tmpdata_stdk = kurtosis(tmpdata).std()

fig, ax = plt.subplots(figsize=(6, 6))
axsns = sns.heatmap(df_F_regime_kurt, linewidths = 0.21, 
            annot = True, fmt='0.1g', xticklabels=range(1,n_opt+1))
axsns.set_title('Daily Returns Kurtosis'); axsns.set_xlabel('Market Regime')
plt.tight_layout(); plt.show()

# Factors Correlations Between Regimes
for n in range(n_opt):
    tmpdata = data_wclust[data_wclust['c'] == n].drop('c', axis=1)
    plot_corrmatrix(tmpdata.apply(np.log).diff().dropna())

# Data segregation by Market Regimes
df_crisis = data_wclust[data_wclust['c']==0].drop('c', axis=1)
df_shifts = data_wclust[data_wclust['c']==1].drop('c', axis=1)
df_trends = data_wclust[data_wclust['c']==2].drop('c', axis=1)

#
df_crisis[['Rates', 'Credit_F']].diff().corr('kendall')
df_shifts[['Rates', 'Credit_F']].diff().corr('kendall')
df_trends[['Rates', 'Credit_F']].diff().corr('kendall')






