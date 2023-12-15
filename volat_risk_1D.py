# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:42:55 2022

@author: jquintero

Script dedicated to perform volatility and risk analysis over
daily prices
"""
##############################################################################
# MODULES
##############################################################################
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm

##############################################################################
# DATA
##############################################################################
csv_data_path = r'C:\Users\jquintero\db\data_1D.csv'
data = pd.read_csv(csv_data_path, index_col=0, 
                   parse_dates=True, infer_datetime_format=True)

# contract point values
df_cval = pd.DataFrame([2000, 1000, 1000, 1000, 1000, 100, 250, 50, 
                     50, 1000, 20, 50, 50, 5],
             index = ['TU1', 'FV1', 'TY1', 'UXY', 'RX1', 'GC1', 'HG1', 'So1', 
                      'Co1', 'CL1', 'NQ1', 'ES1', 'RTY', 'DM1'],
             columns=['cval'])
##############################################################################
# VaR
##############################################################################
# dates window
dt2 = data.index[-1]
dt1 = dt2 - datetime.timedelta(days=252*5)

# average contract size
df_avgSize = pd.DataFrame(
    [8.9, 8.8, 5.4, 3.9, 2.4, 1.9, 1.7, 1.3, 2.4, 1, 1.7, 2.3, 1, 2.1],
    index = df_cval.index,
    columns = df_cval.columns
    )

# contract size values
df_cvalSize = df_cval.mul(data.loc[dt1:dt2,df_cval.index].mean(), axis=0)
df_returns = data.loc[dt1:dt2,df_cval.index].apply(np.log).diff().dropna()

# conf level
alfa = 0.01
# VaR at alfa with empirical dist returns
df_cutoff = (1+df_returns).apply(lambda c: df_cvalSize.loc[c.name][0]*c).\
    apply(lambda c: np.quantile(c,alfa)).to_frame(name='cval')
df_VaR1D = df_cvalSize - df_cutoff

##############################################################################
# CONTRACT SIZE LIMITS
##############################################################################
limit_var1d = 400000*0.8
df_cSize_VaR1D_lim = (limit_var1d/df_VaR1D*.9).astype(int)









