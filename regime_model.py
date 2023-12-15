# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:32:47 2023

@author: jquintero
"""

import warnings
warnings.filterwarnings('ignore')
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_parquet(r'H:\Python\tc_algot_pfl_etf\ETF_DATA.parquet')

returns = data.XLK.CLOSE.apply(np.log).diff().dropna().rename('return').astype(float)

model = sm.tsa.MarkovRegression(endog=returns,
                                k_regimes=2,
                                trend='c',
                                switching_variance=True)

result = model.fit()
regime_probability = result.smoothed_marginal_probabilities[1].to_frame('regime')

regime_probability.plot(figsize=(16,4))
plt.title('XLK Volatility Regimes')
plt.axhline(.5, color='black')
plt.ylabel('Low/High Volatility')
plt.show()