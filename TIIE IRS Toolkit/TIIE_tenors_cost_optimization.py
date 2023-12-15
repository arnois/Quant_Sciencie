
#%% MODULES
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
import os
import sys
import xlwings as xw
#%% CUSTOM MODULES
user_path = os.getcwd()
main_path = '//TLALOC/Cuantitativa/Fixed Income/TIIE IRS Valuation Tool/' +\
    'Main Codes/Portfolio Management/OOP codes/'
sys.path.append(main_path)
import curve_funs as cf

#%% DATA
str_file = r'H:\Python\TIIE IRS Toolkit\TIIE_IRS_Trading.xlsm'
input_sheets = ['USD_OIS', 'USD_SOFR', 
                'USDMXN_XCCY_Basis', 'USDMXN_Fwds']
dic_data = {}
# Pull every curve input
for sheet in input_sheets:
    dic_data[sheet] = pd.read_excel(str_file, sheet)
dic_data['MXN_TIIE'] = pd.read_excel(str_file, 'MXN_TIIE', 
                                     usecols='A:E', nrows=14)
# Filter out unused tenors
dic_data['USD_OIS'].drop(dic_data['USD_OIS'].tail(1).index, inplace=True)
dic_data['USD_SOFR'].drop(dic_data['USD_SOFR'].tail(2).index, inplace=True)

# Bootstrap disc/forc curves
curves = cf.mxn_curves(dic_data)

#%% IRS Inputs
df_isheet = pd.read_excel(str_file, 'MXN_TIIE', nrows=14)
df_params = df_isheet.iloc[0:4,21:23].set_index('OptField')
#df_params = pd.read_excel(str_file, 'MXN_TIIE', usecols='V:W', nrows=4, index_col=0)

#start = datetime(2030, 9, 5)
#end = datetime(2033, 9, 1)
#initial_rate = 9.0050
#target_rate = 8.9800
notional = 1e9
start = df_params.loc['start'][0]
end = df_params.loc['end'][0]
initial_rate = df_params.loc['init'][0]
target_rate = df_params.loc['target'][0]

# BidOffer quotes
bidoffer = df_isheet.iloc[0:14,[1,7,8]]
#bidoffer = pd.read_excel(str_file, 'MXN_TIIE', usecols='H:I', nrows=14)
#bidoffer.insert(0,'Tenor',dic_data['MXN_TIIE']['Tenor'])
# BidOffer bounds
df_bounds = bidoffer[['Bid','Offer']].mean(axis=1).apply(lambda x: (x,x))
# Determine tenor bounds to control for
costcalc = df_isheet.iloc[0:14,10]
#costcalc = pd.read_excel(str_file, 'MXN_TIIE', usecols='K', nrows=14)
idx_tenors = np.where(costcalc == 'x')[0]
tenors_bounds = bidoffer.loc[idx_tenors].\
    apply(lambda x: (x['Bid']-0.15,x['Offer']+0.15), axis=1)
df_bounds.loc[idx_tenors] = tenors_bounds
bounds = df_bounds.to_list()
# Curve mids input
mid_quotes = bidoffer[['Bid','Offer']].mean(axis=1).to_numpy()
df_quotes = dic_data['MXN_TIIE'].copy()
df_quotes['Quotes'] = mid_quotes
curves.change_tiie(df_quotes)
swap = cf.tiieSwap(start, end, notional, initial_rate, curves)
print(f'FairRate (Mid): {swap.fairRate():.4%}')

#%% OBJECTIVE FUNCTION
def fair_rate_l2(tiie_array: np.ndarray, start, end, notional, fixed_rate) -> float:
    
    dftiie = dic_data['MXN_TIIE'].copy()
    dftiie['Quotes'] = tiie_array
    curves.change_tiie(dftiie)
    swap = cf.tiieSwap(start, end, notional, fixed_rate, curves)
    swap_rate = swap.fairRate()*100
    
    return 100*(target_rate - swap_rate)**2

#%% OPTIMIZATION
fixed_args = (start, end, notional, initial_rate)
a = datetime.now()
optimal_rates = minimize(fair_rate_l2, mid_quotes, args=fixed_args,
                         method='L-BFGS-B', bounds=bounds,
                         options = {'maxiter': 400})
b = datetime.now()
print(f'Opt. ETA: {(b-a).seconds/1:.2f} sec')
# Optimal inputs/outputs
optimal_tiies = optimal_rates.x
optimal_dftiie = dic_data['MXN_TIIE'].copy()
optimal_dftiie['Quotes'] = optimal_tiies
curves.change_tiie(optimal_dftiie)
swap = cf.tiieSwap(start, end, notional, initial_rate, curves)
print(f'FairRate (*Inputs): {swap.fairRate():.4%}')
# Cost rates
df_optimal_inputs = dic_data['MXN_TIIE'][['Tenor']].copy()
df_optimal_inputs.insert(1,'Mid',mid_quotes)
df_optimal_inputs.insert(2,'Opt',optimal_tiies)
print(f'Cost Rates\n{df_optimal_inputs}')

#%% PASTE RESULTS

wb = xw.books('TIIE_IRS_Trading.xlsm')
tiie_sheet = wb.sheets('MXN_TIIE')
tiie_sheet.range('Y2').value = df_optimal_inputs[['Opt']].values

#%% SAVE RESULTS
df_optimal_inputs.to_excel(r'C:\Users\jquintero\Downloads\cost_rates.xlsx')
