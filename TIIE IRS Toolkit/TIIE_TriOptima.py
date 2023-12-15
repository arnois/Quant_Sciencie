# -*- coding: utf-8 -*-
"""
TriOptima

@author: jquintero
"""
#%%############################################################################
# MODULES
###############################################################################
import os
import QuantLib as ql
import pandas as pd
#import numpy as np
#import requests
from datetime import date as dt

#import numpy as np
import sys
import warnings
warnings.simplefilter("ignore")
#%%############################################################################
# GLOBALS & UDMODULES
###############################################################################
# WD
#str_cwd = os.path.dirname(os.path.realpath(sys.argv[0]))
str_cwd = '\\\\tlaloc\\cuantitativa\\'+\
    'Fixed Income\\TIIE IRS Valuation Tool\\Arnua\\'
os.chdir(str_cwd)
sys.path.append(str_cwd)
# Files
str_inputsFileName = 'TIIE_CurveCreate_Inputs'
str_inputsFileExt = '.xlsx'
str_file = str_inputsFileName + str_inputsFileExt

# Valuation date
dt_today = dt(2023,11,16)
ql_dt_today = ql.Date(dt_today.day, dt_today.month, dt_today.year)
ql.Settings.instance().evaluationDate = ql_dt_today
#ql_dt_yest = ql.Mexico().advance(ql_dt_today,-1,ql.Days)
dt_posswaps = dt(2023,11,15)

# UDM
import TIIE_CurveCreate_v3 as curveC
import udf_TIIE_PfolioMgmt as udf_pfl

# Portfolios
# Swaps File
str_dt_posswaps = dt_posswaps.strftime('%Y%m%d')
str_posswps_file = r'E:\posSwaps\PosSwaps'+str_dt_posswaps+'.xlsx' # PosSwaps file 
df_tiieSwps = udf_pfl.setPfolio_tiieSwps(str_posswps_file) # Swaps' Portfolio

#%% Selected Trades
path_trioptima = r'C:\Users\jquintero\Downloads\pfolio_liveEx_20231116.xlsx'
path_trioptima = r'C:\Users\jquintero\Downloads\pfolio_dressReh_20231115.xlsx'

xlsheet = 'prevPfoliof' #'pfoliof'
pfoliof_tradeID = pd.read_excel(path_trioptima,xlsheet,usecols='C:C')

# TriOptima Book
df_pfoliof = df_tiieSwps.merge(pfoliof_tradeID,how='inner',
                  left_on='TradeID',
                  right_on='swp_ctrol').drop(columns=['swp_ctrol'])
#%% Input Data
dic_data = curveC.udf.pull_data(str_file, dt_today)
fxrate = dic_data['USDMXN_XCCY_Basis']['Quotes'][0]

# IborIndex
ibor_tiie_cf = udf_pfl.udf.set_ibor_TIIE(
        ql.FlatForward(0, ql.Mexico(), 0.1150, ql.Actual360()), 
        )

#%% Book's Risk & Valuation
dic_book_valrisk = udf_pfl.get_pfolio_riskval(dic_data, df_pfoliof)
dfpfolio_npv = udf_pfl.get_pflio_npv_atDate(dt_today, dic_data, df_pfoliof)
dfpfolio_cf_tdy = udf_pfl.get_pfolio_CF_atDate(dt_today, dfpfolio_npv, ibor_tiie_cf)

# Bucket Risk (KRR)
pfoliof_deltas = df_pfoliof[['TradeID']].merge(dic_book_valrisk['DV01_Swaps'],
                                          how='left',
                                          left_index=True,
                                          right_index=True)
pfoliof_deltas['Flat'] = pfoliof_deltas.drop(columns=['TradeID']).sum(axis=1)

#%% save
save_name = r'C:\Users\jquintero\Downloads\pfolio_deltas_20231116.xlsx'
pfoliof_deltas.to_excel(save_name)

# NPV by Swap
pfoliof_npvs = df_pfoliof[['TradeID']].merge(dic_book_valrisk['NPV_Swaps'],
                                          how='left',
                                          left_index=True,
                                          right_index=True)
# save
save_name = r'C:\Users\jquintero\Downloads\pfolio_npv_20231116.xlsx'
pfoliof_npvs.to_excel(save_name)
