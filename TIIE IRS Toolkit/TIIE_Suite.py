#---------------
#  Description
#---------------

"""


"""
#-------------
#  Libraries
#-------------
import os
import QuantLib as ql
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import timedelta, date, datetime
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")
import Funtions as fn
import udf_pricing as pr
import pickle
import xlwings as xw


    
a = datetime.now()

def menu():
    print('Please choose an option: ')
    print('\n1) Pricing             4) Collapse' +
          '\n2) Update Curves       5) Blotter ' +
          '\n3) Short End Pricing   6) Upload Blotter' +
          '\n7) End Session')

    option = int(float(input('Option:  ')))
    
    print('\n')
    return option

#---------------------
# Evaluation Inputs
#---------------------
str_file = r'TIIE_IRS_Data.xlsm'

wb = xw.Book(str_file)
parameters = wb.sheets('Pricing')
parameters.range('B1:G1').api.Calculate()


evaluation_date = pd.to_datetime(parameters.range('B1').value)
updateAll = parameters.range('B2').value
flag = updateAll
mxn_fx = parameters.range('F1').value
while type(mxn_fx) is not float:
    try:
        mxn_fx = float(parameters.range('F1').value)
    except:
        print('\nPlease check Cell "F1" of Pricing Sheet')
        c=input('When done press "c": ')
        if c == 'c':
            
            parameters = wb.sheets('Pricing')
            mxn_fx = parameters.range('F1').value
        else:
            continue
        
    
print('\nValuation Date: ', evaluation_date)

ql.Settings.instance().evaluationDate = ql.Date(evaluation_date.day, evaluation_date.month, evaluation_date.year)



#---------------------
#  Inputs dictionary
#---------------------

input_sheets = ['USD_OIS', 'USD_SOFR', 'USDMXN_XCCY_Basis', 'USDMXN_Fwds', 'MXN_TIIE']

dic_data = {}


for sheet in input_sheets:
    dic_data[sheet] = pd.read_excel(str_file, sheet)
    if sheet == 'USD_SOFR':
        dic_data = pr.futures_check(dic_data, str_file)
        
       



#-----------------------
#  Curves Bootstraping
#-----------------------
tenors = dic_data['MXN_TIIE']['Period'][1:].tolist()
rates = (dic_data['MXN_TIIE']['Quotes'][1:]/100).tolist()

g_crvs = pr.createCurves(dic_data, updateAll, flag)
flag = g_crvs[-1]
banxico_TIIE28 = pr.banxicoData(evaluation_date)
bo_engines = fn.bid_offer_crvs(dic_data, banxico_TIIE28, g_crvs[1], g_crvs[0])
g_engines = pr.engines(g_crvs[2], g_crvs[3], banxico_TIIE28)
dv01_engines = fn.flat_dv01_curves(dic_data, banxico_TIIE28, g_crvs[1], g_crvs[0])



    
print('Calculating trades...')
pr.tiie_pricing(dic_data, wb, g_crvs, banxico_TIIE28, bo_engines, g_engines, dv01_engines)
b= datetime.now()

pr.proc_CarryCalc(g_engines, wb)
dv01_tab = pd.DataFrame()

for k in range(0, len(tenors)):
    dv01_tab = pr.dv01_table(tenors[k], rates[k], evaluation_date, g_engines, dv01_engines, mxn_fx, dv01_tab)

dv01_tab.index = ['DV01']
wb.sheets['Notional_DV01'].range('J4').value = dv01_tab.T.values


print('\n',b-a)

print('\n----------------------------------------------------\n')
option = 0



while option != 7:   
    while True:
        try:
            option = menu()
            break
        except:
            print('\n###### Please write a number ######\n')
            continue
        
    if option == 1:
        # wb = xw.Book(str_file)
        
        parameters = wb.sheets('Pricing')
        parameters.range('B1:G1').api.Calculate()
        evaluation_date = parameters.range('B1:B1').value
        ql.Settings.instance().evaluationDate = ql.Date(evaluation_date.day, evaluation_date.month, evaluation_date.year)
        
        print('Calculating trades...')
        print('Valuation Date: ', evaluation_date)
        
        pr.tiie_pricing(dic_data, wb, g_crvs, banxico_TIIE28, bo_engines, g_engines, dv01_engines)
        
        
    
    if option == 2:
        a = datetime.now()
        dic_data = {}

        for sheet in input_sheets:
            dic_data[sheet] = pd.read_excel(str_file, sheet)
            if sheet == 'USD_SOFR':
                dic_data = pr.futures_check(dic_data, str_file)
            
        wb = xw.Book(str_file)
        parameters = wb.sheets('Pricing')
        parameters.range('B1:G1').api.Calculate()

        evaluation_date = parameters.range('B1:B1').value
        ql.Settings.instance().evaluationDate = ql.Date(evaluation_date.day, evaluation_date.month, evaluation_date.year)
        
        print('Calculating trades...')
        print('Valuation Date: ', evaluation_date)
        
        updateAll = parameters.range('B2').value
        mxn_fx = parameters.range('F1').value
        
        g_crvs = pr.createCurves(dic_data, updateAll, flag)
        banxico_TIIE28 = pr.banxicoData(evaluation_date)
        bo_engines = fn.bid_offer_crvs(dic_data, banxico_TIIE28, g_crvs[1], g_crvs[0])
        g_engines = pr.engines(g_crvs[2], g_crvs[3], banxico_TIIE28)
        dv01_engines = fn.flat_dv01_curves(dic_data, banxico_TIIE28, g_crvs[1], g_crvs[0])
        b=datetime.now()
        print('Calculating trades...')
        
        pr.tiie_pricing(dic_data, wb, g_crvs, banxico_TIIE28, bo_engines, g_engines, dv01_engines)
        
        pr.proc_CarryCalc(g_engines, wb)
        
        dv01_tab = pd.DataFrame()
        
        for k in range(0, len(tenors)):
            dv01_tab = pr.dv01_table(tenors[k], rates[k], evaluation_date, g_engines, dv01_engines, mxn_fx, dv01_tab)
        
        dv01_tab.index = ['DV01']
        wb.sheets['Notional_DV01'].range('J4').value = dv01_tab.T.values
        
        
        
        print('\n',b-a)
        
    if option == 3:
        wb = xw.Book(str_file)
        #rates_df = pr.smth_stp_rates(ql.Date().from_date(evaluation_date), wb, g_crvs)
        print('\nShort End Pricing...')
        pr.proc_ShortEndPricing(g_crvs[2], g_crvs[3], wb, banxico_TIIE28)
        print('\tTenor Fwd TIIE28 Done!')
        pr.proc_ShortEndPricing_byMPC(g_crvs[3], wb)
        print('\tMPC Date Fwd TIIE28 Done!')
    
    if option == 4:
        
        a = datetime.now()
        dic_data = {}

        for sheet in input_sheets:
            dic_data[sheet] = pd.read_excel(str_file, sheet)
            
        wb = xw.Book(str_file)
        parameters = wb.sheets('Pricing')
        
        updateAll = parameters.range('B2').value
        mxn_fx = parameters.range('F1').value
        
        g_crvs = pr.createCurves(dic_data, updateAll, flag)
        banxico_TIIE28 = pr.banxicoData(evaluation_date)
        bo_engines = fn.bid_offer_crvs(dic_data, banxico_TIIE28, g_crvs[1], g_crvs[0])
        g_engines = pr.engines(g_crvs[2], g_crvs[3], banxico_TIIE28)
        dv01_engines = fn.flat_dv01_curves(dic_data, banxico_TIIE28, g_crvs[1], g_crvs[0])
        b=datetime.now()
        parameters_trades = pr.collapse(wb)
        print('Calculating trades...')
        pr.tiie_pricing(dic_data, wb, g_crvs, banxico_TIIE28, bo_engines, g_engines, dv01_engines, parameters_trades)
        
        
        print('\n',b-a)
        
    
        
    if option==5:
        wb = xw.Book(str_file)
        parameters = wb.sheets('Pricing')
        parameters.range('B1:G1').api.Calculate()

        evaluation_date = parameters.range('B1:B1').value
        ql.Settings.instance().evaluationDate = ql.Date(evaluation_date.day, evaluation_date.month, evaluation_date.year)
        
        print('Valuation Date: ', evaluation_date)
        
        krrs, npv_group, dv01s = pr.tiie_blotter(dic_data, wb, g_crvs, banxico_TIIE28, bo_engines, g_engines, dv01_engines)
        

        print('Blotter done!')
        
    if option == 6:
        
        wb = xw.Book(str_file)
        parameters = wb.sheets('Pricing')
        parameters.range('B1:G1').api.Calculate()

        evaluation_date = parameters.range('B1:B1').value
        ql.Settings.instance().evaluationDate = ql.Date(evaluation_date.day, evaluation_date.month, evaluation_date.year)
        
        print('Valuation Date: ', evaluation_date)
        print('\nUploading Blotter...')
        pr.upload_blotter(wb, evaluation_date)
        print('Upload complete!')
        
        
    
    print('\n----------------------------------------------------\n')
    























print('Esteban y Gaby han sido coronados.\n')

print('⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣆⠀ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ \n'
'⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⢿⡿⡀⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⠀⠀⠀      \n'
'⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠙⣷⣾⠋⣀⡀ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀     \n'
'⠀⠀⠀⠀⡀⠀⠀⠀⢀⣠⣤⣌⡙⢷⣽⣧⡾⢋⣠⣤⣄⡀⠀⠀⠀⢀⠀⠀⠀⠀   \n'
'⠀⠀⠀⣀⣿⢄⡤⣰⡿⠛⠉⠙⠻⣾⣿⣿⣿⠟⠋⠉⠛⢿⣆⢤⡰⡿⣀⠀⠀⠀\n'
'⠀⣠⣤⣝⣷⣼⣷⣿⠁⠀⠀⢠⡶⢿⣿⣿⡷⢶⡄⠀⠀⠈⣿⣾⣧⣾⣫⣤⣄⠀  \n'
'⣼⡟⠉⣽⣿⣿⣿⡿⠛⣷⠀⠸⣧⣴⣿⣿⣶⣼⠇⠀⣾⠛⢿⣿⣿⣿⣯⠉⢻⣧\n'
'⣿⡀⠀⢿⣄⡘⢿⡇⣰⠟⢀⣴⠶⣦⣻⣿⣴⠶⣦⡀⠻⣆⢸⡿⢃⣠⡿⠀⠀⣿\n'
'⢹⣇⠀⠀⠉⠻⠞⣿⡛⠀⢿⡁⠀⠀⣿⣿⠁⠀⢈⡿⠀⢛⣿⠳⠟⠉⠀⠀⣸⡏  \n'
'⠀⠻⣦⡀⠀⠀⢀⣿⣷⣄⠈⠙⢛⣽⣿⣿⣿⡛⠛⠁⣠⣾⣿⡀⠀⠀⢀⣴⠟⠀ \n'
'⠀⠀⢹⣿⣶⣶⣾⣿⡿⠿⠿⠾⠿⠿⠿⠿⠿⠿⠷⠿⠿⢿⣿⣷⣶⣶⣿⡟⠀⠀\n'
'⠀⠀⠈⢩⣵⣶⣶⡇⣶⡌⢻⣿⣿⣏⠰⡷⢸⣿⣿⣿⢡⣶⢘⣶⣶⣾⡭⠁⠀⠀\n'
'⠀⠀⠀⠈⣛⣩⣭⣥⣤⣤⣭⣥⠤⠴⠤⠤⠴⠤⢬⣭⣤⣤⣬⣭⣍⣛⡃⠀⠀⠀ \n')
    

    