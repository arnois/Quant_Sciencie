# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17

@author: JArnulf QC (arnulf.q@gmail.com)

Module for TIIE Curve Bootsrapping
"""
###############################################################################
# MODULES
###############################################################################
import os
import QuantLib as ql
import pandas as pd
import numpy as np
#import requests
from datetime import date as dt
from datetime import timedelta
#import numpy as np
import sys
import warnings
warnings.simplefilter("ignore")
import xlwings as xw
###############################################################################
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
str_tradingfile = r'H:\Python\TIIE IRS Toolkit\TIIE_Trading.xlsx'
str_tradingfile = r'\\tlaloc\cuantitativa\Fixed Income\TIIE IRS Valuation Tool\Arnua'+'\\'+str_file
# Valuation date
dt_today = dt.today()
ql_dt_today = ql.Date(dt_today.day, dt_today.month, dt_today.year)
ql.Settings.instance().evaluationDate = ql_dt_today
ql_dt_yest = ql.Mexico().advance(ql_dt_today,-1,ql.Days)
dt_posswaps = dt(ql_dt_yest.year(), ql_dt_yest.month(), ql_dt_yest.dayOfMonth())
# UDM
import TIIE_CurveCreate_v3 as curveC
import udf_TIIE_PfolioMgmt as udf_pfl
# Portfolios
# Swaps File
str_dt_posswaps = dt_posswaps.strftime('%Y%m%d')
pswpspath = r'\\tlaloc\cuantitativa\Fixed Income\File Dump\PosSwaps\PosSwaps'
str_posswps_file = pswpspath+str_dt_posswaps+'.xlsx' # PosSwaps file 
df_tiieSwps = udf_pfl.setPfolio_tiieSwps(str_posswps_file) # Swaps' Portfolio
## Books
df1814 = df_tiieSwps[df_tiieSwps['BookID'] == 1814].reset_index(drop=True)
df8085 = df_tiieSwps[df_tiieSwps['BookID'] == 8085].reset_index(drop=True) 
df8089 = df_tiieSwps[df_tiieSwps['BookID'] == 8089].reset_index(drop=True) 
###############################################################################
# DATA
###############################################################################
# Inputs
#dic_data = curveC.udf.pull_data(str_file, dt_today)
#str_db = r'H:\Trading\marketmaking\TIIE\db_Curves_mkt.xlsx'
#dic_data = curveC.udf.pull_data2(str_file, dt_today, str_db)  
#fxrate = dic_data['USDMXN_XCCY_Basis']['Quotes'][0]
###############################################################################
# Portfolio Risk
###############################################################################
def bucketRisk_pfolio(str_tradingfile, dt_today, df1814):
    # Data
    dic_data = curveC.udf.pull_data(str_tradingfile, dt_today)
    fxrate = dic_data['USDMXN_XCCY_Basis']['Quotes'][0]
    # Book's Risk & Valuation
    dic_1814_valrisk = udf_pfl.get_pfolio_riskval(dic_data, df1814)
    # Display NPV in MXN
    print(f"\nAt {dt_today}\nNPV (MXN): "+\
          f"{dic_1814_valrisk['NPV_Book']*fxrate:,.0f}")
    # Bucket Risk (KRR)
    df1814_br = dic_1814_valrisk['DV01_Book']
    ## Display KRR
    print(fxrate)
    print(df1814_br)
    # compare with r'\\tlaloc\cuantitativa\Fixed Income\File Dump\Pickle Files\risk_20240327.pickle'
    # tmp = pd.read_pickle(r'\\tlaloc\cuantitativa\Fixed Income\File Dump\Pickle Files\risk_20240327.pickle')
    return df1814_br

def get_NPV_bRisk(str_file, dt_today, df1814):
    # Data
    dic_data = curveC.udf.pull_data(str_file, dt_today)
    fxrate = dic_data['USDMXN_XCCY_Basis']['Quotes'][0]
    # Book's Risk & Valuation
    dic_1814_valrisk = udf_pfl.get_pfolio_riskval(dic_data, df1814)
    # Merge data
    bookNPVRisk = df1814.\
        merge(dic_1814_valrisk['NPV_Swaps']*fxrate,
              left_index=True,right_index=True).\
            merge(dic_1814_valrisk['DV01_Swaps'],
                  left_index=True,right_index=True)
    return bookNPVRisk

def pfolioCF_OTR(str_file, dt_today, df1814):
    ql_dt_today = ql.Date(dt_today.day, dt_today.month, dt_today.year)
    tmpdata = curveC.udf.pull_data(str_file, dt_today)
    curves = udf_pfl.get_curves(tmpdata)
    crvTIIE = curves[3]
    ibor_tiie = curveC.udf.set_ibor_TIIE(crvTIIE)
    df_cfOTR = pd.DataFrame()
    
    for i,r in df1814.iterrows(): #i,r = list(df1814.iterrows())[0]
        start = ql.Date(r['StartDate'].day,r['StartDate'].month,r['StartDate'].year)
        maturity = ql.Date(r['Maturity'].day,r['Maturity'].month,r['Maturity'].year)
        notional = r['Notional']
        rate = r['FxdRate']
        rule = r['mtyOnHoliday']
        typ = r['SwpType']
        swp, swp_schdl = udf_pfl.tiieSwap(start, maturity, notional, 
                                          ibor_tiie, rate, typ, rule)
        swp_cf = udf_pfl.get_CF_tiieSwapOTR(swp, ibor_tiie, ql_dt_today)
        swp_cf['tradeID'] = r['TradeID']
        swp_cf['Notional'] = notional*typ # rec(-1) pays(-1) float rate
        swp_cf['accrTime'] = swp_cf['accDays']/360
        df_cfOTR = pd.concat([df_cfOTR, swp_cf])
    
    df_cfOTR = df_cfOTR.sort_values(by='date')
    return df_cfOTR.reset_index(drop=True)
    
def resetRisk_pfolio(str_file, dt_today, df1814, fxrate=18.10):
    # Pfolio CF Structure
    newpath = r'H:\Python\TIIE IRS Toolkit'+'\\'+ str_file
    df_cfOTR = pfolioCF_OTR(newpath, dt_today, df1814)
    # Banxico Meeting Structure
    path_meetings = r'H:\Python\TIIE IRS Toolkit\TIIE_Trading.xlsx'
    xlsheet = 'Banxico_Meeting_Dates'
    df_meetings = pd.read_excel(path_meetings, xlsheet, usecols='B')
    df_meetings['Eff_Meeting'] = df_meetings+pd.tseries.offsets.BusinessDay()
    reset_sens = []
    # ResetRisk
    for i,r in df_meetings.iterrows(): #i,r = list(df_meetings.iterrows())[0]
        df_CFMeeting = df_cfOTR[df_cfOTR['fixingDate'] == r['Eff_Meeting']]
        net_not_exp = df_CFMeeting['Notional'].sum()
        mean_accT = df_CFMeeting['accrTime'].mean()
        reset_sens.append(net_not_exp*mean_accT*0.0001/fxrate)
    
    df_meetings['Sens'] = reset_sens
    return df_meetings

# Function to evaluate bucket risk in future date # dt_risk = dt(2024,6,28)
def projBucketRisk(str_tradingfile, dt_risk, df1814):
    """
    Bucket risk projected into a future date, assuming last set of inputs.
    """
    # Future date
    ql_eval_dt = ql.Date(dt_risk.day, dt_risk.month, dt_risk.year)
    ql.Settings.instance().evaluationDate = ql_eval_dt
    
    # Data
    dic_data = curveC.udf.pull_data(str_file, dt_today)
    fxrate = dic_data['USDMXN_XCCY_Basis']['Quotes'][0]
    
    # Book's Risk & Valuation
    dic_1814_valrisk = udf_pfl.get_pfolio_riskval(dic_data, df1814)
    
    # Display NPV in MXN
    print(f"\nAt {dt_risk}\nNPV (MXN): "+\
          f"{dic_1814_valrisk['NPV_Book']*fxrate:,.0f}")
    # Bucket Risk (KRR)
    df1814_br = dic_1814_valrisk['DV01_Book']
    
    ## Display KRR
    print(fxrate)
    print(df1814_br)
    return df1814_br

###############################################################################
# Portfolio PnL
###############################################################################
def get_dayblotter(str_path, lst_blttrcols, tiie_sheet = 'BlotterTIIE_auto'):
    blotter_merdin = str_path
    df = pd.read_excel(blotter_merdin, sheet_name=tiie_sheet, 
                       skiprows = 2)
    dayblotter = df.loc[~df.Book.isna(), lst_blttrcols]
    dayblotter[['Fecha Inicio', 'Fecha vencimiento']] = \
        dayblotter[['Fecha Inicio', 'Fecha vencimiento']].\
        astype('datetime64[ns]')
    return dayblotter
    
def proc_dayBlotter(df1814, dt_val_tdy, str_path, lst_blttrcols, bookID = 1814):
    # str_path = r'C:\Users\jquintero\Downloads\Blotter Mesa de Dinero 3.xlsm'
    # New Trades
    dayblotter = get_dayblotter(str_path, lst_blttrcols, 'BlotterTIIE_auto')
    if not dayblotter.empty:
        tenor2ql = {'m': 4, 'y': 52}
        period = dayblotter['Tenor'].str[-1].map(tenor2ql).to_numpy()
        tenor = dayblotter['Tenor'].str[:-1].astype(int).to_numpy()
        weeks = tenor*period
        dayblotter['L'] = weeks
        lst_NAtypes = [pd._libs.tslibs.nattype.NaTType,
                       np.nan]
        for i,r in dayblotter.iterrows():
            if type(r['Fecha Inicio']) in lst_NAtypes:
                idt = ql.Mexico().advance(ql.Date(dt_val_tdy.day,
                                                  dt_val_tdy.month,
                                                  dt_val_tdy.year),
                                          ql.Period(1,ql.Days))
                fdt = idt + ql.Period(r['L'],ql.Weeks)
                dayblotter.loc[i,'Fecha Inicio'] = dt(idt.year(), 
                                                      idt.month(), 
                                                      idt.dayOfMonth())
                dayblotter.loc[i,'Fecha vencimiento']  = dt(fdt.year(), 
                                                      fdt.month(), 
                                                      fdt.dayOfMonth())
        dayblotter['Cuota compensatoria / unwind'] = \
            dayblotter['Cuota compensatoria / unwind'].fillna(0)
        # DayBlotter Swaps Book
        dayblotter['BookID'] = dayblotter['Book'].astype(int)
        dayblotter['TradeID'] = dayblotter['Book'].astype(int) + dayblotter.index+1
        dayblotter['TradeDate'] = dt_val_tdy
        dayblotter['Notional'] = abs(dayblotter['Size'])*1e6
        dayblotter['StartDate'] = dayblotter['Fecha Inicio']
        dayblotter['Maturity'] = dayblotter['Fecha vencimiento']
        dayblotter['CpDate'] = dayblotter['Fecha Inicio'] + timedelta(days=28)
        dayblotter['FxdRate'] = dayblotter['Yield(Spot)']
        dayblotter['SwpType'] = np.sign(dayblotter['Size']).astype(int)*-1
        dayblotter['mtyOnHoliday'] = 0
        
        df1814_bttr = dayblotter[dayblotter['BookID'] == bookID].\
            reset_index(drop=True)[df1814.columns.to_list()]
        df1814_tdy = df1814.append(df1814_bttr)
    else:
        df1814_tdy = df1814

    return df1814_tdy

def proc_PnL_pfolio(dfbook, dt_val_yst, dt_val_tdy, 
                    blotter_merdin, lst_blttrcols, 
                    dic_data_yst, dic_data_tdy,
                    str_file, bookID = 1814):
    dayblotter = get_dayblotter(blotter_merdin, lst_blttrcols, 
                                'BlotterTIIE_auto')
    dfbook_tdy = proc_dayBlotter(dfbook, blotter_merdin, 
                                 lst_blttrcols, bookID = bookID)
    try:
        fees = dayblotter[
            dayblotter['BookID'] == bookID
            ]['Cuota compensatoria / unwind'].sum()
    except:
        try:
            fees = dayblotter[
                dayblotter['Book'] == bookID
                ]['Cuota compensatoria / unwind'].sum()
        except:
            fees = 0

    delta_npv, cf = udf_pfl.get_pfolio_PnL(str_file, dt_val_yst, dt_val_tdy, 
                                           dfbook, dfbook_tdy)
    pnl = delta_npv + cf + fees
    print(f'{bookID}\nFrom {str(dt_val_yst)} to {str(dt_val_tdy)}'+\
          f'\nPnL: {pnl[0]:,.0f}\n\tNPV Chg: {delta_npv:,.0f}'+\
          f'\n\tCF: {cf[0]:,.0f}'+\
              f'\n\tFees: {fees:,.0f}')

def proc_mainPnL(dt_val_yst, dt_val_tdy):
    # New Trades
    blotter_merdin = r'C:\Users\jquintero\Downloads\Blotter Mesa de Dinero 3.xlsm'
    lst_blttrcols = ['Book','Tenor','Yield(Spot)','Size', 'Fecha Inicio', 
                     'Fecha vencimiento','Cuota compensatoria / unwind']
    # Data
    dic_data_yst = curveC.udf.pull_data(str_file, dt_val_yst)
    dic_data_tdy = curveC.udf.pull_data(str_file, dt_val_tdy)
    # PnL
    proc_PnL_pfolio(df1814, dt_val_yst, dt_val_tdy, blotter_merdin, 
                        lst_blttrcols, dic_data_yst, dic_data_tdy,
                        str_file, bookID = 1814)
    proc_PnL_pfolio(df8085, dt_val_yst, dt_val_tdy, blotter_merdin, 
                        lst_blttrcols, dic_data_yst, dic_data_tdy,
                        str_file, bookID = 8085)
    
# Function to update CME marks
def fn_update_dbCME(xlpath, dt_val_tdy, dbmarkspath) -> None:
    xl_db_cme = xw.Book(xlpath)
    xl_last_date = xl_db_cme.sheets('db').range('A1').end('right').value.date()
    # Check for dates in database
    if xl_last_date >= dt_val_tdy:
        # CME marks
        cme_marks = pd.read_csv(dbmarkspath, header=None)
        # Last column to paste marks
        tmprn = xl_db_cme.sheets('db').range('A1').end('right').column-2
        xl_db_cme.sheets('db').range('B1').offset(1, tmprn).value = \
            [[x] for x in cme_marks.iloc[:,1].tolist()]
        # Update todays marks
        if xl_last_date != dt_val_tdy:
            # First row data
            frd = xl_db_cme.sheets('db').range('B1:'+xl_db_cme.sheets('db').\
                    range('A1').offset(0, tmprn).address.replace('$','')).value
            tmpidx = [x.date() for x in frd].index(dt_val_tdy) + 1
            # Actual date to update marks on
            xl_db_cme.sheets('db').range('A1').offset(1, tmpidx).value = \
                [[x] for x in cme_marks.iloc[:,1].tolist()]
        
    # Close and save db
    tmpvar = input('Done. ')
    tmppid = xl_db_cme.app.pid
    xl_db_cme.save()
    xw.apps[tmppid].api.Quit()
    return None
    
###############################################################################
# PnL Run
# if __name__ != '__main__':
#     ###########################################################################
#     # PnL Run
#     ql_yest = ql.Mexico().advance(ql_dt_today,-1,ql.Days)
#     dt_yest = dt(ql_yest.year(), ql_yest.month(), ql_yest.dayOfMonth())
#     dt_val_yst, dt_val_tdy = dt_yest, dt_today
#     proc_mainPnL(dt_val_yst, dt_val_tdy)
    
###############################################################################
# PnL Run By Trade
if __name__ == '__pnl_detailed__':
    # Valuation Dates
    ql_yest = ql.Mexico().advance(ql_dt_today,-1,ql.Days)
    dt_yest = dt(ql_yest.year(), ql_yest.month(), ql_yest.dayOfMonth())
    dt_val_yst, dt_val_tdy = dt_yest, dt_today
    # Today's Trading Blotter
    #blotter_merdin = r'E:\Blotters\231129.xlsx'
    path_blotter = r'\\tlaloc\cuantitativa\Fixed Income\File Dump\Blotters\TIIE'
    blotter_merdin = path_blotter + rf'\{dt_val_tdy.strftime("%y%m%d")}.xlsx'
    lst_blttrcols = ['Book','Tenor','Yield(Spot)','Size', 'Fecha Inicio', 
                     'Fecha vencimiento','Cuota compensatoria / unwind']
    dayblotter = get_dayblotter(blotter_merdin, lst_blttrcols, 'BlotterTIIE_auto')
    # Today's Book
    df1814_tdy = proc_dayBlotter(df1814, dt_val_tdy, blotter_merdin, 
                                 lst_blttrcols, bookID = 1814)
    df8085_tdy = proc_dayBlotter(df8085, dt_val_tdy, blotter_merdin, 
                                 lst_blttrcols, bookID = 8085)
    # Yesterday's Book Valuation
    dic_data_yst = curveC.udf.pull_data(str_file, dt_val_yst)
    dfval_1814_yst = udf_pfl.get_pflio_npv_atDate(dt_val_yst, dic_data_yst, df1814)
    dfval_8085_yst = udf_pfl.get_pflio_npv_atDate(dt_val_yst, dic_data_yst, df8085)
    
    # Verify new closings availability
    tmppath = r'\\tlaloc\cuantitativa\Fixed Income\File Dump\Historical OIS TIIE'
    tmpfname = rf'\IRS_MXN_CURVE_{dt_val_tdy.strftime("%Y%m%d")}.csv'
    contInLoop = True
    while contInLoop:
        tmpFileCond = os.path.isfile(tmppath+tmpfname) and os.path.getsize(tmppath+tmpfname) > 0
        if tmpFileCond:
            print('\nCME valuation marks available!')
            xlpath = r'\\tlaloc\cuantitativa\Fixed Income\File Dump\Database\db_cme.xlsx'
            fn_update_dbCME(xlpath, dt_val_tdy, tmppath+tmpfname)
            contInLoop = False
        else:
            print('\nCloses still not available')
            tmpinput = input(('Continue? (y/n): '))
            if tmpinput[0] == 'y':
                contInLoop = False
        
    # Today's Book Valuation
    dic_data_tdy = curveC.udf.pull_data(str_file, dt_val_tdy)
    dfval_1814_tdy = udf_pfl.get_pflio_npv_atDate(dt_val_tdy, dic_data_tdy, df1814_tdy)
    dfval_8085_tdy = udf_pfl.get_pflio_npv_atDate(dt_val_tdy, dic_data_tdy, df8085_tdy)
    
    # Today's Book CF
    crvUSDOIS, crvUSDSOFR, crvMXNOIS, crvTIIE = udf_pfl.get_curves(dic_data_tdy)
    ibor_tiie_cf = curveC.udf.set_ibor_TIIE(crvTIIE)
    #ibor_tiie_cf = udf_pfl.udf.set_ibor_TIIE(
    #        ql.FlatForward(0, ql.Mexico(0), 0.1150, ql.Actual360()), 
    #        )
    df1814_cf_tdy = udf_pfl.get_pfolio_CF_atDate(dt_val_tdy, dfval_1814_yst, ibor_tiie_cf)
    df8085_cf_tdy = udf_pfl.get_pfolio_CF_atDate(dt_val_tdy, dfval_8085_yst, ibor_tiie_cf)
    cf_1814 =  df1814_cf_tdy.sum()
    cf_8085 =  df8085_cf_tdy.sum()
    
    # Today's fees
    fees1814 = dayblotter[
        dayblotter['Book'] == 1814
        ]['Cuota compensatoria / unwind'].sum()
    fees8085 = dayblotter[
        dayblotter['Book'] == 8085
        ]['Cuota compensatoria / unwind'].sum()
    
    # DeltaNPV
    deltaNPV_1814 = dfval_1814_tdy['NPV'].sum() - dfval_1814_yst['NPV'].sum()
    deltaNPV_8085 = dfval_8085_tdy['NPV'].sum() - dfval_8085_yst['NPV'].sum()
    # CFs
    totalCF_1814 = cf_1814.values[0] #+ fees1814
    totalCF_8085 = cf_8085.values[0] #+ fees8085
    
    # PnL
    print('PnL')
    print(f'1814: {deltaNPV_1814+totalCF_1814+fees1814:,.0f}')
    print(f'\t\tCap Gain: {deltaNPV_1814:,.0f}')
    print(f'\t\tCF: {totalCF_1814:,.0f}\n\t\tFees: {fees1814:,.0f}')
    print(f'8085: {deltaNPV_8085+totalCF_8085+fees8085:,.0f}')
    print(f'\t\tCap Gain: {deltaNPV_8085:,.0f}')
    print(f'\t\tCF: {totalCF_8085:,.0f}\n\t\tFees: {fees8085:,.0f}')
    

if __name__ != '__main__':
    ###########################################################################
    # Valuation Dates
    dt_val_yst, dt_val_tdy = dt(2023,5,2), dt(2023,5,3)
    # New Trades
    blotter_merdin = r'C:\Users\jquintero\Downloads\Blotter Mesa de Dinero 3.xlsm'
    lst_blttrcols = ['Book','Tenor','Yield(Spot)','Size', 'Fecha Inicio', 
                     'Fecha vencimiento','Cuota compensatoria / unwind']
    dayblotter = get_dayblotter(blotter_merdin, lst_blttrcols, 'BlotterTIIE_auto')
    df1814_tdy = proc_dayBlotter(df1814, blotter_merdin, 
                                 lst_blttrcols, bookID = 1814)
    df8085_tdy = proc_dayBlotter(df8085, blotter_merdin, 
                                 lst_blttrcols, bookID = 8085)
    # PnL
    #str_db = r'H:\Trading\marketmaking\TIIE\db_Curves_mkt.xlsx'
    dic_data_yst = curveC.udf.pull_data(str_file, dt_val_yst)
    dic_data_tdy = curveC.udf.pull_data(str_file, dt_val_tdy)
    inpv = udf_pfl.get_pflio_npv_atDate(dt_val_yst, dic_data_yst, df1814)
    fnpv = udf_pfl.get_pflio_npv_atDate(dt_val_tdy, dic_data_tdy, df1814_tdy)
    data_tiie = pd.concat([dic_data_yst['MXN_TIIE'][['Tenor','Quotes']], 
                           dic_data_tdy['MXN_TIIE'][['Quotes']]], axis=1)
    try:
        fees = dayblotter[
            dayblotter['BookID'] == 1814]['Cuota compensatoria / unwind'].sum()
    except:
        fees = 0
    
    delta_npv, cf = udf_pfl.get_pfolio_PnL(str_file, dt_val_yst, dt_val_tdy, 
                                           df1814, df1814_tdy)
    pnl = delta_npv + cf + fees
    print(f'1814\nFrom {str(dt_val_yst)} to {str(dt_val_tdy)}'+\
          f'\nPnL: {pnl[0]:,.0f}\n\tNPV Chg: {delta_npv:,.0f}'+\
          f'\n\tCF: {cf[0]:,.0f}'+\
              f'\n\tFees: {fees:,.0f}')
    try:
        fees = dayblotter[
            dayblotter['BookID'] == 8085]['Cuota compensatoria / unwind'].sum()
    except:
        fees = 0
        
    delta_npv, cf = udf_pfl.get_pfolio_PnL(str_file, dt_val_yst, dt_val_tdy, 
                                           df8085, df8085_tdy)
    pnl = delta_npv + cf + fees
    print(f'8085\nFrom {str(dt_val_yst)} to {str(dt_val_tdy)}'+\
          f'\nPnL: {pnl[0]:,.0f}\n\tNPV Chg: {delta_npv:,.0f}'+\
          f'\n\tCF: {cf[0]:,.0f}'+\
              f'\n\tFees: {fees:,.0f}')
    ###########################################################################











