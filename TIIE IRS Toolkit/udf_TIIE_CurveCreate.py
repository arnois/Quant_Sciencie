# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17

@author: JArnulf QC (arnulf.q@gmail.com)

User-defined functions module for TIIE Curve Bootsrapping
"""
###############################################################################
# Modules
###############################################################################
import QuantLib as ql
import numpy as np
import pandas as pd
import requests
from datetime import timedelta, date
###############################################################################
# Global Variables
###############################################################################
tenor2ql = {'B':ql.Days,'D':ql.Days,'M':ql.Months,'W':ql.Weeks,'Y':ql.Years}
###############################################################################
# Data Mgmt
###############################################################################
# DATA IMPORT
def import_data(str_file):
    tmpxls = pd.ExcelFile(str_file)
    dic_data = {}
    for sheet in tmpxls.sheet_names:
        dic_data[sheet] = pd.read_excel(str_file, sheet)
    tmpxls.close()

    return dic_data

# DATA UPDATE
def pull_data(str_file, dt_today):
    dic_data = import_data(str_file)
    db_cme = pd.read_excel(r'E:\db_cme' + r'.xlsx', 'db').set_index('TENOR')
    db_cme.columns = db_cme.columns.astype(str)
    db_crvs = pd.read_excel(r'E:\db_Curves_mkt' + r'.xlsx', 
                            'bgnPull', 
                            skiprows=3).drop([0,1]).\
        reset_index(drop=True).set_index('Date')
    # # USD Curves Data
    datakeys = ['USD_OIS', 'USD_SOFR','USD_LIBOR_3M', 'USD_LIBOR_3Mvs1M_Basis']
    for mktCrv in datakeys:
        dic_data[mktCrv]['Quotes'] = \
            db_crvs.loc[str(dt_today), dic_data[mktCrv]['Tickers']].\
                fillna(method="ffill").values
    # # MXN Curves Data
    cmenames_mxnfwds = ['FX.USD.MXN.ON', 'FX.USD.MXN.1W', 'FX.USD.MXN.1M', 
                        'FX.USD.MXN.2M', 'FX.USD.MXN.3M', 'FX.USD.MXN.6M', 
                        'FX.USD.MXN.9M', 'FX.USD.MXN.1Y']
    dic_data['USDMXN_Fwds']['Quotes'] = \
        db_cme.loc[cmenames_mxnfwds, str(dt_today)+' 00:00:00'].values
        
    dic_data['USDMXN_XCCY_Basis']['Quotes'][0] = \
        db_cme.loc['FX.USD.MXN', str(dt_today)+' 00:00:00']
    
    dic_data['USDMXN_XCCY_Basis']['Quotes'][-9:] = \
        db_cme.iloc[-9:, 
                    np.where(
                        db_cme.columns == str(dt_today)+' 00:00:00')[0][0]].\
            values*100
    
    dic_data['MXN_TIIE']['Quotes'] = \
        db_cme.iloc[:14, 
                    np.where(
                        db_cme.columns == str(dt_today)+' 00:00:00')[0][0]].\
                        values
            
    return dic_data

def pull_data2(str_file, dt_today, str_db):
    dic_data = import_data(str_file)
    db_cme = pd.read_excel(str_db, 'cme').set_index('TENOR')
    db_cme.columns = db_cme.columns.astype(str)
    db_crvs = pd.read_excel(str_db, 'bgnPull', skiprows=3).drop([0,1]).\
        reset_index(drop=True).set_index('Date')
    # # USD Curves Data
    datakeys = ['USD_OIS', 'USD_SOFR','USD_LIBOR_3M', 'USD_LIBOR_3Mvs1M_Basis']
    for mktCrv in datakeys:
        dic_data[mktCrv]['Quotes'] = \
            db_crvs.loc[str(dt_today), dic_data[mktCrv]['Tickers']].\
                fillna(method="ffill").values
    # # MXN Curves Data
    cmenames_mxnfwds = ['FX.USD.MXN.ON', 'FX.USD.MXN.1W', 'FX.USD.MXN.1M', 
                        'FX.USD.MXN.2M', 'FX.USD.MXN.3M', 'FX.USD.MXN.6M', 
                        'FX.USD.MXN.9M', 'FX.USD.MXN.1Y']
    dic_data['USDMXN_Fwds']['Quotes'] = \
        db_cme.loc[cmenames_mxnfwds, str(dt_today)+' 00:00:00'].values
        
    dic_data['USDMXN_XCCY_Basis']['Quotes'][0] = \
        db_cme.loc['FX.USD.MXN', str(dt_today)+' 00:00:00']
    
    dic_data['USDMXN_XCCY_Basis']['Quotes'][-9:] = \
        db_cme.iloc[-9:, 
                    np.where(
                        db_cme.columns == str(dt_today)+' 00:00:00')[0][0]].\
            values*100
    
    dic_data['MXN_TIIE']['Quotes'] = \
        db_cme.iloc[:14, 
                    np.where(
                        db_cme.columns == str(dt_today)+' 00:00:00')[0][0]].\
                        values
            
    return dic_data
###############################################################################
# Curve Helpers
###############################################################################
# QuantLib's Helper object for USDOIS Crv Bootstrapping
def qlHelper_USDOIS(df):
    # market calendar
    calendar = ql.UnitedStates(0)
    # input data
    tenor = df['Tenors'].str[-1].map(tenor2ql).to_list()
    period = df['Period'].astype(int).to_list()
    data = (df['Quotes']/100).tolist()
    # Deposit rates
    deposits = {(period[0], tenor[0]): data[0]}
    # Swap rates
    n = len(period)
    swaps = {}
    for i in range(1,n):
        swaps[(period[i], tenor[i])] = data[i]
    # Rate Qauntlib.Quote objects
    ## desposits
    for n, unit in deposits.keys():
        deposits[(n,unit)] = ql.SimpleQuote(deposits[(n,unit)])
    ## swap rates
    for n,unit in swaps.keys():
        swaps[(n,unit)] = ql.SimpleQuote(swaps[(n,unit)])
        
    # Rate helpers deposits
    dayCounter = ql.Actual360()
    settlementDays = 2
    ## deposits
    depositHelpers = [ql.DepositRateHelper(
        ql.QuoteHandle(deposits[(n, unit)]),
        ql.Period(n, unit), 
        settlementDays,
        calendar, 
        ql.ModifiedFollowing, 
        False, 
        dayCounter
        ) 
        for n, unit in deposits.keys()
    ]
    ## swap rates
    OIS_Index = ql.FedFunds()
    OISHelpers = [ql.OISRateHelper(
        settlementDays, ql.Period(n, unit),
        ql.QuoteHandle(swaps[(n,unit)]),
        OIS_Index
        ) 
        for n, unit in swaps.keys()
    ]
    ## helpers merge
    helpers = depositHelpers + OISHelpers
    return(helpers)

# QuantLib's Helper object for USD3M Crv Bootstrapping
def qlHelper_USD3M(dic_df, discount_curve):
    # market calendar
    calendar = ql.UnitedStates(0)
    # settlement date
    dt_settlement = ql.UnitedStates(0).advance(
        ql.Settings.instance().evaluationDate,ql.Period('2D'))
    # non-futures idx
    df = dic_df['USD_LIBOR_3M']
    idx_nonfut = (df['Types'] != 'FUT')
    # input data
    tenor = df['Tenors'][idx_nonfut].str[-1].map(tenor2ql).to_list()
    period = df['Period'][idx_nonfut].astype(int).to_list()
    data_nonfut = (df['Quotes'][idx_nonfut]/100).tolist()
    data_fut = (df['Quotes'][~idx_nonfut]/100).tolist()
    # IborIndex
    swapIndex = ql.USDLibor(ql.Period(3, ql.Months))

    # Deposit rates
    deposits = {(period[0], tenor[0]): data_nonfut[0]}
    # Futures rates
    n_fut = len(data_fut)
    imm = ql.IMM.nextDate(dt_settlement)
    imm = dt_settlement
    futures = {}
    for i in range(n_fut):
        imm = ql.IMM.nextDate(imm)
        futures[imm] = 100 - data_fut[i]*100  
    # Swap rates
    n = len(period)
    swaps = {}
    for i in range(1,n):
        swaps[(period[i], tenor[i])] = data_nonfut[i]
        
    # Rate Qauntlib.Quote objects
    ## desposits
    for n, unit in deposits.keys():
        deposits[(n,unit)] = ql.SimpleQuote(deposits[(n,unit)])
    ## futures
    for d in futures.keys():
        futures[d] = futures[d]
    ## swap rates
    for n,unit in swaps.keys():
        swaps[(n,unit)] = ql.SimpleQuote(swaps[(n,unit)])
        
    # Rate helpers deposits
    dayCounter = ql.Actual360()
    settlementDays = 2
    ## deposits
    depositHelpers = [ql.DepositRateHelper(
        ql.QuoteHandle(deposits[(n, unit)]),
        ql.Period(n, unit), 
        settlementDays,
        calendar, 
        ql.ModifiedFollowing, 
        False, 
        dayCounter
        ) 
        for n, unit in deposits.keys()
    ]
    ## futures
    months = 3
    futuresHelpers = [ql.FuturesRateHelper(
        ql.QuoteHandle(ql.SimpleQuote(futures[d])), 
        d, months, calendar, 
        ql.ModifiedFollowing, True, dayCounter
        ) 
        for d in futures.keys()
    ]
    
    ## swap rates
    fixedLegFrequency = ql.Semiannual
    fixedLegAdjustment = ql.ModifiedFollowing
    fixedLegDayCounter = ql.Thirty360()
    
    ## swaphelper
    swapHelpers = [ql.SwapRateHelper(
        ql.QuoteHandle(swaps[(n,unit)]),
        ql.Period(n, unit), 
        calendar,
        fixedLegFrequency, 
        fixedLegAdjustment,
        fixedLegDayCounter, 
        swapIndex, 
        ql.QuoteHandle(), 
        ql.Period(0, ql.Days),
        discount_curve
        )
        for n, unit in swaps.keys()
    ]

    ## helpers merge
    helpers = depositHelpers + futuresHelpers + swapHelpers

    return(helpers)

# QuantLib's Helper object for USD1M Crv Bootstrapping
def qlHelper_USD1M(dic_df, crv_USD3M):
    # market calendar
    calendar = ql.UnitedStates(0)
    
    # dat
    df = dic_df['USD_LIBOR_3Mvs1M_Basis']
    # input data
    tenor = df['Tenor'].str[-1].map(tenor2ql).to_list()
    period = df['Period'].astype(int).to_list()
    data = (df['Quotes']/100).tolist()
    
    # Deposit rates
    deposits = {(period[0], tenor[0]): data[0]}
    # Basis rates
    n = len(period)
    basis = {}
    for i in range(1,n):
        basis[(period[i], tenor[i])] = data[i]/100
        
    # Rate Qauntlib.Quote objects
    for n,unit in deposits.keys():
        deposits[(n,unit)] = ql.SimpleQuote(deposits[(n,unit)])
    for n,unit in basis.keys():
        basis[(n,unit)] = ql.SimpleQuote(basis[(n,unit)])

    # Deposit rate helpers
    dayCounter = ql.Actual360()
    settlementDays = 2
    depositHelpers = [ql.DepositRateHelper(
        ql.QuoteHandle(deposits[(n, unit)]),
        ql.Period(n, unit), 
        settlementDays,
        calendar, 
        ql.ModifiedFollowing,
        False, 
        dayCounter
        )
        for n, unit in deposits.keys()
    ]
    
    # Basis helper
    crv_handle = ql.RelinkableYieldTermStructureHandle()
    baseCurrencyIndex = ql.USDLibor(ql.Period('3M'), crv_USD3M)
    quoteCurrencyIndex = ql.USDLibor(ql.Period('1M'), crv_handle)
    crv_ff = ql.FlatForward(0, calendar, 0.01, ql.Actual360())
    crv_handle.linkTo(crv_ff)
    isFxBaseCurrencyCollateralCurrency = False
    isBasisOnFxBaseCurrencyLeg = False
    basis_helper = [ql.CrossCurrencyBasisSwapRateHelper(
        ql.QuoteHandle(basis[(n, unit)]), 
        ql.Period(n, unit), 
        settlementDays, 
        calendar, 
        ql.ModifiedFollowing, 
        False,
        baseCurrencyIndex, 
        quoteCurrencyIndex, 
        crv_handle,
        isFxBaseCurrencyCollateralCurrency, 
        isBasisOnFxBaseCurrencyLeg
        )
        for n, unit in basis.keys()
    ]
    ## helpers merge
    helpers = depositHelpers + basis_helper

    return(helpers)

# QuantLib's Helper object for USDSOFR Crv Bootstrapping 
def qlHelper_SOFR(dic_df, discount_curve):
    # market calendar
    calendar = ql.UnitedStates(0)
    # settlement date
    dt_settlement = calendar.advance(
            ql.Settings.instance().evaluationDate, ql.Period('2D'))
    # non-futures idx
    df = dic_df['USD_SOFR']
    idx_nonfut = (df['Types'] != 'FUT')
    # input data
    tenor = df['Tenors'][idx_nonfut].str[-1].map(tenor2ql).to_list()
    period = df['Period'][idx_nonfut].astype(int).to_list()
    data_nonfut = (df['Quotes'][idx_nonfut]/100).tolist()
    data_fut = (df['Quotes'][~idx_nonfut]/100).tolist()
    
    # IborIndex
    swapIndex = ql.Sofr()

    # Deposit rates
    deposits = {(period[0], tenor[0]): data_nonfut[0]}
   
    # Futures rates
    n_fut = len(data_fut)
    imm = ql.IMM.nextDate(dt_settlement)
    imm = dt_settlement
    futures = {}
    for i in range(n_fut):
        imm = ql.IMM.nextDate(imm)
        futures[imm] = 100 - data_fut[i]*100  
    
    # Swap rates
    n = len(period)
    swaps = {}
    for i in range(1,n):
        swaps[(period[i], tenor[i])] = data_nonfut[i]
        
    # Rate Qauntlib.Quote objects
    ## desposits
    
    for n, unit in deposits.keys():
        deposits[(n,unit)] = ql.SimpleQuote(deposits[(n,unit)])
    ## futures
    for d in futures.keys():
        futures[d] = futures[d]
    ## swap rates
    for n,unit in swaps.keys():
        swaps[(n,unit)] = ql.SimpleQuote(swaps[(n,unit)])
        
    # Rate helpers deposits
    dayCounter = ql.Actual360()
    settlementDays = 2
    ## deposits
    depositHelpers = [ql.DepositRateHelper(
        ql.QuoteHandle(deposits[(n, unit)]),
        ql.Period(n, unit), 
        settlementDays,
        calendar, 
        ql.ModifiedFollowing, 
        False, 
        dayCounter
        ) 
        for n, unit in deposits.keys()
    ]
    ## futures
    months = 3
    futuresHelpers = [ql.FuturesRateHelper(
        ql.QuoteHandle(ql.SimpleQuote(futures[d])), 
        d, months, calendar, 
        ql.ModifiedFollowing, True, dayCounter
        ) 
        for d in futures.keys()
    ]
    
    ## swap rates
    fixedLegFrequency = ql.Annual
    fixedLegAdjustment = ql.ModifiedFollowing
    fixedLegDayCounter = ql.Actual360()
    ## swaphelper
    swapHelpers = [ql.SwapRateHelper(
        ql.QuoteHandle(swaps[(n,unit)]),
        ql.Period(n, unit), 
        calendar,
        fixedLegFrequency, 
        fixedLegAdjustment,
        fixedLegDayCounter, 
        swapIndex, 
        ql.QuoteHandle(), 
        ql.Period(2, ql.Days),
        discount_curve
        )
        for n, unit in swaps.keys()
    ]

    ## helpers merge
    helpers = depositHelpers + futuresHelpers + swapHelpers

    return(helpers)

# QuantLib's Helper object for Implied MXNOIS Crv Bootstrapping
def qlHelper_MXNOIS(dic_df, discount_curve, crv_usdswp, crvType = 'SOFR'):
    # Calendars
    #calendar_us = ql.UnitedStates(0)
    calendar_mx = ql.Mexico(0)

    # Handle dat
    spotfx = dic_df['USDMXN_XCCY_Basis']['Quotes'][0]
    df_basis = dic_df['USDMXN_XCCY_Basis']
    df_tiie = dic_df['MXN_TIIE']
    df_fwds = dic_df['USDMXN_Fwds']
    # Handle idxs
    str_tenors_fwds = ['%-1B', '%1W', '%1M', '%2M', '%3M','%6M', '%9M', '%1Y'] # ['%3M','%6M', '%9M', '%1Y']
    idx_fwds = np.where(np.isin(df_fwds['Tenor'],
                                str_tenors_fwds))[0].tolist()
    lst_tiieT = ['%1L', '%26L', '%39L', '%52L', '%65L', 
                 '%91L', '%130L', '%195L', '%260L', '%390L']
    idx_tiie = np.where(np.isin(df_tiie['Tenor'],
                     lst_tiieT))[0].tolist()
    # Input data
    tenor = ql.EveryFourthWeek
    basis_period = df_basis['Period'].astype(int).tolist()
    tiie_period = df_tiie['Period'][idx_tiie].astype(int).to_list()
    fwds_period = df_fwds['Period'][idx_fwds].astype(int).to_list()
    fwds_period[-1] = 1
    data_tiie = (df_tiie['Quotes'][idx_tiie]/100).tolist()
    data_fwds = (df_fwds['Quotes'][idx_fwds]/10000).tolist()
    if crvType == 'SOFR':
        data_basis = (-1*df_basis['Quotes']/10000).tolist()
    else:
        data_basis = (df_basis['Quotes']/10000).tolist()
    
    # Basis swaps
    basis_usdmxn = {}
    n_basis = len(basis_period)
    for i in range(1,n_basis):
        basis_usdmxn[(basis_period[i], tenor)] = data_basis[i]

    # Forward Points
    fwds_tenors = [tenor2ql[t[-1]] for t in str_tenors_fwds]
    fwdpts = {}
    n_fwds = len(fwds_period)
    for i in range(n_fwds):
        fwdpts[(fwds_period[i], fwds_tenors[i])] = data_fwds[i]

    # Deposit rates
    deposits = {(tiie_period[0], tenor): data_tiie[0]}
    
    # TIIE Swap rates]
    swaps_tiie = {}
    n_tiie = len(tiie_period)
    for i in range(1,n_tiie):
        swaps_tiie[(tiie_period[i], tenor)] = data_tiie[i]

    # Qauntlib.Quote objects
    for n,unit in basis_usdmxn.keys():
        basis_usdmxn[(n,unit)] = ql.SimpleQuote(basis_usdmxn[(n,unit)])
    for n,unit in fwdpts.keys():
        fwdpts[(n,unit)] = ql.SimpleQuote(fwdpts[(n,unit)])
    for n,unit in deposits.keys():
        deposits[(n,unit)] = ql.SimpleQuote(deposits[(n,unit)])
    for n,unit in swaps_tiie.keys():
        swaps_tiie[(n,unit)] = ql.SimpleQuote(swaps_tiie[(n,unit)])
        
    # Deposit rate helper
    dayCounter = ql.Actual360()
    settlementDays = 1
    depositHelpers = [ql.DepositRateHelper(
        ql.QuoteHandle(deposits[(n, unit)]),
        ql.Period(n*4, ql.Weeks), 
        settlementDays,
        calendar_mx, 
        ql.Following, # Mty date push fwd if holyday even though changes month
        False, 
        dayCounter
        )
        for n, unit in deposits.keys()
    ]

    # FX Forwards helper
    fxSwapHelper = [ql.FxSwapRateHelper(
        ql.QuoteHandle(fwdpts[(n,u)]),
        ql.QuoteHandle(
            ql.SimpleQuote(spotfx)),
        ql.Period(n, u),
        1,
        calendar_mx,
        ql.Following,
        False,
        True,
        discount_curve
        ) 
        for n,u in fwdpts.keys()
    ]

    # Swap rate helpers
    #settlementDays = 2
    fixedLegFrequency = ql.EveryFourthWeek
    fixedLegAdjustment = ql.Following
    fixedLegDayCounter = ql.Actual360()
    if crvType == 'SOFR':
        fxIborIndex = ql.Sofr()
    else:
        fxIborIndex = ql.USDLibor(ql.Period('1M'), crv_usdswp)

    swapHelpers = [ql.SwapRateHelper(ql.QuoteHandle(swaps_tiie[(n,unit)]),
                                   ql.Period(n*4, ql.Weeks), 
                                   calendar_mx,
                                   fixedLegFrequency, 
                                   fixedLegAdjustment,
                                   fixedLegDayCounter, 
                                   fxIborIndex, 
                                   ql.QuoteHandle(basis_usdmxn[(n,unit)]), 
                                   ql.Period(0, ql.Days),
                                   ql.YieldTermStructureHandle(),
                                   1)
                   for n, unit in swaps_tiie.keys() ]

    # Rate helpers merge
    helpers = fxSwapHelper + swapHelpers

    return(helpers)

# QuantLib's Helper object for Implied MXNOIS Crv Bootstrapping w/oFutures
def qlHelper_MXNOISwF(dic_df, discount_curve, crv_usdswp, crvType = 'SOFR'):
    # Calendars
    calendar_mx = ql.Mexico(0)

    # Handle dat
    df_basis = dic_df['USDMXN_XCCY_Basis']
    df_tiie = dic_df['MXN_TIIE']
  
    # Input data
    tenor = ql.EveryFourthWeek
    basis_period = df_basis['Period'].astype(int).tolist()
    tiie_period = df_tiie['Period'].astype(int).to_list()
    data_tiie = (df_tiie['Quotes']/100).tolist()

    if crvType == 'SOFR':
        data_basis = (-1*df_basis['Quotes']/10000).tolist()
    else:
        data_basis = (df_basis['Quotes']/10000).tolist()
    
    # Basis swaps
    basis_usdmxn = {}
    n_basis = len(basis_period)
    for i in range(1,n_basis):
        basis_usdmxn[(basis_period[i], tenor)] = data_basis[i]

    # Deposit rates
    deposits = {(tiie_period[0], tenor): data_tiie[0]}
    
    # TIIE Swap rates]
    swaps_tiie = {}
    n_tiie = len(tiie_period)
    for i in range(1,n_tiie):
        swaps_tiie[(tiie_period[i], tenor)] = data_tiie[i]

    # Qauntlib.Quote objects
    for n,unit in basis_usdmxn.keys():
        basis_usdmxn[(n,unit)] = ql.SimpleQuote(basis_usdmxn[(n,unit)])
    
    for n,unit in deposits.keys():
        deposits[(n,unit)] = ql.SimpleQuote(deposits[(n,unit)])
    for n,unit in swaps_tiie.keys():
        swaps_tiie[(n,unit)] = ql.SimpleQuote(swaps_tiie[(n,unit)])
        
    # Deposit rate helper
    dayCounter = ql.Actual360()
    settlementDays = 1
    depositHelpers = [ql.DepositRateHelper(
        ql.QuoteHandle(deposits[(n, unit)]),
        ql.Period(n*4, ql.Weeks), 
        settlementDays,
        calendar_mx, 
        ql.Following,
        False, 
        dayCounter
        )
        for n, unit in deposits.keys()
    ]

    # Swap rate helpers
    settlementDays = 1
    fixedLegFrequency = ql.EveryFourthWeek
    fixedLegAdjustment = ql.Following
    fixedLegDayCounter = ql.Actual360()
    if crvType == 'SOFR':
        fxIborIndex = ql.Sofr(crv_usdswp)
    else:
        fxIborIndex = ql.USDLibor(ql.Period('1M'), crv_usdswp)

    swapHelpers = [ql.SwapRateHelper(ql.QuoteHandle(swaps_tiie[(n,unit)]),
                                   ql.Period(n*4, ql.Weeks), 
                                   ql.TARGET(),
                                   fixedLegFrequency, 
                                   fixedLegAdjustment,
                                   fixedLegDayCounter, 
                                   fxIborIndex, 
                                   ql.QuoteHandle(basis_usdmxn[(n,unit)]), 
                                   ql.Period(0, ql.Days))
                   for n, unit in swaps_tiie.keys() ]
    
    # Rate helpers merge
    helpers = depositHelpers + swapHelpers

    return(helpers)

# QuantLib's Helper object for TIIE Crv Bootstrapping
def qlHelper_MXNTIIE(df, crv_MXNOIS):
    # calendar
    calendar_mx = ql.Mexico(0)
    # data
    tenor = ql.EveryFourthWeek
    period = df['Period'].astype(int).to_list()
    data = (df['Quotes']/100).tolist()
    
    # Deposit rates
    deposits = {(period[0], tenor): data[0]}
    # Swap rates
    n = len(period)
    swaps = {}
    for i in range(1,n):
        swaps[(period[i], tenor)] = data[i]
        
    # Rate Qauntlib.Quote objects
    ## desposits
    for n, unit in deposits.keys():
        deposits[(n,unit)] = ql.SimpleQuote(deposits[(n,unit)])
    ## swap rates
    for n,unit in swaps.keys():
        swaps[(n,unit)] = ql.SimpleQuote(swaps[(n,unit)])
        
    # Deposit rate helpers
    dayCounter = ql.Actual360()
    settlementDays = 1
    depositHelpers = [ql.DepositRateHelper(
        ql.QuoteHandle(deposits[(n, unit)]),
        ql.Period(n*4, ql.Weeks), 
        settlementDays,
        calendar_mx, 
        ql.Following,
        False, dayCounter
        )
        for n, unit in deposits.keys()
    ]

    # Swap rate helpers
    settlementDays = 1
    fixedLegFrequency = ql.EveryFourthWeek
    fixedLegAdjustment = ql.Following
    fixedLegDayCounter = ql.Actual360()
    ibor_MXNTIIE = ql.IborIndex('TIIE',
                 ql.Period(13),
                 settlementDays,
                 ql.MXNCurrency(),
                 calendar_mx,
                 ql.Following,
                 False,
                 ql.Actual360())
                 # crv_MXNOIS) discounting should be done in the swap helpers

    swapHelpers = [ql.SwapRateHelper(
        ql.QuoteHandle(swaps[(n,unit)]),
        ql.Period(n*4, ql.Weeks), 
        calendar_mx,
        fixedLegFrequency, 
        fixedLegAdjustment,
        fixedLegDayCounter, 
        ibor_MXNTIIE, ql.QuoteHandle(), ql.Period(), crv_MXNOIS)
        for n, unit in swaps.keys()
    ]

    # helpers merge
    helpers = depositHelpers + swapHelpers
    
    return(helpers)

###############################################################################
# Curve Bootstrappers
###############################################################################
# USDOIS CURVE BOOTSTRAPPING
def btstrap_USDOIS(dic_data):
    hlprUSDOIS = qlHelper_USDOIS(dic_data['USD_OIS'])
    crvUSDOIS = ql.PiecewiseLogLinearDiscount(0, ql.UnitedStates(0), 
                                                    hlprUSDOIS, ql.Actual360())
    crvUSDOIS.enableExtrapolation()
    return crvUSDOIS

# USDSOFR CURVE BOOTSTRAPPING
def btstrap_USDSOFR(dic_data, crvUSDOIS):
    crvDiscUSD = ql.RelinkableYieldTermStructureHandle()
    crvDiscUSD.linkTo(crvUSDOIS)
        
    hlprSOFR = qlHelper_SOFR(dic_data, crvDiscUSD)
    crvSOFR = ql.PiecewiseNaturalLogCubicDiscount(0, ql.UnitedStates(0), 
                                                   hlprSOFR, 
                                                   ql.Actual360())
    crvSOFR.enableExtrapolation()
    
    return crvSOFR

# USD3M CURVE BOOTSTRAPPING
def btstrap_USD3M(dic_data, crvUSDOIS):
    crvDiscUSD = ql.RelinkableYieldTermStructureHandle()
    crvDiscUSD.linkTo(crvUSDOIS)
        
    hlprUSD3M = qlHelper_USD3M(dic_data, crvDiscUSD)
    crvUSD3M = ql.PiecewiseNaturalLogCubicDiscount(0, ql.UnitedStates(0), 
                                                   hlprUSD3M, 
                                                   ql.Actual360())
    crvUSD3M.enableExtrapolation()
    return crvUSD3M

# USD1M CURVE BOOTSTRAPPING
def btstrap_USD1M(dic_data, crvUSD3M):
    crv_usd3m = ql.RelinkableYieldTermStructureHandle()
    crv_usd3m.linkTo(crvUSD3M)
    hlprUSD1M = qlHelper_USD1M(dic_data, crv_usd3m)
    crvUSD1M = ql.PiecewiseNaturalLogCubicDiscount(0, ql.UnitedStates(0), 
                                                   hlprUSD1M, 
                                                   ql.Actual360())
    crvUSD1M.enableExtrapolation()
    return crvUSD1M

# MXNOIS CURVE BOOTSTRAPPING with Futures
def btstrap_MXNOIS(dic_data, crvUSDSWP, crvUSDOIS, crvType='SOFR'):
    crvDiscUSD = ql.RelinkableYieldTermStructureHandle()
    crvDiscUSD.linkTo(crvUSDOIS)
    crv_usdswp = ql.RelinkableYieldTermStructureHandle()
    crv_usdswp.linkTo(crvUSDSWP)
    # with Futures
    hlprMXNOIS = qlHelper_MXNOIS(dic_data, crvDiscUSD, crv_usdswp, crvType)
    crvMXNOIS = ql.PiecewiseNaturalLogCubicDiscount(0, ql.Mexico(0), 
                                                   hlprMXNOIS, 
                                                   ql.Actual360())
    crvMXNOIS.enableExtrapolation()
    return crvMXNOIS

# MXNOIS CURVE BOOTSTRAPPING w/o Futures
def btstrap_MXNOISwF(dic_data, crvUSDSWP, crvUSDOIS, crvType='SOFR'):
    crvDiscUSD = ql.RelinkableYieldTermStructureHandle()
    crvDiscUSD.linkTo(crvUSDOIS)
    crv_usdswp = ql.RelinkableYieldTermStructureHandle()
    crv_usdswp.linkTo(crvUSDSWP)
    # w/o Futures
    hlprMXNOIS = qlHelper_MXNOISwF(dic_data, crvDiscUSD, crv_usdswp, crvType)
    crvMXNOIS = ql.PiecewiseNaturalLogCubicDiscount(0, ql.Mexico(0), 
                                                   hlprMXNOIS, 
                                                   ql.Actual360())
    crvMXNOIS.enableExtrapolation()
    return crvMXNOIS

# TIIE CURVE BOOTSTRAPPING
def btstrap_MXNTIIE(dic_data, crvMXNOIS):
    crv_mxnois = ql.RelinkableYieldTermStructureHandle()
    crv_mxnois.linkTo(crvMXNOIS)
    hlprTIIE = qlHelper_MXNTIIE(dic_data['MXN_TIIE'], crv_mxnois)
    crvTIIE = ql.PiecewiseNaturalLogCubicDiscount(0, ql.Mexico(0), hlprTIIE, 
                                                  ql.Actual360())
    crvTIIE.enableExtrapolation()
    return crvTIIE
###############################################################################
# Curves Risk Sensibility
###############################################################################
# Funciton to build curves for flat risk when US Holiday
def crvFlatRisk_TIIE_isUSH(dic_df, crvMXNOIS):
    # Data
    modic = {k:v.copy() for k,v in dic_df.items()}
    # rates data
    df_tiie = modic['MXN_TIIE'].copy()
    # Data Shift
    df_tiie_up = df_tiie['Quotes'] + 1/100
    df_tiie_down = df_tiie['Quotes'] - 1/100
    dic_crvs = {}
    # Curves UpShift
    modic['MXN_TIIE']['Quotes'] = df_tiie_up
    
    # Proj Curve
    crvTIIE = btstrap_MXNTIIE(modic, crvMXNOIS)
    # Save
    dic_crvs['Crv+1'] = [crvMXNOIS, crvTIIE]
    # Curves DownShift
    modic['MXN_TIIE']['Quotes'] = df_tiie_down
    
    # Proj Curve
    crvTIIE = btstrap_MXNTIIE(modic, crvMXNOIS)
    # Save
    dic_crvs['Crv-1'] = [crvMXNOIS, crvTIIE]
    
    return dic_crvs

# Funciton to build curves for flat risk
def crvFlatRisk_TIIE(dic_df, crvDiscUSD, crv_usdswp):
    # Data
    modic = {k:v.copy() for k,v in dic_df.items()}
    # rates data
    df_tiie = modic['MXN_TIIE'].copy()
    # Data Shift
    df_tiie_up = df_tiie['Quotes'] + 1/100
    df_tiie_down = df_tiie['Quotes'] - 1/100
    dic_crvs = {}
    # Curves UpShift
    modic['MXN_TIIE']['Quotes'] = df_tiie_up
    crvMXNOIS = btstrap_MXNOIS(modic, crv_usdswp, 
                               crvDiscUSD, crvType='SOFR')
    # Proj Curve
    crvTIIE = btstrap_MXNTIIE(modic, crvMXNOIS)
    # Save
    dic_crvs['Crv+1'] = [crvMXNOIS, crvTIIE]
    # Curves DownShift
    modic['MXN_TIIE']['Quotes'] = df_tiie_down
    crvMXNOIS = btstrap_MXNOIS(modic, crv_usdswp, 
                               crvDiscUSD, crvType='SOFR')
    # Proj Curve
    crvTIIE = btstrap_MXNTIIE(modic, crvMXNOIS)
    # Save
    dic_crvs['Crv-1'] = [crvMXNOIS, crvTIIE]
    
    return dic_crvs

# Curves for Risk Sens by Tenor with Futures when US Holiday
def crvTenorRisk_TIIE_isUSH(dic_df, crvMXNOIS):
    # Data
    modic = {k:v.copy() for k,v in dic_df.items()}
    # rates data
    df_tiie = modic['MXN_TIIE'].copy()
    
    # Curves by tenor mod
    dict_crvs = dict({})
    for i,r in df_tiie.iterrows():
        tmpdf = df_tiie.copy()
        # Rate mods
        tenor = r['Tenor']
        rate_plus_1bp = r['Quotes'] + 1/100
        rate_min_1bp = r['Quotes'] - 1/100
        # Tenor +1bp
        tmpdf['Quotes'][df_tiie['Tenor'] == tenor] = rate_plus_1bp
        modic['MXN_TIIE']['Quotes'] = tmpdf['Quotes']
        # Disc Curve
        #crvMXNOIS = btstrap_MXNOIS(modic, crv_usdswp, 
        #                           crvDiscUSD, crvType='SOFR')
        # Proj Curve
        crvTIIE = btstrap_MXNTIIE(modic, crvMXNOIS)
        # Save
        dict_crvs[tenor+'+1'] = [crvMXNOIS, crvTIIE]
        # Tenor -1bp
        tmpdf['Quotes'][df_tiie['Tenor'] == tenor] = rate_min_1bp
        modic['MXN_TIIE']['Quotes'] = tmpdf['Quotes']
        # Disc Curve
        #crvMXNOIS = btstrap_MXNOIS(modic, crv_usdswp, 
        #                           crvDiscUSD, crvType='SOFR')
        # Proj Curve
        crvTIIE = btstrap_MXNTIIE(modic, crvMXNOIS)
        # Save
        dict_crvs[tenor+'-1'] = [crvMXNOIS, crvTIIE]
    
    return dict_crvs
    
# Curves for Risk Sens by Tenor with Futures
def crvTenorRisk_TIIE(dic_df, crvDiscUSD, crv_usdswp, crvMXNOIS):
    # Data
    modic = {k:v.copy() for k,v in dic_df.items()}
    # rates data
    df_tiie = modic['MXN_TIIE'].copy()
    
    # Curves by tenor mod
    dict_crvs = dict({})
    for i,r in df_tiie.iterrows():
        tmpdf = df_tiie.copy()
        # Rate mods
        tenor = r['Tenor']
        rate_plus_1bp = r['Quotes'] + 1/100
        rate_min_1bp = r['Quotes'] - 1/100
        
        # Tenor +1bp
        tmpdf['Quotes'][df_tiie['Tenor'] == tenor] = rate_plus_1bp
        modic['MXN_TIIE']['Quotes'] = tmpdf['Quotes']
        
        # Disc Curve
        crvMXNOIS = btstrap_MXNOIS(modic, crv_usdswp, 
                                   crvDiscUSD, crvType='SOFR')
        # Proj Curve
        crvTIIE = btstrap_MXNTIIE(modic, crvMXNOIS)
        # Save
        dict_crvs[tenor+'+1'] = [crvMXNOIS, crvTIIE]
        # Tenor -1bp
        tmpdf['Quotes'][df_tiie['Tenor'] == tenor] = rate_min_1bp
        modic['MXN_TIIE']['Quotes'] = tmpdf['Quotes']
        # Disc Curve
        crvMXNOIS = btstrap_MXNOIS(modic, crv_usdswp, 
                                   crvDiscUSD, crvType='SOFR')
        # Proj Curve
        crvTIIE = btstrap_MXNTIIE(modic, crvMXNOIS)
        # Save
        dict_crvs[tenor+'-1'] = [crvMXNOIS, crvTIIE]
    
    return dict_crvs

# Curves for Risk Sens by Tenor w/o Futures
def crvTenorRisk_TIIEwF(dic_df, crvDiscUSD, crv_usdswp):
    # Data
    modic = dic_df.copy()
    # rates data
    df_tiie = dic_df['MXN_TIIE'][['Tenor','Quotes']].copy()
    
    # Curves by tenor mod
    dict_crvs = dict({})
    for i,r in df_tiie.iterrows():
        tmpdf = df_tiie.copy()
        # Rate mods
        tenor = r['Tenor']
        rate_plus_1bp = r['Quotes'] + 1/100
        rate_min_1bp = r['Quotes'] - 1/100
        # Tenor +1bp
        tmpdf['Quotes'][df_tiie['Tenor'] == tenor] = rate_plus_1bp
        modic['MXN_TIIE']['Quotes'] = tmpdf['Quotes']
        # Disc Curve
        crvMXNOIS = btstrap_MXNOISwF(modic, crv_usdswp, crvDiscUSD, crvType='SOFR')
        # Proj Curve
        crvTIIE = btstrap_MXNTIIE(modic, crvMXNOIS)
        # Save
        dict_crvs[tenor+'+1'] = [crvMXNOIS, crvTIIE]
        # Tenor -1bp
        tmpdf['Quotes'][df_tiie['Tenor'] == tenor] = rate_min_1bp
        modic['MXN_TIIE']['Quotes'] = tmpdf['Quotes']
        # Disc Curve
        crvMXNOIS = btstrap_MXNOISwF(modic, crv_usdswp, crvDiscUSD, crvType='SOFR')
        # Proj Curve
        crvTIIE = btstrap_MXNTIIE(modic, crvMXNOIS)
        # Save
        dict_crvs[tenor+'-1'] = [crvMXNOIS, crvTIIE]
    
    return(dict_crvs)
###############################################################################
# Swap Features
###############################################################################
# TIIE Swap Fixings
def get_fixings_TIIE28_banxico(evaluation_date):
    token="c1b63f15802a3378307cc2eb90a09ae8e821c5d1ef04d9177a67484ee6f9397c" 
    banxico_start_date = (evaluation_date - \
                          timedelta(days = 3600)).strftime('%Y-%m-%d')
    banxico_end_date = evaluation_date.strftime('%Y-%m-%d')
    banxico_TIIE28 = banxico_download_data('SF43783', banxico_start_date, 
                                              banxico_end_date, token)
    return banxico_TIIE28

# TIIE-IRS Ibor Index
def set_ibor_TIIE(crvTIIE, str_tiiefixings_file = None, n = 30):
    # Eval Date
    ql_eval_date = ql.Settings.instance().evaluationDate
    dt_eval_date = date(ql_eval_date.year(),
                        ql_eval_date.month(),
                        ql_eval_date.dayOfMonth())
    # Fixings
    banxico_TIIE28 = get_fixings_TIIE28_banxico(dt_eval_date)
    # TIIE IBOR INDEX
    if type(crvTIIE) != type(ql.RelinkableYieldTermStructureHandle()):
        ibor_tiie_crv = ql.RelinkableYieldTermStructureHandle()
        ibor_tiie_crv.linkTo(crvTIIE)
    else:
        ibor_tiie_crv = crvTIIE
    # IborIndex
    ibor_tiie = ql.IborIndex('TIIE',
                 ql.Period(13),
                 1,
                 ql.MXNCurrency(),
                 ql.Mexico(0),
                 ql.Following,
                 False,
                 ql.Actual360(),
                 ibor_tiie_crv)
    ###########################################################################
    # Ibor Index Fixings
    ibor_tiie.clearFixings()
    for h in range(len(banxico_TIIE28['fecha']) - 1):
        dt_fixing = pd.to_datetime(banxico_TIIE28['fecha'][h])
        ibor_tiie.addFixing(
            ql.Date(dt_fixing.day, dt_fixing.month, dt_fixing.year), 
            banxico_TIIE28['dato'][h+1]
            )
    # fill mixing fixings up until eval date
    dt_last = banxico_TIIE28.iloc[-1]['fecha']
    if dt_last < dt_eval_date:
        ql_hldy = ql.Mexico().\
            holidayList(ql.Date(dt_last.day, dt_last.month, dt_last.year), 
            ql.Date(dt_eval_date.day, dt_eval_date.month, dt_eval_date.year))
        dt_hldy = [date(qdt.year(), qdt.month(), qdt.dayOfMonth()) for qdt in ql_hldy]
        dt_rng = pd.bdate_range(banxico_TIIE28.iloc[-1]['fecha'], 
                                dt_eval_date, freq='C',
                                weekmask = "Mon Tue Wed Thu Fri",
                                holidays=dt_hldy)
        for nfd in dt_rng:
            ibor_tiie.addFixing(
                ql.Date(nfd.day, nfd.month, nfd.year), 
                banxico_TIIE28['dato'].iloc[-1]
                )
    
    ###########################################################################
    # HITORICAL FIXINGS
    # file import
    #df_tiieFxngs = pd.read_excel(str_tiiefixings_file, 'Hoja1', 
    #                             skiprows=17, engine = 'openpyxl')
    #df_tiieFxngs = df_tiieFxngs[['Fecha','SF43783']]
    #df_tiieFxngs = df_tiieFxngs[df_tiieFxngs['SF43783']!='N/E']
    #df_tiieFxngs['qlDate'] = [ql.Mexico(0).advance(
    #    ql.Date(t.day,t.month,t.year),-1,ql.Days) for t in df_tiieFxngs['Fecha']]
    # historical fixings update
    #ibor_tiie.addFixings([t for t in df_tiieFxngs.tail(n).iloc[:,2]], 
    #                        [r/100 for r in df_tiieFxngs.tail(n).iloc[:,1]], 
    #                        forceOverwrite=True)
    return(ibor_tiie)

###############################################################################
# Fixing Rates
###############################################################################
# Fetch TIIE28 IborIndex from Banxico's website
def banxico_download_data(serie, banxico_start_date, banxico_end_date, token):
    # URL
    url = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/" + \
        serie + "/datos/" + banxico_start_date + "/" + banxico_end_date
    # Request specs
    headers={'Bmx-Token':token}
    response = requests.get(url,headers=headers) 
    status = response.status_code 
    if status!=200: #Error en la obtención de los datos
        return print('Error Banxico TIIE 1D')
    # Data pull
    raw_data = response.json()
    data = raw_data['bmx']['series'][0]['datos'] 
    df = pd.DataFrame(data) 
    # Data mgmt
    df["dato"] = df["dato"].str.replace(',','')
    df["dato"] = df["dato"].str.replace('N/E','0')
    df['dato'] = df['dato'].apply(lambda x:float(x)) / 100
    df['fecha'] = pd.to_datetime(df['fecha'],format='%d/%m/%Y')

    return df