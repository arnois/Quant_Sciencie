# Code Description
"""
This Module supports TIIE_Trading for all functions used in the TIIE IRS 
Toolkit. 
"""
###############################################################################
# Modules
###############################################################################
import QuantLib as ql
import numpy as np
import pandas as pd
import requests
import sys, os
str_cwd = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(str_cwd)
import udf_TIIE_CurveCreate as udf
from datetime import timedelta
###############################################################################
# Global variables
###############################################################################
tenor2ql = {'B':ql.Days,'D':ql.Days,'M':ql.Months,'W':ql.Weeks,'Y':ql.Years}
clr_orange_output = (253, 233, 217)
###############################################################################
# Flat DV01
###############################################################################
def flat_DV01(frCrvs, banxico_TIIE28, base_npv,
              start, maturity, notional, rate, typ, rule):
    npvs = []
    for crvKey in frCrvs.keys():
        # UpShift Curves
        crvMXNOIS, crvTIIE = frCrvs[crvKey]
        crv_mxnois = ql.RelinkableYieldTermStructureHandle()
        crv_mxnois.linkTo(crvMXNOIS)
    
        # TIIE Ibor Index
        ibor_tiie_crv = ql.RelinkableYieldTermStructureHandle()
        ibor_tiie_crv.linkTo(crvTIIE)
        ibor_tiie = ql.IborIndex('TIIE',
                     ql.Period(13),
                     1,
                     ql.MXNCurrency(),
                     ql.Mexico(),
                     ql.Following,
                     False,
                     ql.Actual360(),
                     ibor_tiie_crv)
        # TIIE Ibor Index Fixings
        ibor_tiie.clearFixings()
        for h in range(len(banxico_TIIE28['fecha']) - 1):
            dt_fixing = pd.to_datetime(banxico_TIIE28['fecha'][h])
            ibor_tiie.addFixing(
                ql.Date(dt_fixing.day, dt_fixing.month, dt_fixing.year), 
                banxico_TIIE28['dato'][h+1]
                )
        # TIIE Swap Pricing Engine
        rytsMXNOIS = ql.RelinkableYieldTermStructureHandle()
        rytsMXNOIS.linkTo(crvMXNOIS)
        tiie_swp_engine = ql.DiscountingSwapEngine(rytsMXNOIS)
        # TIIE Swap Pricing
        swap_valuation = tiieSwap(start, maturity, notional, 
                                  ibor_tiie, rate, typ, rule)
        swap_valuation.setPricingEngine(tiie_swp_engine)
        npvs.append(swap_valuation.NPV())
    
    # Swap DV01
    npv_up = npvs[0] - base_npv
    npv_down = npvs[1] - base_npv
    npv_dv01 = abs(np.array([npv_up,npv_down])).mean()*np.sign(npv_up)
    
    return npv_dv01
###############################################################################
# Vanilla TIIE Swap
###############################################################################
def tiieSwap(start, maturity, notional, ibor_tiie, rate, typ, rule):
    # TIIE Swap Schedule Specs
    cal = ql.Mexico()
    legDC = ql.Actual360()
    cpn_tenor = ql.Period(13)
    convention = ql.Following
    termDateConvention = ql.Following
    rule = rule
    isEndOfMonth = False
    
    # fix-float leg schedules
    fixfltSchdl = ql.Schedule(start,
                            maturity, 
                            cpn_tenor,
                            cal,
                            convention,
                            termDateConvention,
                            rule,
                            isEndOfMonth)
    
    swap = ql.VanillaSwap(typ, 
                             notional, 
                             fixfltSchdl,
                             rate,
                             legDC,
                             fixfltSchdl,
                             ibor_tiie,
                             0,
                             legDC
                             )

    return swap
###############################################################################
# Swap pricing legs
###############################################################################
# Swap Fixing Dates
def tiieSwap_FixingDates(swp, iborIdx):
    # TIIE Swap Schedule Specs
    fixfltSchdl = swp.floatingSchedule()
    
    return [iborIdx.fixingDate(x) for x in fixfltSchdl][:-1]

# Swap OnTheRun CF
def get_CF_tiieSwapOTR(swp, iborIdx, ql_tdy):
    lst_fxngs = tiieSwap_FixingDates(swp, iborIdx)
    swp_type = swp.type()
    swp_leg0 = tuple([swpleg0 for swpleg0 in swp.leg(0) 
                      if swpleg0.date() >= ql_tdy])
    swp_leg1 = tuple([swpleg1 for swpleg1 in swp.leg(1) 
                      if swpleg1.date() >= ql_tdy])
    n_fxngs = len(swp_leg1)
    fxngs_alive = lst_fxngs[-n_fxngs:]
    cf1_l1 = pd.DataFrame({
        'date': pd.to_datetime(str(cf.date())),
        'accStartDate': pd.to_datetime(str(cf.accrualStartDate().ISO())),
        'accEndDate': pd.to_datetime(str(cf.accrualEndDate().ISO())),
        'fixAmt': cf.amount()*-1*swp_type
        } for cf in map(ql.as_coupon, swp_leg0)
        )
    cf1_l2 = pd.DataFrame({
        'date': pd.to_datetime(str(cf.date())),
        'fixingDate': pd.to_datetime(str(fd)),
        'accStartDate': pd.to_datetime(str(cf.accrualStartDate().ISO())),
        'accEndDate': pd.to_datetime(str(cf.accrualEndDate().ISO())),
        'fltAmt': cf.amount()*swp_type
        } for (cf, fd) in zip(map(ql.as_coupon, swp_leg1), fxngs_alive)
        )

    cf = cf1_l1.copy()
    cf.insert(1, 'fixingDate',cf1_l2['fixingDate'])
    cf['fltAmt'] = cf1_l2['fltAmt']
    cf['netAmt'] = cf['fixAmt'] + cf['fltAmt']
    cf['fixAmt'] = cf['fixAmt']#.map('{:,.0f}'.format)
    cf['fltAmt'] = cf['fltAmt']#.map('{:,.0f}'.format)
    cf['netAmt'] = cf['netAmt']#.map('{:,.0f}'.format)
    cf['accDays'] = 1*(cf['accEndDate'] - cf['accStartDate'])/np.timedelta64(1, 'D')

    return cf

# Swap CF Dates
def get_CF_tiieSwap(swp, iborIdx):
    swp_fixdt = tiieSwap_FixingDates(swp, iborIdx)
    swp_type = swp.type()
    cf1_l1 = pd.DataFrame({
        'Date': pd.to_datetime(str(cf.date())),
        'Start_Date': pd.to_datetime(str(cf.accrualStartDate().ISO())),
        'End_Date': pd.to_datetime(str(cf.accrualEndDate().ISO())),
        'Fix_Amt': cf.amount()*-1*swp_type
        } for cf in map(ql.as_coupon, swp.leg(0))
        )
    cf1_l2 = pd.DataFrame({
        'Date': pd.to_datetime(str(cf.date())),
        'Fixing_Date': pd.to_datetime(str(fd)),
        'Start_Date': pd.to_datetime(str(cf.accrualStartDate().ISO())),
        'End_Date': pd.to_datetime(str(cf.accrualEndDate().ISO())),
        'Float_Amt': cf.amount()*swp_type
        } for (cf, fd) in zip(map(ql.as_coupon, swp.leg(1)), swp_fixdt)
        )

    cf = cf1_l1.copy()
    cf.insert(1, 'Fixing_Date',cf1_l2['Fixing_Date'])
    cf['Float_Amt'] = cf1_l2['Float_Amt']
    cf['Net_Amt'] = cf['Fix_Amt'] + cf['Float_Amt']
    cf['Fix_Amt'] = cf['Fix_Amt'].map('{:,.0f}'.format)
    cf['Float_Amt'] = cf['Float_Amt'].map('{:,.0f}'.format)
    cf['Net_Amt'] = cf['Net_Amt'].map('{:,.0f}'.format)
    cf['Acc_Days'] = (
        1*(cf['End_Date'] - cf['Start_Date']
           )/np.timedelta64(1,'D'))
    cf['Acc_Days'] = [int(i) for i in cf['Acc_Days'].values]

    return cf
###############################################################################
# External Data Pull
###############################################################################
# TIIE28D Fixings Data
def banxico_download_data(serie, banxico_start_date, banxico_end_date, token):
    url = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/" +\
        serie + "/datos/" + banxico_start_date + "/" + banxico_end_date
    headers={'Bmx-Token':token}
    response = requests.get(url,headers=headers) 
    status = response.status_code 
    if status!=200: #Error en la obtención de los datos
        return print('Error Banxico TIIE 1D')
    
    raw_data = response.json()
    data = raw_data['bmx']['series'][0]['datos'] 
    df = pd.DataFrame(data) 

    df["dato"] = df["dato"].str.replace(',','')
    df["dato"] = df["dato"].str.replace('N/E','0')
    df['dato'] = df['dato'].apply(lambda x:float(x)) / 100
    df['fecha'] = pd.to_datetime(df['fecha'],format='%d/%m/%Y')

    return df

###############################################################################
# Dates Mgmt
###############################################################################
# Dates Blotter
def start_end_dates_blotter(i, blotter_trades, ql_settle_dt):
    # iloc index
    i = np.where(blotter_trades.index == i)[0][0]
    # Dates
    settle_date = ql_settle_dt
    
    # Cases Conditions
    isStartTnrStartEndDtBlank = blotter_trades.iloc[i,0] == 0 and \
        blotter_trades.iloc[i,2] == 0 and blotter_trades.iloc[i,3] == 0
    isStartPrdTnrStartDtBlank = blotter_trades.iloc[i,0] == 0 and \
        blotter_trades.iloc[i,1] == 0 and blotter_trades.iloc[i,2] == 0
    isStartPrdTnrBlank = blotter_trades.iloc[i,0] == 0 and \
        blotter_trades.iloc[i,1] == 0
    isStartTnrEndDtBlank = blotter_trades.iloc[i,0] == 0 and \
        blotter_trades.iloc[i,3] == 0
    isStartEndDtBlank = blotter_trades.iloc[i,2] == 0 and \
        blotter_trades.iloc[i,3] == 0

    # Case only Fwd Tenor
    if isStartTnrStartEndDtBlank:
        start = settle_date
        tenor = int(blotter_trades.iloc[i,1])
        maturity = start + ql.Period(tenor*28, ql.Days) 
        
    # Case only End Date
    elif isStartPrdTnrStartDtBlank:
        start = settle_date
        dt_mty = pd.to_datetime(blotter_trades.iloc[i,3])
        maturity = ql.Date(dt_mty.day, dt_mty.month, dt_mty.year)
    
    # Case only Start Date and End Date        
    elif isStartPrdTnrBlank:
        dt_st = pd.to_datetime(blotter_trades.iloc[i,2])
        dt_mty = pd.to_datetime(blotter_trades.iloc[i,3])
        start = ql.Date(dt_st.day, dt_st.month, dt_st.year)
        maturity = ql.Date(dt_mty.day, dt_mty.month, dt_mty.year)

    # Case Period Tenor and Start Date    
    elif isStartTnrEndDtBlank:
        dt_st = pd.to_datetime(blotter_trades.iloc[i,2])
        start = ql.Date(dt_st.day, dt_st.month, dt_st.year)
        tenor = int(blotter_trades.iloc[i,1])
        maturity = start + ql.Period(tenor*28, ql.Days)  
    
    # Case Start and Period Tenor          
    elif isStartEndDtBlank:
        fwdStTenor = int(blotter_trades.iloc[i,0])
        start = settle_date + ql.Period(fwdStTenor*28, ql.Days)  
        tenor = int(blotter_trades.iloc[i,1])
        maturity = start + ql.Period(tenor* 28, ql.Days) 
    
    # Case Start and Period Tenor with Start Date  (IMM)
    elif blotter_trades.iloc[i,3] == 0:
        dt_st = pd.to_datetime(blotter_trades.iloc[i,2])
        start = ql.IMM.nextDate(ql.Date(dt_st.day, dt_st.month, dt_st.year)) 
        tenor = int(blotter_trades.iloc[i,1])
        maturity = start + ql.Period(tenor*28 , ql.Days) 
            
    return start, maturity

# Dates Trading
def start_end_dates_trading(i, parameters_trades, ql_settle_dt):
    # iloc index
    i = np.where(parameters_trades.index == i)[0][0]
    # Dates
    settle_date = ql_settle_dt
    
    # Cases Conditions
    isStartTnrStartEndDtBlank = parameters_trades.iloc[i,2] == 0 and \
        parameters_trades.iloc[i,4] == 0 and parameters_trades.iloc[i,5] == 0
    isStartFwdTnrStartDtBlank = parameters_trades.iloc[i,2] == 0 and \
        parameters_trades.iloc[i,3] == 0 and parameters_trades.iloc[i,4] == 0
    isStartFwdTnrBlank = parameters_trades.iloc[i,2] == 0 and \
        parameters_trades.iloc[i,3] == 0
    isStartTnrEndDtBlank = parameters_trades.iloc[i,2] == 0 and \
        parameters_trades.iloc[i,5] == 0
    isStartEndDtBlank = parameters_trades.iloc[i,4] == 0 and \
        parameters_trades.iloc[i,5] == 0
    
    # Case only Fwd Tenor
    if isStartTnrStartEndDtBlank:
        start = settle_date
        tenor = int(parameters_trades.iloc[i,3])
        maturity = start + ql.Period(tenor*28, ql.Days) 
        
    # Case only End Date
    elif isStartFwdTnrStartDtBlank:
        start = settle_date
        dt_mty = pd.to_datetime(parameters_trades.iloc[i,5])
        maturity = ql.Date(dt_mty.day, dt_mty.month, dt_mty.year)
        
    # Case only Start Date and End Date
    elif isStartFwdTnrBlank:
        dt_st = pd.to_datetime(parameters_trades.iloc[i,4])
        dt_mty = pd.to_datetime(parameters_trades.iloc[i,5])
        start = ql.Date(dt_st.day, dt_st.month, dt_st.year)
        maturity = ql.Date(dt_mty.day, dt_mty.month, dt_mty.year)
    
    # Case Fwd Tenor and Start Date
    elif isStartTnrEndDtBlank:
        dt_st = pd.to_datetime(parameters_trades.iloc[i,4])
        start = ql.Date(dt_st.day, dt_st.month, dt_st.year)
        tenor = int(parameters_trades.iloc[i,3])
        maturity = start + ql.Period(tenor*28, ql.Days)            
    
    # Case Start and Fwd Tenor    
    elif isStartEndDtBlank:
        fwdStTenor = int(parameters_trades.iloc[i,2])
        start = settle_date + ql.Period(fwdStTenor*28, ql.Days)  
        tenor = int(parameters_trades.iloc[i,3])
        maturity = start + ql.Period(tenor* 28, ql.Days)            
    
    # Case Start and Fwd Tenor with Start Date   
    elif parameters_trades.iloc[i,5] == 0:
        dt_st = pd.to_datetime(parameters_trades.iloc[i,4])
        start = ql.IMM.nextDate(ql.Date(dt_st.day, dt_st.month, dt_st.year)) 
        tenor = int(parameters_trades.iloc[i,3])
        maturity = start + ql.Period(tenor*28 , ql.Days) 
            
    return start, maturity
###############################################################################
# Pricer Cases Mgmt
###############################################################################
def get_fixings_TIIE28_banxico(evaluation_date):
    token="c1b63f15802a3378307cc2eb90a09ae8e821c5d1ef04d9177a67484ee6f9397c" 
    banxico_start_date = (evaluation_date - \
                          timedelta(days = 3600)).strftime('%Y-%m-%d')
    banxico_end_date = evaluation_date.strftime('%Y-%m-%d')
    banxico_TIIE28 = banxico_download_data('SF43783', banxico_start_date, 
                                              banxico_end_date, token)
    return banxico_TIIE28

def eval_swprate(i, df_trades, ql_settle_dt, 
                 ibor_tiie, tiie_swp_engine, frCrvs, banxico_TIIE28):
    global input_fxrate
    # Inputs
    rule = (df_trades.loc[i,df_trades.columns[10]] == 'Forward')*1
    input_notional = float(df_trades.loc[i,df_trades.columns[6]])
    #input_npvmxn = float(df_trades.loc[i,df_trades.columns[8]])
    input_dv01usd = float(df_trades.loc[i,df_trades.columns[9]])
    input_rate = 0
    # Default Specs
    start, maturity = start_end_dates_trading(i, df_trades, 
                                                 ql_settle_dt)
    if input_dv01usd==0 and input_notional!=0: # DV01 given Notional
        # Givens
        notional = abs(input_notional)
        typ = int(np.sign(input_notional)*-1)
        # Instance
        swap = tiieSwap(start, maturity, notional, 
                           ibor_tiie, 0.04, typ, rule)
        swap.setPricingEngine(tiie_swp_engine)
        swap_swprate = swap.fairRate()
        lstk = ['Trade#', 'StartDate', 'EndDate', 'InputNotional',
                'Rate', 'FairRate', 'NPV', 'DV01(USD)_Inputs']
    elif input_dv01usd!=0: # Notional given DV01
        # Default givens
        default_notional = int(1e8)
        default_typ = int(np.sign(default_notional)*-1)
        # Instance
        default_swap = tiieSwap(start, maturity, default_notional, 
                           ibor_tiie, 0.04, default_typ, rule)
        default_swap.setPricingEngine(tiie_swp_engine)
        swap_swprate = default_swap.fairRate()
        # Default DV01
        default_flat_dv01 = flat_DV01(frCrvs, banxico_TIIE28, 
                                      default_swap.NPV(),
                                      start, maturity, default_notional, 
                                      swap_swprate, default_typ, rule)
        dv01_100mn = default_flat_dv01/input_fxrate
        # Notional
        notional = input_dv01usd*default_notional/dv01_100mn
        typ = int(np.sign(notional)*-1)
        lstk = ['Trade#', 'StartDate', 'EndDate', 'Notional_Inputs',
                'Rate', 'FairRate', 'NPV', 'InputDV01(USD)']
    else: # Default to Notional given default DV01
        # Default givens
        input_dv01usd = 1.075e3*-1
        default_notional = int(1e8)
        default_typ = int(np.sign(default_notional)*-1)
        # Instance
        default_swap = tiieSwap(start, maturity, default_notional, 
                           ibor_tiie, 0.04, default_typ, rule)
        default_swap.setPricingEngine(tiie_swp_engine)
        swap_swprate = default_swap.fairRate()
        # Default DV01
        default_flat_dv01 = flat_DV01(frCrvs, banxico_TIIE28, 
                                      0,
                                      start, maturity, default_notional, 
                                      swap_swprate, default_typ, rule)
        dv01_100mn = default_flat_dv01/input_fxrate
        # Notional
        notional = input_dv01usd*default_notional/dv01_100mn
        typ = int(np.sign(notional)*-1)
        lstk = ['Trade#', 'StartDate', 'EndDate', 'Notional_Inputs',
                'Rate', 'FairRate', 'NPV', 'DV01(USD)']
    # Swap
    swap = tiieSwap(start, maturity, abs(notional), 
                       ibor_tiie, swap_swprate, typ, rule)
    swap.setPricingEngine(tiie_swp_engine)
    swap_npv = swap.NPV()
    # DV01
    flat_dv01 = flat_DV01(frCrvs, banxico_TIIE28, swap_npv,
                             start, maturity, abs(notional), 
                             swap_swprate, typ, rule)
    # Output
    lstv = [i,swap.startDate(),swap.maturityDate(),notional,
            input_rate,swap_swprate*100,swap_npv,flat_dv01/input_fxrate]
    res_dic = dict(zip(lstk,lstv))
    
    _res_df = pd.DataFrame([res_dic])
    _res_df = _res_df.set_index('Trade#',drop=True)
        
    return _res_df, swap

def eval_npv_dv01(i, df_trades, ql_settle_dt, 
                  ibor_tiie, tiie_swp_engine, frCrvs, banxico_TIIE28):
    global input_fxrate
    # Specs
    start, maturity = start_end_dates_trading(i, df_trades, 
                                                 ql_settle_dt)
    input_notional = float(df_trades.loc[i,df_trades.columns[6]])
    input_rate = df_trades.loc[i,df_trades.columns[7]]
    notional = abs(input_notional)
    rate = float(input_rate)
    typ = int(np.sign(input_notional)*-1)
    rule = (df_trades.loc[i,df_trades.columns[10]] == 'Forward')*1
    # Instance
    swap = tiieSwap(start, maturity, notional, 
                       ibor_tiie, rate, typ, rule)
    swap.setPricingEngine(tiie_swp_engine)
    # Swap val
    swap_npv = swap.NPV()
    swap_swprate = swap.fairRate()
    # Swap risk 
    flat_dv01 = flat_DV01(frCrvs, banxico_TIIE28, swap_npv,
                             start, maturity, notional, 
                             rate, typ, rule)
    # Output
    global input_fxrate
    res_dic = {'Trade#': i, 'StartDate': swap.startDate(), 
     'EndDate': swap.maturityDate(), 'InputNotional_MXN': -1*typ*notional, 
     'InputRate': rate*100, 'FairRate': swap_swprate*100, 
     'NPV_Inputs': swap_npv, 'DV01(USD)_Inputs': flat_dv01/input_fxrate}
    _res_df = pd.DataFrame([res_dic])
    _res_df = _res_df.set_index('Trade#',drop=True)
    
    return _res_df, swap

def eval_notional(i, df_trades, ql_settle_dt, 
                  ibor_tiie, tiie_swp_engine, frCrvs, banxico_TIIE28):
    global input_fxrate
    # Inputs
    input_npvmxn = df_trades.loc[i,df_trades.columns[8]]
    input_dv01usd = df_trades.loc[i,df_trades.columns[9]]
    input_rate = df_trades.loc[i,df_trades.columns[7]]
    # Default Specs
    start, maturity = start_end_dates_trading(i, df_trades, 
                                                 ql_settle_dt)
    notional = int(100e6)
    rate = float(input_rate)
    rule = ql.DateGeneration.Backward
    typ = ql.VanillaSwap.Receiver
    # Default Instance
    swap0 = tiieSwap(start, maturity, notional, ibor_tiie, 
                                 rate, typ, rule)
    swap0.setPricingEngine(tiie_swp_engine)
    npv_100mn = swap0.NPV()
    # Default DV01 (for 100mm)
    flat_dv01 = flat_DV01(frCrvs, banxico_TIIE28, npv_100mn,
                             start, maturity, notional, 
                             rate, typ, rule)
    dv01_100mn = flat_dv01/input_fxrate
    
    # Notional request given input
    if input_npvmxn == 0: # Notional given DV01 (USD)
        x_dv01_usd = float(input_dv01usd)
        y_notional = x_dv01_usd*notional/dv01_100mn
        lstk = ['Trade#', 'StartDate', 'EndDate', 'Notional_Inputs',
                'InputRate', 'FairRate', 'NPV_Inputs', 'InputDV01(USD)']
    elif input_dv01usd == 0: # Notional given NPV (MXN)
        x_npv = float(input_npvmxn)
        y_notional = x_npv*notional/npv_100mn
        lstk = ['Trade#', 'StartDate', 'EndDate', 'Notional_Inputs',
                'InputRate', 'FairRate', 'InputNPV', 'DV01(USD)_Inputs']
        
    # Type given input
    typ = int(np.sign(y_notional)*-1)
    rule = (df_trades.loc[i,df_trades.columns[10]] == 'Forward')*1
    
    # Instance
    swap = tiieSwap(start, maturity, abs(y_notional), 
                       ibor_tiie, rate, typ, rule)
    swap.setPricingEngine(tiie_swp_engine)
    swap_npv = swap.NPV()
    swap_swprate = swap.fairRate()
    # DV01
    flat_dv01 = flat_DV01(frCrvs, banxico_TIIE28, swap_npv,
                             start, maturity, abs(y_notional), 
                             rate, typ, rule)
    # Output
    lstv = [i,swap.startDate(),swap.maturityDate(),y_notional,
            rate*100,swap_swprate*100,swap_npv,flat_dv01/input_fxrate]
    res_dic = dict(zip(lstk,lstv))
    
    _res_df = pd.DataFrame([res_dic])
    _res_df = _res_df.set_index('Trade#',drop=True)
    
    return _res_df, swap

def eval_swap_fixings_banxico(swap, ibor_tiie):
    banxico_flag = False
    # Swap Cashflows Fixing Schedule
    swap_valuation_CF = get_CF_tiieSwap(swap, ibor_tiie)
    swap_CF = swap_valuation_CF.drop(['Date', 'Fix_Amt','Float_Amt'], axis=1)
    dt_swap_fixings = pd.to_datetime(swap_CF['Fixing_Date'].values)
    # Banxico Meetings Schedule
    banxico_df= pd.read_excel('TIIE_IRS_Trading.xlsm', 
                               'Banxico_Meeting_Dates')
    banxico_dates = pd.to_datetime(banxico_df['Meeting_Dates'].to_list())  
    # Crossover
    for bx_d in banxico_dates:
        bx_d_fix = (bx_d + timedelta(days=1)).strftime("%Y-%m-%d")
        if bx_d_fix in dt_swap_fixings:
            print ('\nCheck Banxico Meeting Date: ', bx_d.strftime("%Y-%m-%d"))
            banxico_flag = True
    # Cashflow Info
    isFirstLastCFIrr = int(swap_CF['Acc_Days'][0]) < 28 or \
        int(swap_CF['Acc_Days'][-1:]) < 28
    if isFirstLastCFIrr or banxico_flag == True:
        if isFirstLastCFIrr:
            print('Irregular Cashflows:')
        else:
            print('Cashflows:')
        print(swap_CF)
    return

def eval_swap_krr(i, df_trades, ql_settle_dt, brCrvs, base_npv, dic_data):
    # Get swap decoy
    start, maturity = start_end_dates_trading(i, df_trades, ql_settle_dt)
    input_notional = float(df_trades.loc[i,df_trades.columns[6]])
    input_rate = df_trades.loc[i,df_trades.columns[7]]
    notional = abs(input_notional)
    rate = float(input_rate)
    typ = int(np.sign(input_notional)*-1)
    rule = (df_trades.loc[i,df_trades.columns[10]] == 'Forward')*1

    # DV01 market tenor
    modNPV = {}    
    for tenor in brCrvs.keys():
        # new yieldcurves
        rytsDisc = ql.RelinkableYieldTermStructureHandle()
        rytsForc = ql.RelinkableYieldTermStructureHandle()
        discCrv, forcCrv = brCrvs[tenor]
        rytsDisc.linkTo(discCrv)
        rytsForc.linkTo(forcCrv)
        # disc-forc engines
        discEngine = ql.DiscountingSwapEngine(rytsDisc)
        ibor_tiie_krr = ql.IborIndex('TIIE',
                     ql.Period(13),
                     1,
                     ql.MXNCurrency(),
                     ql.Mexico(0),
                     ql.Following,
                     False,
                     ql.Actual360(),
                     rytsForc)
        swap_list = []
        
        swap_krr = tiieSwap(start, maturity, notional, ibor_tiie_krr, rate, 
                            typ, rule)     
        swap_krr.setPricingEngine(discEngine)  
        swap_list.append(swap_krr.NPV())
        modNPV[tenor] = swap_list
    # Mod NPV DataFrame    
    df_modNPV = pd.DataFrame(
        modNPV,
        index = [i],
        )
    # Bucket Risk DataFrame
    brTenors = dic_data['MXN_TIIE']['Tenor'].tolist()
    df_tenorDV01 = pd.DataFrame(None, index = [i])
    for tenor in brTenors:
        df_tenorp1 = df_modNPV[tenor+'+1']
        df_tenorm1 = df_modNPV[tenor+'-1']
        df_deltap1 = df_tenorp1 - base_npv
        df_deltam1 = df_tenorm1 - base_npv
        df_signs = np.sign(df_deltap1)
        df_tenorDV01[tenor] = \
            df_signs*(abs(df_deltap1)+abs(df_deltam1))/2
    return df_tenorDV01

def print_eval_output(i, df_output):
    # Output data
    lst_ttls = [df_output.index.name]+df_output.columns.tolist()
    start, maturity, volume, rate, swpRate, npv, flat_dv01usd = df_output.loc[i,]
    print('')
    print(f'{lst_ttls[0]}:\t {i}\n'+\
          f'{lst_ttls[1]}:\t {start}\n'+\
          f'{lst_ttls[2]}:\t {maturity}\n'+\
          f'{lst_ttls[3]}:\t {volume:,.0f}\n'+\
          f'{lst_ttls[4]}:\t {rate:.4f}\n'+\
          f'{lst_ttls[5]}:\t {swpRate:.4f}\n'+\
          f'{lst_ttls[6]}:\t {npv:,.0f}\n'+\
          f'{lst_ttls[7]}:\t {flat_dv01usd:,.0f}')
        
    return

def print_krr(swap_krr):
    global input_fxrate
    print('\nKRR DV01(USD):')
    print(f'Outright: {(swap_krr/input_fxrate).sum().sum():,.0f}')
    print((swap_krr/input_fxrate).sum().map('{:,.0f}'.format))
    return

def print_krr_group(krr_group, npv_group, krr_list):
    global input_fxrate
    # NPV Group
    npv_group_df = pd.DataFrame.from_dict(npv_group).T
    npv_group_df = npv_group_df.rename(columns={0: 'KRRG', 1: 'NPV'})
    # KRR Group
    krr_group.index = krr_list
    krr_group = ((krr_group.groupby(krr_group.index).sum())/input_fxrate).T
    pd.options.display.float_format = '{:,.0f}'.format
    # Output
    print('\n')
    print(r'------------------------------------------------'+\
          '----------------')
    print('KRR DV01 (USD) by Group:')
    print(r'------------------------------------------------'+\
          '----------------')
    print(krr_group)
    print('\n')

    for i in list(krr_group.columns):
        npv_group_df_i = npv_group_df.loc[npv_group_df['KRRG'] == i]
        npv_group_df_i = npv_group_df_i['NPV'].sum()
        sum_kkr_g = krr_group[i].sum()
        print(r'------------------------------------------------'+\
              '----------------')
        print('Group ' + str(i))
        print('Outright DV01 (USD): ', "{:,.0f}".format(sum_kkr_g))
        print('NPV (MXN): ', "{:,.0f}".format(npv_group_df_i))
        print(r'------------------------------------------------'+\
              '----------------\n')
    return

def eval_swprate_fillblanks(i, df_res, wb_pricing):
    # Swap Rate
    wb_pricing.range("A3").offset(i,8).value = \
        df_res['FairRate'].values[0]/100
    wb_pricing.range("A3").offset(i,8).color = clr_orange_output
    # Conditions
    isNotionalBlank = wb_pricing.range("A3").offset(i,7).value == None
    isDV01Blank = wb_pricing.range("A3").offset(i,10).value == None
    # Notional & DV01
    if isNotionalBlank and isDV01Blank:
        # Notional
        wb_pricing.range("A3").offset(i,7).value = \
            df_res['Notional_Inputs'].values[0]
        wb_pricing.range("A3").offset(i,7).color = clr_orange_output
        # Dv01
        dv01col = [s for s in df_res.columns.tolist() if 'DV01(USD)' in s][0]
        wb_pricing.range("A3").offset(i,10).value = \
            df_res[dv01col].values[0]
        wb_pricing.range("A3").offset(i,10).color = clr_orange_output
    elif isNotionalBlank:
        # Notional
        wb_pricing.range("A3").offset(i,7).value = \
            df_res['Notional_Inputs'].values[0]
        wb_pricing.range("A3").offset(i,7).color = clr_orange_output
    else:
        # DV01
        dv01col = [s for s in df_res.columns.tolist() if 'DV01(USD)' in s][0]
        wb_pricing.range("A3").offset(i,10).value = \
            df_res[dv01col].values[0]
        wb_pricing.range("A3").offset(i,10).color = clr_orange_output
    return

def eval_npv_dv01_fillblanks(i, df_res, wb_pricing):
    # NPV
    wb_pricing.range("A3").offset(i,9).value = \
        df_res['NPV_Inputs'].values[0]
    wb_pricing.range("A3").offset(i,9).color = clr_orange_output
    # DV01
    dv01col = [s for s in df_res.columns.tolist() if 'DV01(USD)' in s][0]
    wb_pricing.range("A3").offset(i,10).value = \
        df_res[dv01col].values[0]
    wb_pricing.range("A3").offset(i,10).color = clr_orange_output
    return 
###############################################################################
# Menu
###############################################################################
def displayMenu(ql_eval_dt, ql_settle_dt) -> int:
    print('\n---------------------------------------------------------------'+\
          '----------------------')
    print(f'Valuation Date: {ql_eval_dt}\n'+\
          f'Settle Date: {ql_settle_dt}')
    print(r'---------------------------------------------------------------'+\
          '----------------------\n')
    print('What do you want to do?\n'+\
          '\t\t 1) Pricing                7) TW Collapse\n'+\
          '\t\t 2) ShortEnd Pricing       8) End Session\n'+\
          '\t\t 3) Update Curves          9) Print Bstrpd Crvs\n'+\
          '\t\t 4) Fwds                  10) Cost Rates\n'+\
          '\t\t 5) Blotter               11) BO Rates\n'+\
          '\t\t 6) Corros pull           12) w5...\n')
    return int(input('\n\t\t option: '))
###############################################################################
# Main processess
###############################################################################
# Function to bypass US Holidays in MXNOIS Bootstrapping
def mxnois_when_USH(prev_date, crv_Z):
    period_file = min(len(crv_Z), 11650)
    # Schedule specs for DF dates
    effective_date = ql.Date(prev_date.day, prev_date.month, prev_date.year)
    period = ql.Period(period_file -1, ql.Days)
    termination_date = effective_date + period
    tenor = ql.Period(ql.Daily)
    calendar = ql.Mexico()
    business_convention = ql.Unadjusted
    termination_business_convention = ql.Following
    date_generation = ql.DateGeneration.Forward
    end_of_month = True
    
    # DF Schedule
    schedule = ql.Schedule(effective_date, termination_date, tenor, calendar,
                           business_convention, 
                           termination_business_convention, date_generation,
                           end_of_month)
    # DF dates
    dates = []
    for i, d in enumerate(schedule):
        dates.append(d)
    
    # List of DF from zero rates
    lstOIS_dfs = [1]
    valores_ois = crv_Z['VALOR'][:min(crv_Z.shape[0]-1,11649)]
    plazos_ois = crv_Z['PLAZO'][:min(crv_Z.shape[0]-1,11649)]
    lstOIS_dfs.extend(
        [1/(1 + r*t/36000) for (r, t) in zip(valores_ois, plazos_ois)])
    
    # MXNOIS Curve
    crvMXNOIS = ql.DiscountCurve(dates, lstOIS_dfs, ql.Actual360(), ql.Mexico())
    
    return crvMXNOIS
    

def proc_BuildCurves(dic_data, banxico_TIIE28):
    # Check for US Holiday
    isUSH = ql.UnitedStates(0).isHoliday(ql.Settings.instance().evaluationDate)
    # USD Curves
    crvUSDOIS = udf.btstrap_USDOIS(dic_data)
    crvUSDSOFR = udf.btstrap_USDSOFR(dic_data, crvUSDOIS)
    # Bootstrapping
    if isUSH:
        # Prev date
        qldate_prev = ql.Mexico().advance(ql.Settings.instance().evaluationDate,ql.Period(-1,ql.Days))
        dt_prevDt = qldate_prev.to_date()

        # Read prev built curve
        tmppath = r'\\tlaloc\cuantitativa\Fixed Income\Historical OIS TIIE'
        tmpfile = r'\OIS_'+dt_prevDt.strftime("%Y%m%d")+'.xlsx'
        crv_Z_MXNOIS = pd.read_excel(tmppath+tmpfile)
        crvMXNOIS = mxnois_when_USH(dt_prevDt, crv_Z_MXNOIS)
        # FlatRisk
        frCrvs = udf.crvFlatRisk_TIIE_isUSH(dic_data, crvMXNOIS)
        # BucketRisk
        brCrvs = udf.crvTenorRisk_TIIE_isUSH(dic_data, crvMXNOIS)
    else:
        crvMXNOIS = udf.btstrap_MXNOIS(dic_data, crvUSDSOFR, crvUSDOIS, 'SOFR')
        # FlatRisk
        frCrvs = udf.crvFlatRisk_TIIE(dic_data, crvUSDOIS, crvUSDSOFR)
        # BucketRisk
        brCrvs = udf.crvTenorRisk_TIIE(dic_data, crvUSDOIS, crvUSDSOFR, crvMXNOIS)
    
    # TIIE Curve, double-bootstrapped
    crvTIIE = udf.btstrap_MXNTIIE(dic_data, crvMXNOIS)

    ###########################################################################
    # Ibor Index
    ibor_tiie_crv = ql.RelinkableYieldTermStructureHandle()
    ibor_tiie_crv.linkTo(crvTIIE)
    ibor_tiie = ql.IborIndex('TIIE',
                 ql.Period(13),
                 1,
                 ql.MXNCurrency(),
                 ql.Mexico(),
                 ql.Following, # Moves fwd mty date even though changes month
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
    rytsMXNOIS = ql.RelinkableYieldTermStructureHandle()
    rytsMXNOIS.linkTo(crvMXNOIS)
    tiie_swp_engine = ql.DiscountingSwapEngine(rytsMXNOIS)
    return crvUSDOIS, crvUSDSOFR, crvMXNOIS, crvTIIE, \
            frCrvs, brCrvs, ibor_tiie, tiie_swp_engine
            
#
# FAST BUILDCURVES
# Only computes TIIE Curve
def proc_fastBuildCurves0(dic_data, ibor_tiie, crvUSDSOFR, crvUSDOIS, crvMXNOIS):
    # Curve Bootstrapping
    crvTIIE = udf.btstrap_MXNTIIE(dic_data, crvMXNOIS)
    
    # FlatRisk Curves
    frCrvs = udf.crvFlatRisk_TIIE(dic_data, crvUSDOIS, crvUSDSOFR)
    # BucketRisk Curves
    brCrvs = udf.crvTenorRisk_TIIE(dic_data, crvUSDOIS, crvUSDSOFR, crvMXNOIS)
   
    # Ibor Index Fwd Curve Update
    ibor_tiie.forwardingTermStructure = ql.RelinkableYieldTermStructureHandle(crvTIIE)
    
    return crvTIIE, frCrvs, brCrvs

# Only bootstraps TIIE Curves
def proc_fastBuildCurves(dic_data, banxico_TIIE28, crvUSDSOFR, crvUSDOIS):
    # Double-Curve Bootstrapping
    crvMXNOIS = udf.btstrap_MXNOIS(dic_data, crvUSDSOFR, crvUSDOIS, 'SOFR')
    crvTIIE = udf.btstrap_MXNTIIE(dic_data, crvMXNOIS)
    
    # FlatRisk Curves
    frCrvs = udf.crvFlatRisk_TIIE(dic_data, crvUSDOIS, crvUSDSOFR)
    # BucketRisk Curves
    brCrvs = udf.crvTenorRisk_TIIE(dic_data, crvUSDOIS, crvUSDSOFR, crvMXNOIS)
   
    # Ibor Index
    ibor_tiie_crv = ql.RelinkableYieldTermStructureHandle()
    ibor_tiie_crv.linkTo(crvTIIE)
    ibor_tiie = ql.IborIndex('TIIE',
                 ql.Period(13),
                 1,
                 ql.MXNCurrency(),
                 ql.Mexico(),
                 ql.Following,
                 False,
                 ql.Actual360(),
                 ibor_tiie_crv)
    
    # Ibor Index Fixings
    ibor_tiie.clearFixings()
    for h in range(len(banxico_TIIE28['fecha']) - 1):
        dt_fixing = pd.to_datetime(banxico_TIIE28['fecha'][h])
        ibor_tiie.addFixing(
            ql.Date(dt_fixing.day, dt_fixing.month, dt_fixing.year), 
            banxico_TIIE28['dato'][h+1]
            )
        
    # Discounting Curve    
    rytsMXNOIS = ql.RelinkableYieldTermStructureHandle()
    rytsMXNOIS.linkTo(crvMXNOIS)
    tiie_swp_engine = ql.DiscountingSwapEngine(rytsMXNOIS)
    
    return crvMXNOIS, crvTIIE, frCrvs, brCrvs, ibor_tiie, tiie_swp_engine

def proc_Pricing(ql_settle_dt, ibor_tiie, tiie_swp_engine, frCrvs, brCrvs,
                 banxico_TIIE28, dic_data, str_cwd, str_file, 
                 wb_pricing, rng_pricing):
    
    # ReInit Requests
    wb_pricing.range(rng_pricing).color = None
    isPrintRequestPX=str(
        input('\tPrint Ticket? (Yes/No): ')[0]).lower()=='y'
    isPrintRequestKRR=str(
        input('\tPrint KRR? (Yes/No): ')[0]).lower()=='y'
    
    # Read file
    ## Takes ~8s to read excel file
    #parameters_trades = pd.read_excel(str_cwd+r'\\'+str_file, 
    #                                  'Pricer', 
    #                                  skiprows=2).set_index('Trade#').fillna(0)
    ## Takes ~0.04s to read range; a more than 200-fold improvement
    cellstr1 = "A3"
    cellstr2 = wb_pricing.range(cellstr1).end('right').\
        address.replace('$','')[0]+\
            str(wb_pricing.range(cellstr1).end('down').row)
    parameters_trades = wb_pricing.range(cellstr1+':'+cellstr2).\
        options(pd.DataFrame, header=1, index=False, expand='table').value
    # Trades table
    parameters_trades = parameters_trades.set_index('Trade#').fillna(0)
    # Pricing trades table
    new_parameters_trades = parameters_trades[parameters_trades['valChck'] != 0]
    # Pfolio groups
    krr_group = pd.DataFrame(columns = list(dic_data['MXN_TIIE']['Tenor']))
    krr_list = []
    npv_group = {}
    # Retreive USDMXN
    global input_fxrate 
    #input_fxrate = pd.read_excel(str_cwd+r'\\'+str_file,'Pricer',
    #                             header=None).iloc[0,4]
    input_fxrate = wb_pricing.range('E1').value
    
    # Pricing Requests Processing
    for i,r in new_parameters_trades.iterrows():
        # Inputs
        j = np.where(parameters_trades.index == i)[0][0]
        input_krr_grp = parameters_trades.iloc[j, 11]        
        # Conditions
        isValCheck = str(r['valChck']).lower() == 'x' 
        isNPVDV01Blank = r['NPV_MXN'] == 0 and r['DV01_USD'] == 0
        isKRRChecked = str(r['krrChck']).lower() == 'x'
        isKRRAsked = isKRRChecked  or r['KRR_Group'] != 0
        isSwpRateAsked = r['Rate'] == 0 or r['Rate'] == ''
        #######################################################################
        ### CASE ASSESSMENT ###
        # SwapRate given inputs
        if isValCheck and isSwpRateAsked: # Case Rate Unknown
            # Evaluate missing fields: FairRate & Notional with default DV01
            df_res, swap = eval_swprate(i, parameters_trades, ql_settle_dt, 
                         ibor_tiie, tiie_swp_engine, frCrvs, banxico_TIIE28)
            # Fill missing fields
            parameters_trades.loc[i, 'Rate'] = df_res.loc[i,'FairRate']/100
            eval_swprate_fillblanks(i, df_res, wb_pricing)
        
        # NPV & DV01 given inputs
        elif isValCheck and isNPVDV01Blank: # Case NPV & DV01 Unknown
            # Evaluate for missing fields: NPV & DV01
            df_res, swap = eval_npv_dv01(i, parameters_trades, ql_settle_dt, 
                          ibor_tiie, tiie_swp_engine, frCrvs, banxico_TIIE28)
            # Fill missing fields
            eval_npv_dv01_fillblanks(i, df_res, wb_pricing)
                    
        # Notional given inputs
        elif isValCheck and r['Notional_MXN'] == 0: # Case Notional Unknown
            # Evaluate for missing fields
            df_res, swap = eval_notional(i, parameters_trades, ql_settle_dt, 
                                         ibor_tiie, tiie_swp_engine, frCrvs, 
                                         banxico_TIIE28)   
            # Fill missing fields: Notional & (Fair)Rate
            eval_swprate_fillblanks(i, df_res, wb_pricing)
        else:
            continue
        ####################################################################### 
        # Swap NPV by group 
        npvcol = [s for s in df_res.columns.tolist() if 'NPV' in s]
        npv_group[i] = [input_krr_grp, df_res.loc[i,npvcol[0]]]
        
        # Print request
        if isPrintRequestPX:
            print_eval_output(i, df_res)
               
        # Swap FixingsXBanxico
        #eval_swap_fixings_banxico(swap, ibor_tiie)
    
        # KRR
        if isPrintRequestKRR and isKRRAsked:
            # Swap Risk Sens By Tenor
            swap_krr = eval_swap_krr(i, parameters_trades, ql_settle_dt, 
                                     brCrvs, df_res.loc[i,npvcol[0]], dic_data)
            # KRR Print
            if isKRRChecked:
                #print_krr(swap_krr)
                wb_pricing.range('S'+str(int(i+3))).value = \
                    swap_krr.values.reshape(-1,)/input_fxrate
            # Grouped KRR      
            if input_krr_grp != 0:
                krr_list.append(int(input_krr_grp))
                krr_group = pd.concat([krr_group, swap_krr])
        
    # KRR Group
    if not krr_group.empty:
        print_krr_group(krr_group, npv_group, krr_list)

# Function to compute vanilla rates for given swap rates
def proc_CostRates(ql_settle_dt, ibor_tiie, tiie_swp_engine, frCrvs, brCrvs,
                 banxico_TIIE28, dic_data, str_cwd, str_file, 
                 wb_pricing, wb_tiie):
    
    # custom module
    main_path = '//TLALOC/Cuantitativa/Fixed Income/TIIE IRS Valuation Tool/'+\
        'Main Codes/Portfolio Management/OOP codes/'
    sys.path.append(main_path)
    # Copy data and format types
    opt_dic_data = {}
    for name in dic_data.keys():
        if name != 'USD_SOFR':
            opt_dic_data[name] = dic_data[name].astype({'Period':'int'})
        else:
            opt_dic_data[name] = dic_data[name].copy()
            opt_dic_data[name].loc[[0]+list(range(6,21))] = \
                dic_data[name].loc[[0]+list(range(6,21))].astype({'Period':'int'})   
        
    # Filter out unused tenors
    opt_dic_data['USD_OIS'].drop(opt_dic_data['USD_OIS'].tail(1).index, inplace=True)
    opt_dic_data['USD_SOFR'].drop(opt_dic_data['USD_SOFR'].tail(2).index, inplace=True)
        
    import curve_funs as cf
    # Bootstrap disc/forc curves
    curves = cf.mxn_curves(opt_dic_data)
    
    # Trades range
    cellstr1 = "A3"
    cellstr2 = wb_pricing.range(cellstr1).end('right').\
        address.replace('$','')[0]+\
            str(wb_pricing.range(cellstr1).end('down').row)
    parameters_trades = wb_pricing.range(cellstr1+':'+cellstr2).\
        options(pd.DataFrame, header=1, index=False, expand='table').value
    parameters_trades = parameters_trades.set_index('Trade#').fillna(0)
    
    # Range to eval if calc costs
    int_lRow = wb_pricing.range(cellstr1).end('down').row
    rng_calc_check = wb_pricing.range("AH4:AH"+str(int_lRow)).value
    
    # Clear contents
    wb_pricing.range("AI4:AV"+str(int_lRow)).clear_contents()
    
    # Checkd trades table
    parameters_trades.insert(1,'CalcCost',rng_calc_check)
    ididx_check = (parameters_trades['valChck'] != 0)*\
        (parameters_trades['CalcCost'] == 'x')
    new_parameters_trades = parameters_trades[ididx_check]
    new_parameters_trades = new_parameters_trades.drop('CalcCost', axis=1)
    parameters_trades = parameters_trades.drop('CalcCost', axis=1)
    
    # USDMXN
    global input_fxrate 
    input_fxrate = wb_pricing.range('E1').value
    
    # TIIE market
    bidoffer = wb_tiie.range('G1:I15').\
        options(pd.DataFrame, header=1, index=False, expand='table').value
    bidoffer['Tenor'] = wb_tiie.range('B2:B15').value
    bidoffer = bidoffer[['Tenor','Bid','Offer']]
    
    # BidOffer bounds
    df_bounds = bidoffer[['Bid','Offer']].mean(axis=1).apply(lambda x: (x,x))
    df_bounds.index = bidoffer['Tenor']
    
    # Objective function
    from scipy.optimize import minimize
    def fair_rate_l2(tiie_array: np.ndarray, start, end, notional, fixed_rate) -> float:
        
        dftiie = opt_dic_data['MXN_TIIE'].copy()
        dftiie['Quotes'] = tiie_array
        curves.change_tiie(dftiie)
        swap = cf.tiieSwap(start, end, notional, fixed_rate, curves)
        swap_rate = swap.fairRate()*100
        
        return 100*(target_rate - swap_rate)**2
    
    # Loop through each aceptable trade
    for i,r in new_parameters_trades.iterrows():
        # Reset bounds
        df_bounds = bidoffer[['Bid','Offer']].mean(axis=1).apply(lambda x: (x,x))
        df_bounds.index = bidoffer['Tenor']
        
        # Get NPV
        df_res, swap = eval_npv_dv01(i, parameters_trades, ql_settle_dt, 
                      ibor_tiie, tiie_swp_engine, frCrvs, banxico_TIIE28)
        # KRR
        swap_krr = eval_swap_krr(i, parameters_trades, ql_settle_dt, 
                                 brCrvs, df_res.loc[i, 'NPV_Inputs'], dic_data)
        # Most contributing buckets
        swap_krr_mcb = swap_krr.T[swap_krr.T.abs()/swap_krr.T.abs().sum() > 0.15]
        idx_tenors = swap_krr_mcb.index[(~swap_krr_mcb.isna()).to_numpy().reshape(-1,)]
        
        # Optimization params
        notional = r['Notional_MXN']
        start, end = start_end_dates_trading(i, parameters_trades, ql_settle_dt)
        initial_rate = swap.fairRate()*100
        target_rate = r['Rate']*100
        target_tenors = bidoffer[bidoffer['Tenor'].apply(lambda x: x in idx_tenors.to_list())]
        tenors_bounds = target_tenors.\
            apply(lambda x: (x['Bid']-0.15,x['Offer']+0.15), axis=1)
        df_bounds.loc[idx_tenors] = tenors_bounds.values
        bounds = df_bounds.to_list()
        
        # Curve mids input
        mid_quotes = bidoffer[['Bid','Offer']].mean(axis=1).to_numpy()
        df_quotes = dic_data['MXN_TIIE'].copy()
        df_quotes['Quotes'] = mid_quotes
        curves.change_tiie(df_quotes)
        swap = cf.tiieSwap(start.to_date(), end.to_date(), notional, initial_rate, curves)
        
        # Optimization
        fixed_args = (start.to_date(), end.to_date(), notional, initial_rate)
        optimal_rates = minimize(fair_rate_l2, mid_quotes, args=fixed_args,
                                 method='L-BFGS-B', bounds=bounds,
                                 options = {'maxiter': 400})
        # Optimal inputs/outputs
        optimal_tiies = optimal_rates.x
        optimal_dftiie = dic_data['MXN_TIIE'].copy()
        optimal_dftiie['Quotes'] = optimal_tiies
        curves.change_tiie(optimal_dftiie)
        swap = cf.tiieSwap(start.to_date(), end.to_date(), notional, initial_rate, curves)
        
        # Cost rates
        df_optimal_inputs = dic_data['MXN_TIIE'][['Tenor']].copy()
        df_optimal_inputs.insert(1,'Mid',mid_quotes)
        df_optimal_inputs.insert(2,'Opt',optimal_tiies)
        df_optimal_inputs = df_optimal_inputs.set_index('Tenor')
        df_optimal_output = df_optimal_inputs.copy()[['Opt']]
        df_optimal_output['Opt'] = ''
        df_optimal_output.loc[idx_tenors,'Opt'] = df_optimal_inputs.loc[idx_tenors, 'Opt']
        
        # Print
        wb_pricing.range("AI"+str(int(i+3))).value = df_optimal_output.T.values
    
    return None

# Function to compute bid/offer rates
def proc_BidOffer(ql_settle_dt, ibor_tiie, tiie_swp_engine, frCrvs, brCrvs,
                 banxico_TIIE28, dic_data, str_cwd, str_file, 
                 wb_pricing, wb_tiie):
    
    # custom module
    main_path = '//TLALOC/Cuantitativa/Fixed Income/TIIE IRS Valuation Tool/'+\
        'Main Codes/Portfolio Management/OOP codes/'
    sys.path.append(main_path)
    # Copy data and format types
    opt_dic_data = {}
    for name in dic_data.keys():
        if name != 'USD_SOFR':
            opt_dic_data[name] = dic_data[name].astype({'Period':'int'})
        else:
            opt_dic_data[name] = dic_data[name].copy()
            opt_dic_data[name].loc[[0]+list(range(6,21))] = \
                dic_data[name].loc[[0]+list(range(6,21))].astype({'Period':'int'})   
        
    # Filter out unused tenors
    opt_dic_data['USD_OIS'].drop(opt_dic_data['USD_OIS'].tail(1).index, inplace=True)
    opt_dic_data['USD_SOFR'].drop(opt_dic_data['USD_SOFR'].tail(2).index, inplace=True)
        
    import curve_funs as cf
    # Bootstrap disc/forc curves
    curves = cf.mxn_curves(opt_dic_data)
    
    # Trades range
    cellstr1 = "A3"
    cellstr2 = wb_pricing.range(cellstr1).end('right').\
        address.replace('$','')[0]+\
            str(wb_pricing.range(cellstr1).end('down').row)
    parameters_trades = wb_pricing.range(cellstr1+':'+cellstr2).\
        options(pd.DataFrame, header=1, index=False, expand='table').value
    parameters_trades = parameters_trades.set_index('Trade#').fillna(0)
    
    # Range to eval if calc costs
    int_lRow = wb_pricing.range(cellstr1).end('down').row
    rng_calc_check = wb_pricing.range("AX4:AX"+str(int_lRow)).value
    
    # Clear contents
    wb_pricing.range("BM4:BO"+str(int_lRow)).clear_contents()
    
    # Checkd trades table
    parameters_trades.insert(1,'CalcBO',rng_calc_check)
    ididx_check = (parameters_trades['valChck'] != 0)*\
        (parameters_trades['CalcBO'] == 'x')
    new_parameters_trades = parameters_trades[ididx_check]
    new_parameters_trades = new_parameters_trades.drop('CalcBO', axis=1)
    parameters_trades = parameters_trades.drop('CalcBO', axis=1)
    
    # USDMXN
    global input_fxrate 
    input_fxrate = wb_pricing.range('E1').value
    
    # TIIE market
    bidoffer = wb_tiie.range('G1:I15').\
        options(pd.DataFrame, header=1, index=False, expand='table').value
    bidoffer['Tenor'] = wb_tiie.range('B2:B15').value
    bidoffer = bidoffer[['Tenor','Bid','Offer']]
    mkt_mid = bidoffer[['Bid','Offer']].mean(axis=1); mkt_mid.index=bidoffer['Tenor']
    
    # Loop through each aceptable trade
    for i,r in new_parameters_trades.iterrows():        
        # Swap fair rate
        df_res, swap = eval_npv_dv01(i, parameters_trades, ql_settle_dt, 
                      ibor_tiie, tiie_swp_engine, frCrvs, banxico_TIIE28)
        # KRR
        swap_krr = eval_swap_krr(i, parameters_trades, ql_settle_dt, 
                                 brCrvs, df_res.loc[i, 'NPV_Inputs'], dic_data)
        # Bucket spread
        tmptnr = wb_pricing.range("AY3:BL3").value
        tmpsprd = wb_pricing.range(f"AY{int(i+3)}:BL{int(i+3)}").value
        df_bcktsprd = pd.DataFrame(tmpsprd, 
                     index = tmptnr, 
                     columns=[i]).fillna(0)*swap_krr.T.apply(np.sign)/100
        
        # Mkt mids with BO
        mod_mkt_mid = mkt_mid.rename(i).to_frame() + df_bcktsprd
        
        # Curve mids input
        mid_quotes = mod_mkt_mid.to_numpy()
        df_quotes = dic_data['MXN_TIIE'].copy()
        df_quotes['Quotes'] = mid_quotes
        curves.change_tiie(df_quotes)
        
        # Swap inputs
        notional = r['Notional_MXN']
        start, end = start_end_dates_trading(i, parameters_trades, ql_settle_dt)
        # BO Swap
        swap_bo = cf.tiieSwap(start.to_date(), end.to_date(), 
                           notional, swap.fairRate()*100, curves)
        
        # BO spread
        sprd_bo = abs(10000*(swap_bo.fairRate() - swap.fairRate()))
        tmpBid = (swap.fairRate() - sprd_bo/10000)
        tmpOffer = (swap.fairRate() + sprd_bo/10000)
        
        # Print
        wb_pricing.range(f"BM{int(i+3)}").value = tmpBid
        wb_pricing.range(f"BN{int(i+3)}").value = tmpOffer
        wb_pricing.range(f"BO{int(i+3)}").value = sprd_bo*2
    
    return None

# Function to price collapse from tw
def proc_PriceCollpse_tw(ql_settle_dt, ibor_tiie, tiie_swp_engine, frCrvs, 
                         brCrvs, banxico_TIIE28, dic_data, str_cwd, str_file, 
                         wb, rng_pricing):
    
    # Read file
    wb_ctw = wb.sheets('Collapse_tw')
    cellstr1 = "A1"
    cellstr2 = wb_ctw.range(cellstr1).end('down').row
    parameters_trades = wb_ctw.range("A1:D"+str(cellstr2)).\
        options(pd.DataFrame, header=1, index=False).value

    # Pricing trades table
    new_parameters_trades = parameters_trades[~parameters_trades['Rate'].isna()]
    
    # Pfolio groups
    krr_group = pd.DataFrame(columns = list(dic_data['MXN_TIIE']['Tenor']))
    
    # Retreive USDMXN
    global input_fxrate 
    input_fxrate = wb.sheets('Pricer').range('E1').value
    
    # Pricing Requests Processing
    wb_ctw.range("E:F").clear_contents()
    df_res = pd.DataFrame()
    for j in range(4): parameters_trades.insert(0,j,0)
    for j in range(2): parameters_trades.insert(len(parameters_trades.columns),'C'+str(j),0)
    parameters_trades['Dt_Gen'] = 'Backward'
    for i,r in new_parameters_trades.iterrows():
        # NPV & DV01 given inputs: Evaluate for NPV & DV01
        # Specs
        start = ql.DateParser.parseFormatted(r['Start_Date'].strftime('%Y-%m-%d'),'%Y-%m-%d')
        maturity = ql.DateParser.parseFormatted(r['End_Date'].strftime('%Y-%m-%d'),'%Y-%m-%d')
        input_notional = float(r['Notional_MXN'])
        input_rate = r['Rate']
        notional = abs(input_notional)
        rate = float(input_rate)
        typ = int(np.sign(input_notional)*-1)
        rule = 0 # Backward
        # Instance
        swap = tiieSwap(start, maturity, notional, ibor_tiie, rate, typ, rule)
        swap.setPricingEngine(tiie_swp_engine)
        # Swap val
        swap_npv = swap.NPV()

        # Swap risk 
        flat_dv01 = flat_DV01(frCrvs, banxico_TIIE28, swap_npv, start, 
                              maturity, notional, rate, typ, rule)
        # Swap priced des
        res_dic = {'Trade#': i, 'StartDate': swap.startDate(), 
         'EndDate': swap.maturityDate(), 'InputNotional_MXN': -1*typ*notional, 
         'InputRate': rate*100, 'NPV_Inputs': swap_npv, 
         'DV01(USD)_Inputs': flat_dv01/input_fxrate}
        _res_df = pd.DataFrame([res_dic])
        _res_df = _res_df.set_index('Trade#',drop=True)
        df_res = pd.concat([df_res, _res_df])
        
        # NPV 
        npvcol = [s for s in df_res.columns.tolist() if 'NPV' in s]
        
        # Swap Risk Sens By Tenor 
        swap_krr = eval_swap_krr(i, parameters_trades, ql_settle_dt, 
                                 brCrvs, df_res.loc[i,npvcol[0]], dic_data)
        # Grouped KRR      
        krr_group = pd.concat([krr_group, swap_krr])
        
    # Print Swap NPV & DV01
    wb_ctw.range("E1").value = ['NPV', 'DV01']
    wb_ctw.range("E2").value = df_res[['NPV_Inputs', 'DV01(USD)_Inputs']].values
    
    ####################################################################### 
    # Pfolio Output
    # NPV 
    print(r'----------------------------------------------------------------')
    print("PFOLIO COLLAPSE")
    print(r'----------------------------------------------------------------')
    print(f"\tNPV: {df_res['NPV_Inputs'].sum():,.0f}")
    print(f"\tOutright DV01: {df_res['DV01(USD)_Inputs'].sum():,.0f}")
    print(f"\tAbs Fee: {df_res['DV01(USD)_Inputs'].abs().sum()*0.5*input_fxrate:,.0f} MXN")
    # KRR
    tmpKRR = krr_group.sum()/input_fxrate
    print(r'----------------------------------------------------------------')
    print('KRR DV01 (USD):')
    print(tmpKRR.apply(lambda x: '{:,.0f}'.format(x)))
    print('\n')


# Function to valuate trades in day blotter
def proc_Pricing_Blotter(ql_settle_dt, ibor_tiie, tiie_swp_engine, 
                         wb_pricing, brCrvs, dic_data):
    # Read table of trades
    cellstr1 = "A1"
    cellstr2 = wb_pricing.range(cellstr1).end('right').\
        address.replace('$','')[0]+\
            str(wb_pricing.range(cellstr1).end('down').row)
    parameters_trades = wb_pricing.range(cellstr1+':'+cellstr2).\
        options(pd.DataFrame, header=1, index=False, expand='table').value
    parameters_trades = parameters_trades.set_index('Trade_#').fillna(0)    
    cols = wb_pricing.range('R1:AE1').value
    # Params Dataframe adj
    ptc = parameters_trades.columns.tolist().index('Start_Date')
    if ptc != 4:
        cols2Add = 4 - ptc
        for i in range(cols2Add):
            parameters_trades.insert(0,str(i+1),None)
        
    # Pricing Requests Processing
    df_npv = pd.DataFrame(index = parameters_trades.index,columns=['npv'])
    df_bRisk = pd.DataFrame(index = parameters_trades.index,columns=cols)
    for i,r in parameters_trades.iterrows():        
        # NPV given inputs
        # Swap parameters
        notional = abs(r['Notional_MXN'])
        rate = float(r['Rate'])
        typ = int(np.sign(r['Notional_MXN'])*-1)
        rule = (r['Date_Generation'] == 'Forward')*1
        
        # Swap
        eff = ql.Date(r['Start_Date'].day,r['Start_Date'].month,r['Start_Date'].year)
        mty = ql.Date(r['End_Date'].day,r['End_Date'].month,r['End_Date'].year)
        swap = tiieSwap(eff, mty, notional, ibor_tiie, rate, typ, rule)
        swap.setPricingEngine(tiie_swp_engine)
        
        # Swap val
        swap_npv = swap.NPV()
        df_npv.loc[i,'npv'] = swap_npv
        #wb_pricing.range("A1").offset(i,15).value = swap_npv
        # Swap risk
        df_risk = eval_swap_krr(i, parameters_trades, ql_settle_dt, 
                                 brCrvs, swap_npv, dic_data)
        df_bRisk.loc[i,:] = df_risk.values
        
    wb_pricing.range("P2").value = df_npv.values
    wb_pricing.range("R2").value = df_bRisk.values
        
    return None
        
def proc_displayCF(wb_pricing, ql_settle_dt, ibor_tiie, tiie_swp_engine):
    # Read range
    cellstr1 = "A3"
    cellstr2 = wb_pricing.range(cellstr1).end('right').\
        address.replace('$','')[0]+\
            str(wb_pricing.range(cellstr1).end('down').row)
    params = wb_pricing.range(cellstr1+':'+cellstr2).\
        options(pd.DataFrame, header=1, index=False, expand='table').value
    params = params.set_index('Trade#').fillna(0)
    # Filter trades
    new_params = params[params['valChck'] != 0]
    new_params = new_params[new_params['Cashflows'] != 0]
    
    # CF Display for each trade
    dic_trade_cf = {}
    for i,r in new_params.iterrows():
        global frCrvs 
        global banxico_TIIE28
        # Conditions
        isSwpRateBlnk = r['Rate'] == 0 or r['Rate'] == ''
        isNotionalBlnk = r['Notional_MXN'] == 0 or r['Notional_MXN'] == ''
        # Date Params
        start, maturity = start_end_dates_trading(i, new_params, 
                                                     ql_settle_dt)
        # Rate Param
        if isSwpRateBlnk:
            df_res, swap = eval_swprate(i, new_params, ql_settle_dt, 
                         ibor_tiie, tiie_swp_engine, frCrvs, banxico_TIIE28)
            swap_rate = swap.fairRate()
        else:
            swap_rate = r['Rate']
        # Notional Param
        if isNotionalBlnk:
            notional = +1e9
        else:
            notional = r['Notional_MXN']
        # Swap RemParam
        typ = int(np.sign(notional)*-1)
        rule = (r['Date_Generation'] == 'Forward')*1
        swap = tiieSwap(start, maturity, abs(notional), 
                        ibor_tiie, swap_rate, typ, rule)
        # Swap CF Pmt Structure
        swap_cfOTR = get_CF_tiieSwapOTR(swap,ibor_tiie,ql_settle_dt)
        dic_trade_cf[i] = swap_cfOTR
    
    return dic_trade_cf  

def proc_ShortEndPricing(crvMXNOIS, crvTIIE, wb):
    # Move to sheet
    wb_stir = wb.sheets('shortEndPricing')
    wb_stir.activate()
    # Read dates data
    df_StEndDt = pd.DataFrame(wb_stir.range('Q3:R28').value,
                              columns = wb_stir.range('Q2:R2').value)
    # Fwd & Disc Rates
    fwdRates = []
    discF = []
    for i,r in df_StEndDt.iterrows():
        start, end = r
        fwdRates.append(crvTIIE.forwardRate(ql.Date().from_date(start), 
                                            ql.Date().from_date(end),
                                            ql.Actual360(),
                                            ql.Simple).rate())
        discF.append(crvMXNOIS.discount(ql.Date().from_date(end)))
    df_StEndDt['FltRate'] = fwdRates
    df_StEndDt['DF'] = discF
    
    # Update Rates
    wb_stir.range('T3:T28').value = df_StEndDt[['FltRate']].values
    wb_stir.range('Z3:Z28').value = df_StEndDt[['DF']].values
    
def proc_ShortEndPricing_byMPC(crvTIIE, wb):
    # Monte to sheet
    wb_stir = wb.sheets('shortEndPricing')
    wb_stir.activate()
    # Read MPC dates
    mpcdates = wb_stir.range("B46:B62").value
    # Get Fwd TIIE28 Rates
    mx_cal = ql.Mexico()
    lst_ftiie28 = []
    for mtngdate in mpcdates:
        qldate = ql.Date(mtngdate.day, mtngdate.month, mtngdate.year)
        stdt = mx_cal.advance(qldate,1,ql.Days)
        eddt = stdt + ql.Period('28D')
        lst_ftiie28.append(
            [crvTIIE.forwardRate(stdt,eddt,ql.Actual360(),ql.Simple).rate()])
    # Update FwdRates
    wb_stir.range('C46:C62').value = lst_ftiie28
    
# Function to pull inputs for curve bootstrapping
def proc_UpdateCurves_Inputs(input_sheets, str_cwd, str_file, wb):
    # Pull input data by curve type
    dic_data = {}
    for sheet in input_sheets:
        tmp_sheet = wb.sheets(sheet)
        end_row = tmp_sheet.range('A1').end('down').row
        tmpxldata = tmp_sheet.range('A1:E'+str(end_row)).\
            options(pd.DataFrame, header=1, index=False, expand='table').value
        dic_data[sheet] = tmpxldata

    return dic_data

def proc_UpdateCurves(dic_data, banxico_TIIE28):
    return proc_BuildCurves(dic_data, banxico_TIIE28)

def proc_fastUpdateCurves(dic_data, banxico_TIIE28, crvUSDSOFR, crvUSDOIS):
    return proc_fastBuildCurves(dic_data, banxico_TIIE28, crvUSDSOFR, crvUSDOIS)

def proc_CarryCalc(ibor_tiie, tiie_swp_engine, str_cwd, str_file, wb):
    # Read params
    params_carry = pd.read_excel(str_cwd+r'\\'+str_file, 'Carry', header=None,
                                 usecols='B:C', nrows=5).dropna().drop(3)
    params_carry[2] = pd.to_datetime(params_carry[2])
    params_carry.reset_index(drop=True, inplace=True)
    carry_shiftL = pd.read_excel(str_cwd+r'\\'+str_file, 'Carry', header=None,
                                 usecols='C', skiprows=3,nrows=1).iloc[0,0]
    # Read trades
    params_carry_trades = pd.read_excel(str_cwd+r'\\'+str_file, 'Carry',
                                 usecols='E:H', nrows=36, skiprows=3)
    # Selected Trades
    params_carry_trades = params_carry_trades[
        ~params_carry_trades[['Start_Tenor','Period_Tenor']].isna().all(axis=1)]
    params_carry_trades['Start_Tenor'] = \
        params_carry_trades['Start_Tenor'].fillna(0)
    # Start Dates
    dt_sttl = params_carry.loc[1,2]
    ql_sttl = ql.Date(dt_sttl.day,dt_sttl.month,dt_sttl.year)
    lst_stdt, lst_eddt, lst_swprate = [], [], []
    for i,r in params_carry_trades.iterrows():
        ql_stdt = ql_sttl+ql.Period(int(r['Start_Tenor']*28),ql.Days)
        lst_stdt.append(ql_stdt)
        ql_eddt = ql_stdt+ql.Period(int(r['Period_Tenor']*28),ql.Days)
        lst_eddt.append(ql_eddt)
        swp0 = tiieSwap(ql_stdt, ql_eddt, 1e9, ibor_tiie, 0.04, -1, 0)
        swp0.setPricingEngine(tiie_swp_engine)
        lst_swprate.append(swp0.fairRate())
    # Swap Rates
    df_trades = params_carry_trades.\
        merge(pd.DataFrame([lst_stdt,lst_eddt,lst_swprate]).T,
              left_index=True, right_index=True)
    df_trades.columns = params_carry_trades.columns.to_list()+\
        ['Start', 'Mty','Rate']
    # Roll
    dt_hrzn = params_carry.loc[2,2]
    ql_hrzn = ql.Date(dt_hrzn.day,dt_hrzn.month,dt_hrzn.year)
    lst_roll = []
    for i,r in df_trades.iterrows():
        if r['Start'] == ql_sttl: # ErodedSwap vs FwdEroded
            period = int(r['Period_Tenor'] - carry_shiftL)*28
            if period <= 0:
                lst_roll.append(0)
                continue
            start = ql_hrzn
            swp1 = tiieSwap(ql_sttl, 
                               ql_sttl+ql.Period(period,ql.Days), 1e9, 
                               ibor_tiie, r['Rate'], -1, 0)
            swp2 = tiieSwap(start, r['Mty'], 1e9, 
                               ibor_tiie, r['Rate'], -1, 0)
            swp1.setPricingEngine(tiie_swp_engine)
            swp2.setPricingEngine(tiie_swp_engine)
            lst_roll.append(1e4*(swp1.fairRate() - swp2.fairRate())*-1)
        else:
            if int(r['Start_Tenor']-carry_shiftL) <0:
                   lst_roll.append(0)
                   continue
            period = int(r['Period_Tenor'])*28
            start = ql_sttl + ql.Period(
                int(r['Start_Tenor']-carry_shiftL)*28,ql.Days)
            maturity = start + ql.Period(period,ql.Days)
            swp1 = tiieSwap(start,maturity, 1e9, ibor_tiie, 
                               r['Rate'], -1, 0)
            swp1.setPricingEngine(tiie_swp_engine)
            lst_roll.append(1e4*(swp1.fairRate() - r['Rate'])*-1)
    
    df_trades['CarryRoll'] = lst_roll
    wb.sheets('Carry').range("I5").value = df_trades[['CarryRoll']].values

# In[]
###############################################################################
# DATA EXTRACTION FROM PDFs
import pdfplumber

# Function to pull swaps closing prices from REMATE pdf
def remate_closes(wb, evaluation_date, str_wbSheet='MXN_TIIE'):
    """
    Fills Remate closes in TIIE_IRS_Data Excel file.

    Parameters
    ----------
    wb : xw.Book
        TIIE_IRS_Data Excel file.
    evaluation_date : datetime
        Date of evaluation.
    tiie_28_yst : float
        Last TIIE28.

    Returns
    -------
    None

    """
    
    # Look for pdf file with Remate closes
    strpath = r'U:\Fixed Income\File Dump\Valuaciones' # r'\\tlaloc\cuantitativa\Fixed Income\File Dump\Valuaciones'
    strname = r'\REMATE CLOSING PRICES FOR '
    file_path = strpath + strname
    yesterday_ql = ql.Mexico().advance(ql.Date().from_date(evaluation_date), 
                                       ql.Period(-1, ql.Days))
    file = file_path + yesterday_ql.to_date().strftime('%m%d%Y') + '.pdf'
    print(f'Fetching Remate closings from {yesterday_ql}')
    try:
        with pdfplumber.open(file) as pb:
            text = pb.pages[0].extract_text()
    except:
        print('Remate file not found.')
        return None
        
    renglones = text.split('\n')
    
    # Start row
    irs_r = renglones.index([i for i in renglones 
                             if 'SOFR Basis Swaps' in i][0])
    
    # End row
    options_r = renglones.index([i for i in renglones if 'Vols' in i][0])    
    
    cols=['Tenor', 'Bid', 'Offer', 'Chg']
    irs_df = pd.DataFrame(columns=cols)

    for r in range(irs_r+2, options_r):
        datos = renglones[r].split(' ')
        tenor = datos[0]
        bid = datos[1]
        offer = datos[3]
        chg = datos[4]
        irs_df_a = pd.DataFrame({'Tenor': [tenor], 'Bid': [bid], 'Offer': [offer],
                                 'Chg': [chg]})
        irs_df = pd.concat([irs_df, irs_df_a], ignore_index=True)

    irs_df['Mid'] = (irs_df['Bid'].astype(float) + irs_df['Offer'].astype(float))/2
    
    out_sheet = wb.sheets(str_wbSheet)
    out_sheet.range('N3').value = irs_df[['Mid']].values