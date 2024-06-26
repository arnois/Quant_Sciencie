# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17

@author: JArnulf QC (arnulf.q@gmail.com)

User-defined functions module for TIIE Curve Bootsrapping
"""
###############################################################################
# MODULES
###############################################################################
import QuantLib as ql
import numpy as np
import pandas as pd
from datetime import timedelta
# Working directory
#import os, sys
#str_cwd = '\\\\tlaloc\\cuantitativa\\'+\
#    'Fixed Income\\TIIE IRS Valuation Tool\\Arnua\\'
#os.chdir(str_cwd)
#sys.path.append(str_cwd)
import udf_TIIE_CurveCreate as udf
###############################################################################
# PORTFOLIO FEATURES
###############################################################################
# Swaps pfolio construction by data-import
def setPfolio_tiieSwps(str_posswps_file):
    # posswaps file import
    ## str_posswps_file = r'E:\posSwaps\PosSwaps20230213.xlsx'
    df_posSwps = pd.read_excel(str_posswps_file, 'Hoja1')
    
    # posswaps filter columns
    lst_selcols = ['swp_usuario', 'swp_ctrol', 'swp_fecop', 'swp_monto', 
                   'swp_fec_ini', 'swp_fec_vto', 'swp_fec_cor', 'swp_val_i_pa',
                   'swp_val_i', 'swp_serie', 'swp_emisora', 'swp_pzoref']
    lst_selcols_new = ['BookID','TradeID','TradeDate','Notional',
                       'StartDate','Maturity','CpDate','RateRec',
                       'RatePay','PayIndex','RecIndex','CouponReset']
    df_posSwps = df_posSwps[lst_selcols]
    df_posSwps.columns = lst_selcols_new
    df_posSwps[['RecIndex','PayIndex']] = df_posSwps[['RecIndex',
                                        'PayIndex']].replace(' ','',regex=True)
    df_tiieSwps = df_posSwps[df_posSwps['RecIndex'].isin(['TIIE28','TF.MN.'])]
    df_tiieSwps = df_tiieSwps[df_tiieSwps['CouponReset'] == 28]
    df_tiieSwps['FxdRate'] = df_tiieSwps['RateRec'] + df_tiieSwps['RatePay']
    df_tiieSwps['SwpType'] = -1
    df_tiieSwps['SwpType'][df_tiieSwps['RecIndex']=='TF.MN.'] = 1
    df_tiieSwps = df_tiieSwps.drop(['RecIndex','PayIndex',
                                    'CouponReset','RatePay','RateRec'], 
                     axis=1)
    df_tiieSwps[['TradeDate','StartDate',
                 'Maturity','CpDate']] = df_tiieSwps[['TradeDate','StartDate',
                 'Maturity','CpDate']].apply(lambda t: 
                                             pd.to_datetime(t,format='%Y%m%d'))
    df_tiieSwps = df_tiieSwps.reset_index(drop=True)
    # SPOTTING NON-IRREGULAR-TURNED-IRREGULAR CPNS
    # schdl generation rule
    endOnHoliday = {}
    for i,row in df_tiieSwps.iterrows():
        bookid, tradeid, tradedate, notnl, stdt, mty, cpdt, r, swptyp = row
        mod28 = (mty - stdt).days%28
        omty = mty - timedelta(days=mod28)
        omtyql = ql.Date(omty.day,omty.month,omty.year)
        swpEndInHolyDay = ql.Mexico().isHoliday(omtyql)*1
        endOnHoliday[tradeid] = swpEndInHolyDay
    endOnHoliday = pd.Series(endOnHoliday)
    df_tiieSwps['mtyOnHoliday'] = 0
    df_tiieSwps['mtyOnHoliday'][
        np.where(df_tiieSwps['TradeID'].isin(endOnHoliday.index))[0]
        ] = endOnHoliday

    return(df_tiieSwps)

# Bucket Risk
def get_risk_byBucket(df_book, brCrvs, crvMXNOIS, 
                      ibor_tiie, str_tiiefixings_file, fxrate):
    # discounting engine
    rytsMXNOIS = ql.RelinkableYieldTermStructureHandle()
    rytsMXNOIS.linkTo(crvMXNOIS)
    tiie_swp_engine = ql.DiscountingSwapEngine(rytsMXNOIS)
    # swap obj df 1814
    dfbookval = pd.DataFrame(None, 
                             columns=df_book.columns.tolist()+['SwpObj',
                                                              'NPV',
                                                              'evalDate'])
    # CONTROL VAR
    #print(ql.Settings.instance().evaluationDate)
    # Book's Base NPV
    for i,row in df_book.iterrows():
        bookid, tradeid, tradedate, notnl,\
            stdt, mty, cpdt, r, swptyp, schdlrule = row
        sdt = ql.Date(stdt.day,stdt.month,stdt.year)
        edt = ql.Date(mty.day,mty.month,mty.year)
        swp = tiieSwap(sdt, edt, notnl, ibor_tiie, r/100, swptyp, schdlrule)
        swp[0].setPricingEngine(tiie_swp_engine)
        npv = swp[0].NPV()
        tmpdict = {'BookID': bookid, 
                   'TradeID': tradeid, 
                   'TradeDate': tradedate, 
                   'Notional': notnl, 
                   'StartDate': stdt, 
                   'Maturity': mty, 
                   'CpDate': cpdt, 
                   'FxdRate': r, 
                   'SwpType': swptyp,
                   'mtyOnHoliday': schdlrule,
                   'SwpObj': swp[0],
                   'NPV': npv,
                   'evalDate': ql.Settings.instance().evaluationDate}
        new_row = pd.DataFrame(tmpdict, index=[0])
        dfbookval = pd.concat([dfbookval.loc[:], new_row])
    dfbookval = dfbookval.reset_index(drop=True) 
    # Book's Bucket Sens NPV
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
        ibor_tiie_br = udf.set_ibor_TIIE(rytsForc, str_tiiefixings_file)
        
        # swaps
        lst_npvs = []
        for i,row in dfbookval.iterrows():
            bookid, tradeid, tradedate, notnl, \
                stdt, mty, cpdt, r, \
                swptyp, rule, swpobj, onpv, evDate = row
            sdt = ql.Date(stdt.day,stdt.month,stdt.year)
            edt = ql.Date(mty.day,mty.month,mty.year)
            swp = tiieSwap(sdt, edt, notnl, ibor_tiie_br, r/100, swptyp, rule)
            swp[0].setPricingEngine(discEngine)
            lst_npvs.append(swp[0].NPV())
        modNPV[tenor] = lst_npvs

    df_modNPV = pd.DataFrame(
        modNPV,
        index = dfbookval.index,
        )
    # Bucket DV01
    used = set()
    brTenors = [x for x in 
                [text.split('L', 1)[0]+'L' for text in brCrvs.keys()] 
                if x not in used and (used.add(x) or True)]
    #brTenors = dic_data['MXN_TIIE']['Tenor'].tolist()
    df_tenorDV01 = pd.DataFrame(None, index = dfbookval.index)
    for tenor in brTenors:
        df_tenorp1 = df_modNPV[tenor+'+1']
        df_tenorm1 = df_modNPV[tenor+'-1']
        df_deltap1 = df_tenorp1 - dfbookval['NPV']
        df_deltam1 = df_tenorm1 - dfbookval['NPV']
        df_signs = np.sign(df_deltap1)
        df_tenorDV01[tenor] = df_signs*(abs(df_deltap1)+abs(df_deltam1))/2
    # Book Bucket Risks
    dfbr = pd.Series((df_tenorDV01.sum()/fxrate).sum(), 
                               index = ['OutrightRisk'])
    dfbr = dfbr.append((df_tenorDV01.sum()/fxrate))
    dfbr = dfbr.map('{:,.0f}'.format)
    dic_bookRisks = {
        'NPV_Book': dfbookval.NPV.sum()/fxrate,
        'NPV_Swaps': dfbookval.NPV/fxrate,
        'DV01_Book': dfbr,
        'DV01_Swaps': df_tenorDV01/fxrate
        }
    return(dic_bookRisks)

# By and large NPV
def get_book_npv(df_book, tiie_swp_engine, ibor_tiie):
    # Swap's book dataframe
    dfbookval = pd.DataFrame(None, 
                             columns=df_book.columns.tolist()+['SwpObj',
                                                              'NPV',
                                                              'evalDate'])
    # Book's NPV
    for i,row in df_book.iterrows():
        bookid, tradeid, tradedate, notnl,\
            stdt, mty, cpdt, r, swptyp, schdlrule = row
        sdt = ql.Date(stdt.day,stdt.month,stdt.year)
        edt = ql.Date(mty.day,mty.month,mty.year)
        swp = tiieSwap(sdt, edt, notnl, ibor_tiie, r/100, swptyp, schdlrule)
        swp[0].setPricingEngine(tiie_swp_engine)
        npv = swp[0].NPV()
        tmpdict = {'BookID': bookid, 
                   'TradeID': tradeid, 
                   'TradeDate': tradedate, 
                   'Notional': notnl, 
                   'StartDate': stdt, 
                   'Maturity': mty, 
                   'CpDate': cpdt, 
                   'FxdRate': r, 
                   'SwpType': swptyp,
                   'mtyOnHoliday': schdlrule,
                   'SwpObj': swp[0],
                   'NPV': npv,
                   'evalDate': ql.Settings.instance().evaluationDate}
        new_row = pd.DataFrame(tmpdict, index=[0])
        dfbookval = pd.concat([dfbookval.loc[:], new_row])
    dfbookval = dfbookval.reset_index(drop=True) 
    
    return dfbookval

# Swap Fixing Dates
def tiieSwap_FixingDates(iborIdx, start, maturity, schdl_rule = 0):
    # TIIE Swap Schedule Specs
    cal = ql.Mexico()
    cpn_tenor = ql.Period(13)
    convention = iborIdx.businessDayConvention()
    termDateConvention = iborIdx.businessDayConvention()
    isEndOfMonth = False
    sdt = ql.Date(start.day,start.month,start.year)
    edt = ql.Date(maturity.day,maturity.month,maturity.year)
    # fix-float leg schedules
    fixfltSchdl = ql.Schedule(sdt,
                            edt, 
                            cpn_tenor,
                            cal,
                            convention,
                            termDateConvention,
                            schdl_rule,
                            isEndOfMonth)
    
    return [iborIdx.fixingDate(x) for x in fixfltSchdl][:-1]

# Swap Fixing Dates
def sofrSwap_FixingDates(iborIdx, start, maturity, schdl_rule = 0):
    # TIIE Swap Schedule Specs
    cal = ql.UnitedStates()
    cpn_tenor = ql.Period('1Y')
    convention = iborIdx.businessDayConvention()
    termDateConvention = iborIdx.businessDayConvention()
    isEndOfMonth = True
    sdt = ql.Date(start.day,start.month,start.year)
    edt = ql.Date(maturity.day,maturity.month,maturity.year)
    # fix-float leg schedules
    fixfltSchdl = ql.Schedule(sdt,
                            edt, 
                            cpn_tenor,
                            cal,
                            convention,
                            termDateConvention,
                            schdl_rule,
                            isEndOfMonth)
    
    return [iborIdx.fixingDate(x) for x in fixfltSchdl][:-1]

# Vanilla TIIE Swap
def tiieSwap(start, maturity, notional, iborIdx, rate = 0.04,
              typ=ql.VanillaSwap.Receiver, schdl_rule = 0):
    # TIIE Swap Schedule Specs
    cal = ql.Mexico()
    legDC = ql.Actual360()
    cpn_tenor = ql.Period(13)
    convention = iborIdx.businessDayConvention()
    termDateConvention = iborIdx.businessDayConvention()
    # rule = ql.DateGeneration.Backward
    isEndOfMonth = False
    
    # fix-float leg schedules
    fixfltSchdl = ql.Schedule(start,
                            maturity, 
                            cpn_tenor,
                            cal,
                            convention,
                            termDateConvention,
                            schdl_rule,
                            isEndOfMonth)

    swap = ql.VanillaSwap(typ, 
                             notional, 
                             fixfltSchdl,
                             rate,
                             legDC,
                             fixfltSchdl,
                             iborIdx,
                             0,
                             legDC
                             )

    return swap, [iborIdx.fixingDate(x) for x in fixfltSchdl][:-1]

# Swap OnTheRun CF
def get_CF_tiieSwap(swp, lst_fxngs, ql_tdy):
    #ql_tdy = ql.Settings.instance().evaluationDate
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

# Swap CF at specified date
def get_CF_tiieSwap_atDate(swp, lst_fxngs, ql_tdy):
    #ql_tdy = ql.Settings.instance().evaluationDate
    swp_type = swp.type()
    swp_leg0 = tuple([swpleg0 for swpleg0 in swp.leg(0) 
                      if swpleg0.date() == ql_tdy])
    swp_leg1 = tuple([swpleg1 for swpleg1 in swp.leg(1) 
                      if swpleg1.date() == ql_tdy])
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
    if cf.empty:
        # return cf.append({'date': None, 'netAmt': None}, ignore_index=True)
        return pd.concat([cf, pd.DataFrame({'date': None, 'netAmt': None}, index=[0])], ignore_index=True)
    else:
        cf.insert(1, 'fixingDate',cf1_l2['fixingDate'])
        cf['fltAmt'] = cf1_l2['fltAmt']
        cf['netAmt'] = cf['fixAmt'] + cf['fltAmt']
        cf['fixAmt'] = cf['fixAmt']
        cf['fltAmt'] = cf['fltAmt']
        cf['netAmt'] = cf['netAmt']
        cf['accDays'] = 1*(cf['accEndDate'] - \
                           cf['accStartDate'])/np.timedelta64(1, 'D')
        return cf
###############################################################################
# PORTFOLIO RISK
###############################################################################
# Curves Bootstrapping
def get_curves(dic_data):
    crvUSDOIS = udf.btstrap_USDOIS(dic_data)
    crvUSDSOFR = udf.btstrap_USDSOFR(dic_data, crvUSDOIS)
    crvMXNOIS = udf.btstrap_MXNOIS(dic_data, crvUSDSOFR, crvUSDOIS, 'SOFR')
    crvTIIE = udf.btstrap_MXNTIIE(dic_data, crvMXNOIS)
    
    return crvUSDOIS, crvUSDSOFR, crvMXNOIS, crvTIIE

# Bucket risk curves
def get_tenor_curves(dic_data, crvUSDOIS, crvUSDSOFR, crvMXNOIS):
    return udf.crvTenorRisk_TIIE(dic_data, crvUSDOIS, crvUSDSOFR, crvMXNOIS)

# Swap Pricing Engine
def get_pricing_engine(crvMXNOIS):
    rytsMXNOIS = ql.RelinkableYieldTermStructureHandle()
    rytsMXNOIS.linkTo(crvMXNOIS)
    tiie_swp_engine = ql.DiscountingSwapEngine(rytsMXNOIS)
    
    return tiie_swp_engine

# Porfolio Risk & Valuation
def get_pfolio_riskval(dic_data, str_tiiefixings_file, df_book):
    # FX rate
    fxrate = dic_data['USDMXN_XCCY_Basis']['Quotes'][0]
    # Curves
    crvUSDOIS, crvUSDSOFR, crvMXNOIS, crvTIIE = get_curves(dic_data)
    # Bucket Sens Curves
    brCrvs = get_tenor_curves(dic_data, crvUSDOIS, crvUSDSOFR, crvMXNOIS)
    # Swap Ibor Index
    ibor_tiie = udf.set_ibor_TIIE(crvTIIE, str_tiiefixings_file)
    # Book RiskVal
    dic_book_valrisk = get_risk_byBucket(df_book, brCrvs, crvMXNOIS, 
                                    ibor_tiie, str_tiiefixings_file, fxrate)
    
    return dic_book_valrisk
###############################################################################
# PORTFOLIO PnL
###############################################################################
# Portfolio Valuation & CF Payments at Date
def get_pflio_npv_atDate(dt_valDate, dic_data, str_file_fixngs, df_book):
    # Valuation Date
    ql_vldt = ql.Date(dt_valDate.day, dt_valDate.month, dt_valDate.year)
    ql.Settings.instance().evaluationDate = ql_vldt
    # Curves
    crvUSDOIS, crvUSDSOFR, crvMXNOIS, crvTIIE = get_curves(dic_data)
    # Ibor Index
    ibor_tiie = udf.set_ibor_TIIE(crvTIIE, str_file_fixngs, n=10000)
    # Pricing engine
    tiie_swp_engine = get_pricing_engine(crvMXNOIS)
    # NPVs
    pfolio_npv = get_book_npv(df_book, tiie_swp_engine, ibor_tiie)
    
    return pfolio_npv

# Portfolio Coupon Payments at Date
def get_pfolio_CF_atDate(dt_date, df_book_npv, ibor_tiie):
    # Payment Date
    ql_dt_pmt = ql.Date(dt_date.day, dt_date.month, dt_date.year)
   
    # Coupon Payments by Swap
    dic_cf = {}
    for i,r in df_book_npv.iterrows():
        # Swap
        tmpswp = r['SwpObj']
        # Swap payment dates
        tmpfxnglst =  tiieSwap_FixingDates(ibor_tiie,
                                     r['StartDate'],
                                     r['Maturity'],
                                     r['mtyOnHoliday'])
        # Swap payment at date
        tmpcf = get_CF_tiieSwap_atDate(tmpswp, tmpfxnglst, ql_dt_pmt)
        # Payment at date by swap ID
        dic_cf[r['TradeID']] = tmpcf[tmpcf['date'] == str(dt_date)]['netAmt']
    df_cf = pd.DataFrame.from_dict(dic_cf, orient='index')
    df_cf.columns = ['CF_'+str(dt_date)]
    
    return df_cf

# Portfolio PnL between given dates
def get_pfolio_PnL(str_file, dt_val_yst, dt_val_tdy, 
                   df1814_yst, df1814_tdy, 
                   str_tiiefixings_file):
    # Coupon Payments Date
    dt_cf = dt_val_tdy
    # T-1 Valuation
    dic_data_yst = udf.pull_data(str_file, dt_val_yst)
    df1814_npv_yst = get_pflio_npv_atDate(dt_val_yst, dic_data_yst, 
                                          str_tiiefixings_file, 
                                          df1814_yst)
    # Coupon Payments
    ibor_tiie_cf = udf.set_ibor_TIIE(
            ql.FlatForward(0, ql.Mexico(), 0.11295, ql.Actual360()), 
            str_tiiefixings_file, 
            10000)
    df1814_cf_tdy = get_pfolio_CF_atDate(dt_cf, df1814_npv_yst, 
                                                 str_tiiefixings_file, 
                                                 ibor_tiie_cf)
    # T Valuation
    dic_data_yst = udf.pull_data(str_file, dt_val_tdy)
    df1814_npv_tdy = get_pflio_npv_atDate(dt_val_tdy, dic_data_yst, 
                                                  str_tiiefixings_file, 
                                                  df1814_tdy)
    # Attributes
    cf =  df1814_cf_tdy.sum()
    delta_npv = df1814_npv_tdy.sum()['NPV'] - df1814_npv_yst.sum()['NPV']
    
    return delta_npv, cf


if __name__ == 'main':
    print("HelloWorld3.14159")