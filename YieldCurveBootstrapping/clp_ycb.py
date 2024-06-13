# -*- coding: utf-8 -*-
"""
Yield Curve Builder

@author: arnulf.q
"""
#%% MODULES
import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%% GLOBALVARS
tenor2ql = {'B':ql.Days,'D':ql.Days,'M':ql.Months,'W':ql.Weeks,'Y':ql.Years}
futMty2Month = {'U': 9, 'Z': 12, 'H': 3, 'M': 6}
#%% CONVERT CLASS
# utility class for different QuantLib type conversions 
class Convert:
    # Convert date string 'yyyy-mm-dd' to QuantLib Date object
    def to_date(s):
        monthDictionary = {
            '01': ql.January, '02': ql.February, '03': ql.March,
            '04': ql.April, '05': ql.May, '06': ql.June,
            '07': ql.July, '08': ql.August, '09': ql.September,
            '10': ql.October, '11': ql.November, '12': ql.December
        }
        s = s.split('-')
        return ql.Date(int(s[2]), monthDictionary[s[1]], int(s[0]))
    
    # convert string to QuantLib businessdayconvention enumerator
    def to_businessDayConvention(s):
        if (s.upper() == 'FOLLOWING'): return ql.Following
        if (s.upper() == 'MODIFIEDFOLLOWING'): return ql.ModifiedFollowing
        if (s.upper() == 'PRECEDING'): return ql.Preceding
        if (s.upper() == 'MODIFIEDPRECEDING'): return ql.ModifiedPreceding
        if (s.upper() == 'UNADJUSTED'): return ql.Unadjusted
        
    # convert string to QuantLib calendar object
    def to_calendar(s):
        if (s.upper() == 'TARGET'): return ql.TARGET()
        if (s.upper() == 'UNITEDSTATES'): return ql.UnitedStates()
        if (s.upper() == 'UNITEDKINGDOM'): return ql.UnitedKingdom()
        if (s.upper() == 'MEXICO'): return ql.Mexico()
        if (s.upper() == 'CHILE'): return ql.Chile()
        # TODO: add new calendar here
        
    # convert string to QuantLib swap type enumerator
    def to_swapType(s):
        if (s.upper() == 'PAYER'): return ql.VanillaSwap.Payer
        if (s.upper() == 'RECEIVER'): return ql.VanillaSwap.Receiver
        
    # convert string to QuantLib frequency enumerator
    def to_frequency(s):
        if (s.upper() == 'DAILY'): return ql.Daily
        if (s.upper() == 'WEEKLY'): return ql.Weekly
        if (s.upper() == 'MONTHLY'): return ql.Monthly
        if (s.upper() == 'QUARTERLY'): return ql.Quarterly
        if (s.upper() == 'SEMIANNUAL'): return ql.Semiannual
        if (s.upper() == 'ANNUAL'): return ql.Annual

    # convert string to QuantLib date generation rule enumerator
    def to_dateGenerationRule(s):
        if (s.upper() == 'BACKWARD'): return ql.DateGeneration.Backward
        if (s.upper() == 'FORWARD'): return ql.DateGeneration.Forward
        # TODO: add new date generation rule here

    # convert string to QuantLib day counter object
    def to_dayCounter(s):
        if (s.upper() == 'ACTUAL360'): return ql.Actual360()
        if (s.upper() == 'ACTUAL365FIXED'): return ql.Actual365Fixed()
        if (s.upper() == 'ACTUALACTUAL'): return ql.ActualActual()
        if (s.upper() == 'ACTUAL365NOLEAP'): return ql.Actual365NoLeap()
        if (s.upper() == 'BUSINESS252'): return ql.Business252()
        if (s.upper() == 'ONEDAYCOUNTER'): return ql.OneDayCounter()
        if (s.upper() == 'SIMPLEDAYCOUNTER'): return ql.SimpleDayCounter()
        if (s.upper() == 'THIRTY360'): return ql.Thirty360()

    # convert string (ex.'USD.3M') to QuantLib ibor index object
    def to_iborIndex(s):
        s = s.split('.')
        if(s[0].upper() == 'USD'): return ql.USDLibor(ql.Period(s[1]))
        if(s[0].upper() == 'EUR'): return ql.Euribor(ql.Period(s[1]))
        
#%% UDF
def print_zero(date,yieldcurve):
    day_count = ql.Actual360()
    spots = []
    dates = []
    tenors = []
    df = []
    for d in yieldcurve.dates():
        yrs = day_count.yearFraction(date, d)
        df.append(yieldcurve.discount(d))
        dates.append(d)
        compounding = ql.Simple
        freq = ql.Annual
        zero_rate = yieldcurve.zeroRate(yrs, compounding, freq)
        tenors.append(yrs)
        eq_rate = zero_rate.equivalentRate(day_count,compounding,freq,date,d).rate()
        zero_rate.equivalentRate(day_count,compounding,freq,date,d).rate()
        spots.append(100*eq_rate)

    datatable = {'Dates':dates,'Years': tenors,'DF':df,'Zero': spots}
    datatable = pd.DataFrame.from_dict(datatable)
    print(datatable)

#%% CLP SWAPS YCB

# CLP Swaps Market Tickers
clpswpstkrs = ['CHOVCHOV Index','CHSWPA Curncy','CHSWPB Curncy','CHSWPC Curncy',
        'CHSWPF Curncy','CHSWPI Curncy','CHSWP1 Curncy','CHSWP1F Curncy',
        'CHSWP2 Curncy','CHSWP3 Curncy','CHSWP4 Curncy','CHSWP5 Curncy',
        'CHSWP6 Curncy','CHSWP7 Curncy','CHSWP8 Curncy','CHSWP9 Curncy',
        'CHSWP10 Curncy','CHSWP12 Curncy','CHSWP15 Curncy','CHSWP20 Curncy']
term = [1,1,2,3,6,9,12,18,2,3,4,5,6,7,8,9,10,12,15,20]
tenor = ['D','M','M','M','M','M','M','M','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y']
df_mkt_clp = pd.DataFrame(zip(term, tenor),
             columns = ['Term','Tenor'], 
             index=clpswpstkrs)

# Eval. Date
date_ql = ql.Date(13,6,2024)
ql.Settings.instance().evaluationDate = date_ql
str_evDate = date_ql.to_date().strftime('%Y-%m-%d')

# Spot Curve
dbpath = r'U:\Fixed Income\File Dump\Database\db_Curves_mkt.xlsx'
dfdb = pd.read_excel(dbpath, 'bgnPull',skiprows=3)
dfdb = dfdb.drop([0,1]).set_index('Date')
df_mkt_clp = df_mkt_clp.merge(dfdb.loc[str_evDate, clpswpstkrs], 
                              left_index=True, right_index=True)
df_mkt_clp['qlTenor'] = df_mkt_clp['Tenor'].map(tenor2ql).values

# Curve conventions
cal_clp = ql.Chile()
dc = ql.Actual360()
settlement_days_icp = 2
fixingDays = 0
busAdj = ql.ModifiedFollowing
eom = False
pmtFreq = ql.Semiannual

# Reference index
ICP = ql.OvernightIndex("ICP",settlement_days_icp,ql.CLPCurrency(),cal_clp,dc)
# ON Market
onMkt = df_mkt_clp.loc['CHOVCHOV Index']
# Bullet Market
zMkt = df_mkt_clp[df_mkt_clp['qlTenor']<3].drop('CHOVCHOV Index')
# Coupon Market
cMkt = df_mkt_clp[df_mkt_clp['qlTenor']>=3]

# Helpers for 0-1 Days
ql_TPM = ql.QuoteHandle(ql.SimpleQuote(onMkt[pd.Timestamp(str_evDate)]/100))
ql_term = onMkt['Term']
ql_tenor = onMkt['qlTenor']
helpers_ON = [ql.DepositRateHelper(ql_TPM, 
                                ql.Period(int(onMkt['Term']), int(onMkt['qlTenor'])), 
                                fixingDays, cal_clp, busAdj, eom, dc)]

# Helpers for 1-18 Months
zeroHelpers = []
for i,r in zMkt.iterrows():
    qlQte = ql.QuoteHandle(ql.SimpleQuote(r[pd.Timestamp(str_evDate)]/100))
    qlPer = ql.Period(int(r['Term']), int(r['qlTenor']))
    zeroHelpers += [ql.DepositRateHelper(qlQte, qlPer, settlement_days_icp, 
                                         cal_clp, busAdj, eom, dc)]
# Helpers for 2-20 Years
swapHelpers = []
for i,r in cMkt.iterrows():
    qlQte = ql.QuoteHandle(ql.SimpleQuote(r[pd.Timestamp(str_evDate)]/100))
    qlPer = ql.Period(int(r['Term']), int(r['qlTenor']))
    swapHelpers += [ql.OISRateHelper(settlementDays=settlement_days_icp, 
                                 tenor=qlPer,
                                 rate=qlQte,
                                 index=ICP,
                                 discountingCurve=ql.YieldTermStructureHandle(), 
                                 telescopicValueDates=False, 
                                 paymentLag=0, 
                                 paymentConvention=busAdj, 
                                 paymentFrequency=pmtFreq, 
                                 paymentCalendar=cal_clp)]
# Curve Helpers
clpHelpers = helpers_ON + zeroHelpers + swapHelpers

# CLP Curve Bootstrapping
crvCLP = ql.PiecewiseCubicZero(date_ql, clpHelpers, dc)
crvCLP.enableExtrapolation()

print_zero(date_ql, crvCLP)
#%% CLP SWAP PRICING

# CLP Yield Term Structure
rYTS_CLP = ql.RelinkableYieldTermStructureHandle(crvCLP)
ICP = ql.OvernightIndex("ICP",settlement_days_icp,ql.CLPCurrency(),cal_clp,dc,
                        rYTS_CLP)
icp_swp_engine = ql.DiscountingSwapEngine(rYTS_CLP)

# Mkt RepRicing
dic_pmtFreq = {'Y':ql.Semiannual, 'M': ql.Annual}
df_icp = pd.DataFrame()
for i,row in df_mkt_clp.drop('CHOVCHOV Index').iterrows():
    icp_tenor = ql.Period(int(row['Term']), int(row['qlTenor']))
    icp_swap = ql.MakeOIS(swapTenor=icp_tenor, overnightIndex=ICP, 
                          fixedRate=0.04, settlementDays=settlement_days_icp,
                          paymentFrequency=dic_pmtFreq[row['Tenor']],
                          paymentAdjustmentConvention=busAdj)
    icp_swap.setPricingEngine(icp_swp_engine)
    newDrow = [row.name, 100*icp_swap.fairRate()]
    df_icp = pd.concat([df_icp, pd.DataFrame(newDrow).T])
df_icp.columns = ['0', 'Model']
df_icp.index = df_mkt_clp.drop('CHOVCHOV Index').index
df_icp = df_icp.drop('0', axis=1)
df_icp.merge(df_mkt_clp, left_index=True, right_index=True)[[pd.Timestamp(str_evDate),'Model']]
        
        
        
        
        
        
        
        
        