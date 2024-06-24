# -*- coding: utf-8 -*-
"""
Yield Curve Builder for Brazilian Swaps

@author: arnulf.q
"""
#%% MODULES
import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%% GLOBALVARS
tenor2ql = {'B':ql.Days,'D':ql.Days,'M':ql.Months,'W':ql.Weeks,'Y':ql.Years}
futMty2Month = {'F':1, 'G':2, 'J':4, 'K':5, 'N':7, 'Q':8, 
                'U': 9, 'V':10, 'X':11, 'Z': 12, 'H': 3, 'M': 6}
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
#%% WORKING DIRECTORY
import os
rsc_path = os.getcwd()+r'\resources'
if not os.path.exists(rsc_path):
    # Current working path
    cwdp = os.getcwd()
    rsc_path = cwdp + r'\resources'

#%% BRL SWAPS YCB
cal_brl = ql.Brazil(0)

# BRL Swaps Market Tickers
brlswpstkrs = ['BZDIOVRA Index']+[f'OD{n+1} Comdty' for n in range(39)]
df_mkt_ = pd.DataFrame(None, columns = ['CTCode','Term','Tenor'], 
             index=brlswpstkrs)

# Eval. Date
date_ql = ql.Date(21,6,2024)
date_ts = date_ql.to_date()
ql.Settings.instance().evaluationDate = date_ql
str_evDate = date_ql.to_date().strftime('%Y-%m-%d')

# Spot Curve
dbpath = rsc_path+r'\db_Curves_mkt.xlsx'
dfdb = pd.read_excel(dbpath, 'bgnPull',skiprows=3)
ct_codes = dfdb.loc[0][df_mkt_.index]
dfdb = dfdb.drop([0,1]).set_index('Date')
df_mkt_ = df_mkt_.merge(dfdb.loc[str_evDate, brlswpstkrs], 
                              left_index=True, right_index=True)
df_mkt_['CTCode'] = ct_codes

df_mkt_['Term'] = df_mkt_.drop('BZDIOVRA Index').\
    apply(lambda x: pd.tseries.offsets.BMonthEnd().\
          rollforward(pd.Timestamp(day=1,
                                   month=futMty2Month[x['CTCode'][2]],
                                   year=2000+int(x['CTCode'][-2:])).date()), 
          axis=1)

df_mkt_.loc['BZDIOVRA Index', 'Term'] = pd.Timestamp(
    year=cal_brl.advance(date_ql,ql.Period('1D')).year(),
    month=cal_brl.advance(date_ql,ql.Period('1D')).month(),
    day=cal_brl.advance(date_ql,ql.Period('1D')).dayOfMonth())

## dates to qlDate
df_mkt_['Term'] = df_mkt_['Term'].apply(lambda x: ql.Date(x.day, x.month, x.year))

## Bussiness days
df_mkt_['Tenor'] = df_mkt_['Term'].apply(lambda x: cal_brl.businessDaysBetween(date_ql, x))

# Curve conventions
dc = ql.Business252()
settleDays = 1
fixingDays = 0
busAdj = ql.ModifiedFollowing
eom = False
pmtFreq = None # Zero-Coupon

# Reference index
CDI = ql.OvernightIndex("CDI",settleDays,ql.BRLCurrency(),cal_brl,dc)
# ON Market
onMkt = df_mkt_.loc['BZDIOVRA Index']
# Bullet Market
zMkt = df_mkt_.drop('BZDIOVRA Index')


# Helpers for 0-1 Days
ql_TPM = ql.QuoteHandle(ql.SimpleQuote(onMkt[pd.Timestamp(str_evDate)]/100))
helpers_ON = [ql.DepositRateHelper(ql_TPM, ql.Period('1D'), fixingDays, cal_brl, busAdj, eom, dc)]

# Helpers for ZCSwaps
zeroHelpers = []
sdt_dt = pd.tseries.offsets.BMonthBegin().rollback(zMkt.iloc[0]['Term'].to_date())
sdt = ql.Date(sdt_dt.day, sdt_dt.month, sdt_dt.year)
for i,r in zMkt.iterrows():
    edt = r['Term']
    qtR = ql.QuoteHandle(ql.SimpleQuote(r[pd.Timestamp(str_evDate)]/100))
    zeroHelpers += [ql.DatedOISRateHelper(sdt, edt, qtR, CDI)]
    sdt = cal_brl.advance(r['Term'],ql.Period('1D'))

# Curve Helpers
crvHelpers = helpers_ON + zeroHelpers

# BRL Curve Bootstrapping
crvBRL = ql.PiecewiseCubicZero(date_ql, crvHelpers, dc)
crvBRL = ql.PiecewiseNaturalLogCubicDiscount(date_ql, crvHelpers, dc)
crvBRL.enableExtrapolation()

print_zero(date_ql, crvBRL)
#%% BRL SWAP PRICING

# BRL Yield Term Structure
rYTS_BRL = ql.RelinkableYieldTermStructureHandle(crvBRL)
CDI = ql.OvernightIndex("CDI",settleDays,ql.BRLCurrency(),cal_brl,dc,rYTS_BRL)
cdi_swp_engine = ql.DiscountingSwapEngine(rYTS_BRL)
if cal_brl.isHoliday(date_ql):
    CDI.addFixing(date_ql - ql.Period('1D'), 
                  rYTS_BRL.forwardRate(date_ql, date_ql+ql.Period('1D'), 
                                       dc, ql.Compounded, ql.Daily).rate())

# Mkt RepRicing
df_cdi = pd.DataFrame()
schdDates = []
start = ql.Date(sdt_dt.day, sdt_dt.month, sdt_dt.year)
for i,row in df_mkt_.drop('BZDIOVRA Index').iterrows():
    end = row['Term']
    qlSch = ql.MakeSchedule(start, end, ql.Period('1M'), ql.NoFrequency, cal_brl)
    schdDates += [list(qlSch.dates())]
    cdi_swap = ql.OvernightIndexedSwap(-1, 1e6, qlSch, 0.04, dc, CDI)
    cdi_swap.setPricingEngine(cdi_swp_engine)
    newDrow = [row.name, 100*cdi_swap.fairRate()]
    df_cdi = pd.concat([df_cdi, pd.DataFrame(newDrow).T])
    start = cal_brl.advance(end, ql.Period('1D'))
df_cdi.columns = ['0', 'Model']
df_cdi.index = df_mkt_.drop('BZDIOVRA Index').index
df_cdi = df_cdi.drop('0', axis=1)
df_cdi = df_cdi.merge(df_mkt_, left_index=True, 
                      right_index=True)[[pd.Timestamp(str_evDate),'Model']]
df_cdi.insert(0,'CTCode', df_mkt_.loc[df_cdi.index]['CTCode'])
print('\n',df_cdi)
styles = ['-','--']
xlabs = df_cdi['CTCode']
df_cdi.plot(legend=None, style=styles)
plt.xticks(range(xlabs.shape[0]), xlabs.tolist(), rotation=45)
plt.legend(['Spot Mkt','Model']); plt.tight_layout();plt.show()

# Market Quote
tnrPer = ql.Period('1Y')
sTnr = start + tnrPer
eTnr = sTnr + tnrPer
pFrq = None
qteSch = ql.MakeSchedule(sTnr, eTnr, ql.Period('1Y'), pFrq, cal_brl)
cdi_swap = ql.OvernightIndexedSwap(-1, 100e6, qteSch, 0.04, dc, CDI)
cdi_swap.setPricingEngine(cdi_swp_engine)
ibr_swpR = 100*cdi_swap.fairRate()
print(f"\n1Y1Y: {ibr_swpR:,.4f}\n")

#%% BRL FORWARD CURVE

# BRL 1M Forwards
st = cal_brl.advance(date_ql, ql.Period('1D'))
df_copFwd = pd.DataFrame()
for n in range(120):
    ed = st + ql.Period('1M')
    fr = rYTS_BRL.forwardRate(st, ed, dc, ql.Compounded, ql.Daily).rate()
    df_copFwd = pd.concat([df_copFwd,pd.DataFrame([st,ed,100*fr]).T])
    st = ed
df_copFwd.reset_index(drop=True, inplace=True)
df_copFwd.columns=['Start','End','Rate']
df_copFwd.plot(x='End', y='Rate', legend=None, title='CDI Fwd Curve', ylabel='1M Rate')
df_copFwd['Pricing'] = 100*(df_copFwd['Rate'] -\
                    df_mkt_.loc['BZDIOVRA Index',pd.Timestamp(str_evDate)])

# BCB Pricing
df_pmd = pd.read_excel(rsc_path+r'\CBPMD.xlsx')
bcb_dates = df_pmd[df_pmd['BRL'] >= pd.Timestamp(date_ql.to_date())]['BRL'].\
    apply(lambda x: ql.Date(x.day,x.month,x.year)).\
        apply(lambda x: cal_brl.advance(x, 1, ql.Days))
df_bcb = pd.DataFrame()
for d in bcb_dates:
    d2 = cal_brl.advance(d, 1, ql.Days)
    fr = rYTS_BRL.forwardRate(d, d2, dc, ql.Compounded, ql.Daily).rate()
    df_bcb = pd.concat([df_bcb,pd.DataFrame([d.to_date(),d2.to_date(),100*fr]).T])
df_bcb.columns = ['Start','End','Rate']
df_bcb['Pricing'] = 100*(df_bcb['Rate'] -\
                    df_mkt_.loc['BZDIOVRA Index',pd.Timestamp(str_evDate)])
df_bcb['IncPric'] = df_bcb['Pricing'] - df_bcb['Pricing'].shift().fillna(0)
df_bcb.index=df_pmd[df_pmd['BRL'] >= pd.Timestamp(date_ql.to_date())]['BRL'].rename('Date')
## adjusting fwd rates wit more than 1 day compounded
df_bcb['Pricing'] = 100*(df_bcb.\
    apply(lambda x: ((1+x['Rate']*(x['End']-x['Start']).days/36000)**\
                     (1/((x['End']-x['Start']).days))-1)*36000, axis=1) -\
        df_mkt_.loc['BZDIOVRA Index',pd.Timestamp(str_evDate)])
df_bcb['IncPric'] = df_bcb['Pricing'] - df_bcb['Pricing'].shift().fillna(0)
# Plot
from matplotlib.dates import DateFormatter
date_form = DateFormatter("%Y-%b")
fig, ax = plt.subplots()
ax.bar(x=df_bcb.index, height=df_bcb['IncPric'], width=25, align='center')
ax.xaxis.set_major_formatter(date_form)
ax.figure.autofmt_xdate()
plt.title('BCB Inc. Pricing')
plt.tight_layout(); plt.show()
print(df_bcb.iloc[:5])


