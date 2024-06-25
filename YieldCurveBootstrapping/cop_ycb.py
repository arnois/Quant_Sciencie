# -*- coding: utf-8 -*-
"""
Yield Curve Builder for Colombian Swaps

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

#%% COP SWAPS YCB

# COP Swaps Market Tickers
copswpstkrs = ['COOVIBR Index','CLSWIBA BGN Curncy','CLSWIBC BGN Curncy','CLSWIBF BGN Curncy',
        'CLSWIBI BGN Curncy','CLSWIB1 BGN Curncy','CLSWIB1F BGN Curncy','CLSWIB2 BGN Curncy',
        'CLSWIB3 BGN Curncy','CLSWIB4 BGN Curncy','CLSWIB5 BGN Curncy','CLSWIB6 BGN Curncy',
        'CLSWIB7 BGN Curncy','CLSWIB8 BGN Curncy','CLSWIB9 BGN Curncy','CLSWIB10 BGN Curncy',
        'CLSWIB12 BGN Curncy','CLSWIB15 BGN Curncy','CLSWIB20 BGN Curncy']
term = [1,1,3,6,9,12,18,2,3,4,5,6,7,8,9,10,12,15,20]
tenor = ['D','M','M','M','M','M','M','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y']
df_mkt_cop = pd.DataFrame(zip(term, tenor),
             columns = ['Term','Tenor'], 
             index=copswpstkrs)

# Eval. Date
date_ql = ql.Date(25,6,2024)
ql.Settings.instance().evaluationDate = date_ql
str_evDate = date_ql.to_date().strftime('%Y-%m-%d')

# Spot Curve
dbpath = r'U:\Fixed Income\File Dump\Database\db_Curves_mkt.xlsx'
dfdb = pd.read_excel(dbpath, 'bgnPull',skiprows=3)
dfdb = dfdb.drop([0,1]).set_index('Date')
df_mkt_cop = df_mkt_cop.merge(dfdb.loc[str_evDate, copswpstkrs], 
                              left_index=True, right_index=True)
df_mkt_cop['qlTenor'] = df_mkt_cop['Tenor'].map(tenor2ql).values

# Curve conventions
cal_cop = ql.TARGET()
dc = ql.Actual360()
settleDays = 2
fixingDays = 0
busAdj = ql.ModifiedFollowing
eom = False
pmtFreq = ql.Quarterly

# Reference index
IBR = ql.OvernightIndex("IBR",settleDays,ql.COPCurrency(),cal_cop,dc)
# ON Market
onMkt = df_mkt_cop.loc['COOVIBR Index']
# Bullet Market
zMkt = df_mkt_cop[df_mkt_cop['qlTenor']<3].drop('COOVIBR Index')
# Coupon Market
cMkt = df_mkt_cop[df_mkt_cop['qlTenor']>=3]

# Helpers for 0-1 Days
ql_TPM = ql.QuoteHandle(ql.SimpleQuote(onMkt[pd.Timestamp(str_evDate)]/100))
ql_term = onMkt['Term']
ql_tenor = onMkt['qlTenor']
helpers_ON = [ql.DepositRateHelper(ql_TPM, 
                                ql.Period(int(onMkt['Term']), int(onMkt['qlTenor'])), 
                                fixingDays, cal_cop, busAdj, eom, dc)]

# Helpers for 1-18 Months
zeroHelpers = []
for i,r in zMkt.iterrows():
    qlQte = ql.QuoteHandle(ql.SimpleQuote(r[pd.Timestamp(str_evDate)]/100))
    qlPer = ql.Period(int(r['Term']), int(r['qlTenor']))
    zeroHelpers += [ql.DepositRateHelper(qlQte, qlPer, settleDays, 
                                         cal_cop, busAdj, eom, dc)]
# Helpers for 2-20 Years
swapHelpers = []
for i,r in cMkt.iterrows():
    qlQte = ql.QuoteHandle(ql.SimpleQuote(r[pd.Timestamp(str_evDate)]/100))
    qlPer = ql.Period(int(r['Term']), int(r['qlTenor']))
    swapHelpers += [ql.OISRateHelper(settlementDays=settleDays, 
                                 tenor=qlPer,
                                 rate=qlQte,
                                 index=IBR,
                                 discountingCurve=ql.YieldTermStructureHandle(), 
                                 telescopicValueDates=False, 
                                 paymentLag=0, 
                                 paymentConvention=busAdj, 
                                 paymentFrequency=pmtFreq, 
                                 paymentCalendar=cal_cop)]
# Curve Helpers
crvHelpers = helpers_ON + zeroHelpers + swapHelpers

# COP Curve Bootstrapping
crvCOP = ql.PiecewiseCubicZero(date_ql, crvHelpers, dc)
crvCOP = ql.PiecewiseNaturalLogCubicDiscount(date_ql, crvHelpers, dc)
crvCOP.enableExtrapolation()

print_zero(date_ql, crvCOP)
#%% COP SWAP PRICING

# COP Yield Term Structure
rYTS_COP = ql.RelinkableYieldTermStructureHandle(crvCOP)
IBR = ql.OvernightIndex("IBR",settleDays,ql.COPCurrency(),cal_cop,dc,
                        rYTS_COP)
ibr_swp_engine = ql.DiscountingSwapEngine(rYTS_COP)
if cal_cop.isHoliday(date_ql):
    IBR.addFixing(date_ql - ql.Period('1D'), 
                  rYTS_COP.forwardRate(date_ql, date_ql+ql.Period('1D'), 
                                       dc, ql.Compounded, ql.Daily).rate())

# Mkt RepRicing
df_ibr = pd.DataFrame()
schdDates = []
start = cal_cop.advance(date_ql, 2, ql.Days)
for i,row in df_mkt_cop.drop('COOVIBR Index').iterrows():
    ibr_tenor = ql.Period(int(row['Term']), int(row['qlTenor']))
    # end = cal_cop.advance(start, icp_tenor)
    end = start + ibr_tenor
    pmtFreq = None
    if row['qlTenor'] >= 3: pmtFreq = ql.Quarterly
    qlSch = ql.MakeSchedule(start, start + ibr_tenor, ibr_tenor, pmtFreq, cal_cop)
    schdDates += [list(qlSch.dates())]
    ibr_swap = ql.OvernightIndexedSwap(-1, 1e6, qlSch, 0.04, dc, IBR)
    ibr_swap.setPricingEngine(ibr_swp_engine)
    newDrow = [row.name, 100*ibr_swap.fairRate()]
    df_ibr = pd.concat([df_ibr, pd.DataFrame(newDrow).T])
df_ibr.columns = ['0', 'Model']
df_ibr.index = df_mkt_cop.drop('COOVIBR Index').index
df_ibr = df_ibr.drop('0', axis=1)
df_ibr = df_ibr.merge(df_mkt_cop, left_index=True, 
                      right_index=True)[[pd.Timestamp(str_evDate),'Model']]
df_ibr.insert(0,'Tenor', df_mkt_cop.loc[df_ibr.index].apply(lambda x: str(x['Term'])+x['Tenor'],axis=1))
print('\n',df_ibr)
styles = ['-','--']
xlabs = df_ibr['Tenor']
df_ibr.plot(legend=None, style=styles)
plt.xticks(range(xlabs.shape[0]), xlabs.tolist(), rotation=45)
plt.legend(['Spot Mkt','Model']); plt.tight_layout();plt.show()

# Market Quote
tnrPer = ql.Period('1Y')
sTnr = start + tnrPer
eTnr = sTnr + tnrPer
pFrq = None
if (eTnr - start)/360 >= 2: pFrq = ql.Quarterly
qteSch = ql.MakeSchedule(sTnr, eTnr, ibr_tenor, pFrq, cal_cop)
ibr_swap = ql.OvernightIndexedSwap(-1, 100e6, qteSch, 0.04, dc, IBR)
ibr_swap.setPricingEngine(ibr_swp_engine)
ibr_swpR = 100*ibr_swap.fairRate()
print(f"\n1Y1Y: {ibr_swpR:,.4f}\n")

#%% COP FORWARD CURVE

# COP 1M Forwards
st = start
df_copFwd = pd.DataFrame()
for n in range(120):
    ed = st + ql.Period('1M')
    fr = rYTS_COP.forwardRate(st, ed, dc, ql.Compounded, ql.Daily).rate()
    df_copFwd = pd.concat([df_copFwd,pd.DataFrame([st,ed,100*fr]).T])
    st = ed
df_copFwd.reset_index(drop=True, inplace=True)
df_copFwd.columns=['Start','End','Rate']
df_copFwd.plot(x='End', y='Rate', legend=None, title='IBR Fwd Curve', ylabel='1M Rate')
df_copFwd['Pricing'] = 100*(df_copFwd['Rate'] -\
                    df_mkt_cop.loc['COOVIBR Index',pd.Timestamp(str_evDate)])

# BanRepCol Pricing
df_pmd = pd.read_excel(r'U:\Fixed Income\File Dump\Database\CBPMD.xlsx')
bdrc_dates = df_pmd[df_pmd['COP'] >= pd.Timestamp(date_ql.to_date())]['COP'].\
    apply(lambda x: ql.Date(x.day,x.month,x.year)).\
        apply(lambda x: cal_cop.advance(x, 2, ql.Days))
df_bdrc = pd.DataFrame()
for d in bdrc_dates:
    d2 = cal_cop.advance(d, 1, ql.Days)
    fr = rYTS_COP.forwardRate(d, d2, dc, ql.Compounded, ql.Daily).rate()
    df_bdrc = pd.concat([df_bdrc,pd.DataFrame([d.to_date(),d2.to_date(),100*fr]).T])
df_bdrc.columns = ['Start','End','Rate']
df_bdrc['Pricing'] = 100*(df_bdrc['Rate'] -\
                    df_mkt_cop.loc['COOVIBR Index',pd.Timestamp(str_evDate)])
df_bdrc['IncPric'] = df_bdrc['Pricing'] - df_bdrc['Pricing'].shift().fillna(0)
df_bdrc.index=df_pmd[df_pmd['COP'] >= pd.Timestamp(date_ql.to_date())]['COP'].rename('Date')
## adjusting fwd rates wit more than 1 day compounded
df_bdrc['Pricing'] = 100*(df_bdrc.\
    apply(lambda x: ((1+x['Rate']*(x['End']-x['Start']).days/36000)**\
                     (1/((x['End']-x['Start']).days))-1)*36000, axis=1) -\
        df_mkt_cop.loc['COOVIBR Index',pd.Timestamp(str_evDate)])
df_bdrc['IncPric'] = df_bdrc['Pricing'] - df_bdrc['Pricing'].shift().fillna(0)
# Plot
from matplotlib.dates import DateFormatter
date_form = DateFormatter("%Y-%b")
fig, ax = plt.subplots()
ax.bar(x=df_bdrc.index, height=df_bdrc['IncPric'], width=25, align='center')
ax.xaxis.set_major_formatter(date_form)
ax.figure.autofmt_xdate()
plt.title('BDRC Inc. Pricing')
plt.tight_layout(); plt.show()
print(df_bdrc.iloc[:5])


