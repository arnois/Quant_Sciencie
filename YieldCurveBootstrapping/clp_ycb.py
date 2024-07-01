# -*- coding: utf-8 -*-
"""
Yield Curve Builder for Chilean Swaps

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

#%% WORKING DIRECTORY
import os
rsc_path = os.getcwd()+r'\resources'
if not os.path.exists(rsc_path):
    # Current working path
    cwdp = os.getcwd()
    rsc_path = cwdp + r'\resources'


#%% EVALUATION DATE
date_ql = ql.Date(28,6,2024)
ql.Settings.instance().evaluationDate = date_ql
str_evDate = date_ql.to_date().strftime('%Y-%m-%d')

#%% MARKET DATA

# Import
dbpath = rsc_path+r'\db_Curves_mkt.xlsx'
dfdb = pd.read_excel(dbpath, 'bgnPull',skiprows=3)
dfdb = dfdb.drop([0,1]).set_index('Date')

###############################################################################
# USDOIS Swaps Market Tickers (USDOIS SMT)
ois_idx_tkr = ['FEDL01 Index']
sec = ['1Z','2Z','3Z','A','B','C','D','E','F','I','1','1F']
lec = [2,3,4,5,7,10,12,15,20,25,30,40]
ois_swp_tkr = [f'USSO{s} Curncy' for s in sec]+[f'USSO{s} Curncy' for s in lec]
oistkrs = ois_idx_tkr + ois_swp_tkr
ois_tenors = ['1B','1W','2W','3W']+[f'{n+1}M' for n in list(range(6))+[8,11,17]]
ois_tenors += [f'{n}Y' for n in range(2,6)]
ois_tenors += [f'{n}Y' for n in [7,10,12,15,20,25,30,40]]
ois_periods = [int(s.replace('B','').replace('W','').replace('M','').\
                   replace('Y','')) for s in ois_tenors]
df_mkt_ois = pd.DataFrame(zip(ois_tenors, ois_periods), 
                          columns=['Tenors', 'Period'], index=oistkrs)
df_mkt_ois = df_mkt_ois.merge(dfdb.loc[str_evDate, oistkrs], 
                              left_index=True, right_index=True)
df_mkt_ois = df_mkt_ois.rename(columns={pd.Timestamp(str_evDate):'Quotes'})

###############################################################################
# IMM Futures OTR
imm_dates = [ql.IMM.nextDate(ql.IMM.nextDate(date_ql)-ql.Period('3M')-ql.Period('1W'))]
for n in range(4):
    imm_dates += [ql.IMM.nextDate(imm_dates[n])]

# SOFR SMT
sofr_idx_tkr = ['SOFRRATE Index']
sofr_fut_tkr = [f'SFR{ql.IMM.code(m)} Comdty' for m in imm_dates]
lst_sofrswpTenors = list(range(2,11)) + [12] + list(range(15,30,5)) + list(range(30,50,10))
sofr_swp_tkr = [f'USOSFR{n} BGN Curncy' for n in lst_sofrswpTenors]
sofrtkrs = sofr_idx_tkr + sofr_fut_tkr + sofr_swp_tkr
sofr_tenors = ['1B']+[s.replace(' Comdty','') for s in sofr_fut_tkr]
sofr_tenors += [f'{n}Y' for n in list(range(2,11))+[12,15,20,25,30,40]]
sofr_periods = ['1','1F','2F','3F','4F','5F']+[s.replace('Y','') for s in sofr_tenors if 'Y' in s]
df_mkt_sofr = pd.DataFrame(zip(sofr_tenors, sofr_periods), 
                          columns=['Tenors', 'Period'], index=sofrtkrs)
df_mkt_sofr = df_mkt_sofr.merge(dfdb.loc[str_evDate, sofrtkrs], 
                                left_index=True, right_index=True)
df_mkt_sofr = df_mkt_sofr.rename(columns={pd.Timestamp(str_evDate):'Quotes'})
df_mkt_sofr['Types'] = 'SWAP'
df_mkt_sofr['Types'][df_mkt_sofr['Tenors']=='1B'] = 'DEPO'
df_mkt_sofr['Types'][df_mkt_sofr['Period'].apply(lambda x: x[-1])=='F'] = 'FUT'
df_mkt_sofr.loc[df_mkt_sofr['Types'] == 'SWAP','Tenors'] = \
    df_mkt_sofr[df_mkt_sofr['Types'] == 'SWAP']['Tenors'].apply(lambda x: '%'+x)
###############################################################################
# CLP SMT
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
# Spot Curve
df_mkt_clp = df_mkt_clp.merge(dfdb.loc[str_evDate, clpswpstkrs], 
                              left_index=True, right_index=True)
df_mkt_clp['qlTenor'] = df_mkt_clp['Tenor'].map(tenor2ql).values

# CLP NDF Tickers
clp_ndf_tkrs = ['USDCLP Curncy', 'CHN1W Curncy', 'CHN2W Curncy', 'CHN3W Curncy',
                'CHN1M Curncy', 'CHN2M Curncy', 'CHN3M Curncy',	'CHN4M Curncy',	
                'CHN5M Curncy',	'CHN6M Curncy',	'CHN7M Curncy',	'CHN8M Curncy',	
                'CHN9M Curncy',	'CHN10M Curncy', 'CHN11M Curncy',
                'CHN12M Curncy', 'CHN18M Curncy']
# CLP NDS Tickers
clp_nds_tkrs = ['CPXOSS2 BGN Curncy',	'CPXOSS3 BGN Curncy', 
                  'CPXOSS4 BGN Curncy',	'CPXOSS5 BGN Curncy',
                  'CPXOSS6 BGN Curncy',	'CPXOSS7 BGN Curncy',
                  'CPXOSS8 BGN Curncy',	'CPXOSS9 BGN Curncy',	
                  'CPXOSS10 BGN Curncy','CPXOSS12 BGN Curncy',
                  'CPXOSS15 BGN Curncy', 'CPXOSS20 BGN Curncy']
# CLP vs SOFR Basis Spot Curve
clp_basis_tkrs = clp_ndf_tkrs+clp_nds_tkrs
df_mkt_clp_basis = pd.DataFrame(None, columns = ['Term','Tenor'], 
                                index=clp_basis_tkrs)
df_mkt_clp_basis = df_mkt_clp_basis.merge(dfdb.loc[str_evDate, clp_basis_tkrs], 
                              left_index=True, right_index=True)
df_mkt_clp_basis.loc[clp_ndf_tkrs[0],['Term','Tenor']] = [1,'B']
df_mkt_clp_basis.loc[clp_ndf_tkrs[1:],'Tenor'] = [s.replace('CHN','').\
                                                  replace(' Curncy','')[-1] 
                                                  for s in clp_ndf_tkrs[1:]]
    
df_mkt_clp_basis.loc[clp_ndf_tkrs[1:],'Term'] = [int(s.replace('CHN','').\
                                                  replace(' Curncy','')[:-1])
                                                  for s in clp_ndf_tkrs[1:]]
df_mkt_clp_basis.loc[clp_nds_tkrs, 'Term'] = [int(s.replace('CPXOSS','').\
                                                  replace(' BGN Curncy','')) 
                                              for s in clp_nds_tkrs]
df_mkt_clp_basis.loc[clp_nds_tkrs, 'Tenor'] = 'Y'
df_mkt_clp_basis['qlTenor'] = df_mkt_clp_basis['Tenor'].map(tenor2ql).values


#%% USDOIS CURVE
import udf_PWTSB as f
pcbUSDOIS = f.PiecewiseCurveBuilder_OIS(df_mkt_ois)
pcbUSDOIS.set_qlHelper_USDOIS()
pcbUSDOIS.btstrap_USDOIS('NLC')

#%% SOFR CURVE
pcbSOFR = f.PiecewiseCurveBuilder_SWAP(market_data=df_mkt_sofr, discCrv=pcbUSDOIS.crv)
pcbSOFR.set_qlHelper_SOFR()
pcbSOFR.btstrap_USDSOFR('NLC')

#%% CLP SWAPS YCB
exit()
# Curve conventions
cal_clp = ql.Chile(0)
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

# CLP IMPLIED DISC CURVE
bSett = 2
fxFixingDays = 2
fxBusAdj = ql.Following
fxEOM = False
fxIsBaseColl = True
fxCollCrv = ql.RelinkableYieldTermStructureHandle()
fxCollCrv.linkTo(pcbUSDOIS.crv)
## Depo
helpers_ON = [ql.DepositRateHelper(ql_TPM, 
                                ql.Period(int(onMkt['Term']), int(onMkt['qlTenor'])), 
                                fixingDays, cal_clp, busAdj, eom, dc)]

## NDF
fxSwapHelper = []
spotfx = df_mkt_clp_basis.loc[clp_ndf_tkrs[0], pd.Timestamp(str_evDate)]
qlSpotFX = ql.QuoteHandle(ql.SimpleQuote(spotfx))
for i,r in df_mkt_clp_basis.loc[clp_ndf_tkrs[1:]].iterrows():
    qlQte = ql.QuoteHandle(ql.SimpleQuote(r[pd.Timestamp(str_evDate)]/1))
    qlPer = ql.Period(int(r['Term']), int(r['qlTenor']))
    fxSwapHelper += [ql.FxSwapRateHelper(qlQte, 
                                         qlSpotFX, 
                                         qlPer, 
                                         fxFixingDays,
                                         cal_clp, 
                                         fxBusAdj, 
                                         fxEOM, 
                                         fxIsBaseColl, 
                                         fxCollCrv)]
## NDS
ndsHelpers = []
fxIborIndex = ql.Sofr()
for i,r in df_mkt_clp_basis.loc[clp_nds_tkrs].iterrows():
    srate = cMkt[cMkt['Term'] == r['Term']][pd.Timestamp(str_evDate)].iloc[0]/100
    qlQteS = ql.QuoteHandle(ql.SimpleQuote(srate))
    qlQteB = ql.QuoteHandle(ql.SimpleQuote(r[pd.Timestamp(str_evDate)]/10000))
    qlPer = ql.Period(int(r['Term']), int(r['qlTenor']))
    ndsHelpers += [ql.SwapRateHelper(qlQteS, qlPer, cal_clp, pmtFreq, busAdj,
                                     dc, fxIborIndex, qlQteB, ql.Period(0, ql.Days), 
                                     ql.YieldTermStructureHandle(), 1)]

basisHlprs = fxSwapHelper + ndsHelpers

# CLP Disc Curve
crvCLPOIS = ql.PiecewiseLogLinearDiscount(0, cal_clp, basisHlprs, dc)
crvCLPOIS.enableExtrapolation()
crvCLPOIS.zeroRate(date_ql+ql.Period('2Y'),dc,ql.Simple)
crvCLPOIS.forwardRate(date_ql+ql.Period('1Y'), date_ql+ql.Period('2Y'), dc, ql.Simple)
discCLP = ql.YieldTermStructureHandle(crvCLPOIS)

# CLP FORC CURVE
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
                                 discountingCurve=discCLP, 
                                 telescopicValueDates=False, 
                                 paymentLag=0, 
                                 paymentConvention=busAdj, 
                                 paymentFrequency=pmtFreq, 
                                 paymentCalendar=cal_clp)]
# Curve Helpers
clpHelpers = helpers_ON + zeroHelpers + swapHelpers

# CLP Curve Bootstrapping
crvCLP = ql.PiecewiseCubicZero(date_ql, clpHelpers, dc)
crvCLP = ql.PiecewiseNaturalLogCubicDiscount(date_ql, clpHelpers, dc)
crvCLP.enableExtrapolation()

print_zero(date_ql, crvCLP)
#%% CLP SWAP PRICING

# CLP Yield Term Structure
rYTS_CLP = ql.RelinkableYieldTermStructureHandle(crvCLP)
ICP = ql.OvernightIndex("ICP",settlement_days_icp,ql.CLPCurrency(),cal_clp,dc,
                        rYTS_CLP)
icp_swp_engine = ql.DiscountingSwapEngine(rYTS_CLP)
if cal_clp.isHoliday(date_ql):
    ICP.addFixing(date_ql - ql.Period('1D'), 
                  rYTS_CLP.forwardRate(date_ql, date_ql+ql.Period('1D'), 
                                       dc, ql.Compounded, ql.Daily).rate())

# Mkt RepRicing
df_icp = pd.DataFrame()
schdDates = []
start = cal_clp.advance(date_ql, 2, ql.Days)
for i,row in df_mkt_clp.drop('CHOVCHOV Index').iterrows():
    icp_tenor = ql.Period(int(row['Term']), int(row['qlTenor']))
    # end = cal_clp.advance(start, icp_tenor)
    end = start + icp_tenor
    pmtFreq = None
    if row['qlTenor'] >= 3: pmtFreq = ql.Semiannual
    qlSch = ql.MakeSchedule(start, start + icp_tenor, icp_tenor, pmtFreq, cal_clp)
    schdDates += [list(qlSch.dates())]
    icp_swap = ql.OvernightIndexedSwap(-1, 1e6, qlSch, 0.04, dc, ICP)
    icp_swap.setPricingEngine(icp_swp_engine)
    newDrow = [row.name, 100*icp_swap.fairRate()]
    df_icp = pd.concat([df_icp, pd.DataFrame(newDrow).T])
df_icp.columns = ['0', 'Model']
df_icp.index = df_mkt_clp.drop('CHOVCHOV Index').index
df_icp = df_icp.drop('0', axis=1)
df_icp = df_icp.merge(df_mkt_clp, left_index=True, 
                      right_index=True)[[pd.Timestamp(str_evDate),'Model']]
df_icp.insert(0,'Tenor', df_mkt_clp.loc[df_icp.index].apply(lambda x: str(x['Term'])+x['Tenor'],axis=1))
print('\n',df_icp)
styles = ['-','--']
xlabs = df_icp['Tenor']
df_icp.plot(legend=None, style=styles)
plt.xticks(range(xlabs.shape[0]), xlabs.tolist(), rotation=45)
plt.legend(['Spot Mkt','Model']); plt.tight_layout();plt.show()

# Market Quote
tnrPer = ql.Period('1Y')
sTnr = start + tnrPer
eTnr = sTnr + tnrPer
eTnr - start
pFrq = None
if (eTnr - start)/360 >= 2: pFrq = ql.Semiannual
qteSch = ql.MakeSchedule(sTnr, eTnr, icp_tenor, pFrq, cal_clp)
icp_swap = ql.OvernightIndexedSwap(-1, 100e6, qteSch, 0.04, dc, ICP)
icp_swap.setPricingEngine(icp_swp_engine)
icp_swpR = 100*icp_swap.fairRate()
print(f'\n 1y1y: {icp_swpR:,.4f}\n')

#%% CLP FORWARD CURVE

# CLP 1M Forwards
st = start
df_clpFwd = pd.DataFrame()
for n in range(120):
    ed = st + ql.Period('1M')
    fr = rYTS_CLP.forwardRate(st, ed, dc, ql.Compounded, ql.Daily).rate()
    df_clpFwd = pd.concat([df_clpFwd,pd.DataFrame([st,ed,100*fr]).T])
    st = ed
df_clpFwd.reset_index(drop=True, inplace=True)
df_clpFwd.columns=['Start','End','Rate']
df_clpFwd.plot(x='End', y='Rate', legend=None, title='CAM Fwd Curve', ylabel='1M Rate')
df_clpFwd['Pricing'] = 100*(df_clpFwd['Rate'] -\
                    df_mkt_clp.loc['CHOVCHOV Index',pd.Timestamp(str_evDate)])

# BCCE Pricing
df_pmd = pd.read_excel(r'U:\Fixed Income\File Dump\Database\CBPMD.xlsx')
bcce_dates = df_pmd[df_pmd['CLP'] >= pd.Timestamp(date_ql.to_date())]['CLP'].\
    apply(lambda x: ql.Date(x.day,x.month,x.year)).\
        apply(lambda x: cal_clp.advance(x, 2, ql.Days))
df_bcce = pd.DataFrame()
for d in bcce_dates:
    d2 = cal_clp.advance(d, 1, ql.Days)
    fr = rYTS_CLP.forwardRate(d, d2, dc, ql.Compounded, ql.Daily).rate()
    df_bcce = pd.concat([df_bcce,pd.DataFrame([d.to_date(),d2.to_date(),100*fr]).T])
df_bcce.columns = ['Start','End','Rate']
df_bcce['Pricing'] = 100*(df_bcce['Rate'] -\
                    df_mkt_clp.loc['CHOVCHOV Index',pd.Timestamp(str_evDate)])
df_bcce['IncPric'] = df_bcce['Pricing'] - df_bcce['Pricing'].shift().fillna(0)
df_bcce.index=df_pmd[df_pmd['CLP'] >= pd.Timestamp(date_ql.to_date())]['CLP'].rename('Date')
## adjusting fwd rates wit more than 1 day compounded
df_bcce['Pricing'] = 100*(df_bcce.\
    apply(lambda x: ((1+x['Rate']*(x['End']-x['Start']).days/36000)**\
                     (1/((x['End']-x['Start']).days))-1)*36000, axis=1) -\
        df_mkt_clp.loc['CHOVCHOV Index',pd.Timestamp(str_evDate)])
df_bcce['IncPric'] = df_bcce['Pricing'] - df_bcce['Pricing'].shift().fillna(0)
# Plot
from matplotlib.dates import DateFormatter
date_form = DateFormatter("%Y-%b")
fig, ax = plt.subplots()
ax.bar(x=df_bcce.index, height=df_bcce['IncPric'], width=25, align='center')
ax.xaxis.set_major_formatter(date_form)
ax.figure.autofmt_xdate()
plt.title('BCCE Inc. Pricing')
plt.tight_layout(); plt.show()
print(df_bcce.iloc[:5])


