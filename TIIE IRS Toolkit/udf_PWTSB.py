# -*- coding: utf-8 -*-
"""
Builder for Piecewise Trem Structure


@author: arnulf.q
"""
#%% MODULES
import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
#%% Piecewise Curve Builder Class

# Create piecewise yield term structure
class PiecewiseCurveBuilder:
    def __init__(self, settleDays, calendar, dayCounter, crvDisc = None):
        self.helpers = []
        self.settleDays = settleDays
        self.calendar = calendar
        self.dayCounter = dayCounter
        self.discount_curve = crvDisc

    # DepositRateHelper(Rate rate, const shared_ptr<IborIndex> &iborIndex)
    def AddDeposit(self, rate, qlPeriod, convention):
        qlRate = ql.QuoteHandle(ql.SimpleQuote(rate))
        #helper = ql.DepositRateHelper(qlRate, iborIndex)
        helper = ql.DepositRateHelper(
            qlRate,
            qlPeriod, 
            2,
            self.calendar, 
            convention, 
            False, 
            self.dayCounter
            ) 
        self.helpers.append(helper)

    # FraRateHelper(Rate rate, Natural monthsToStart, const shared_ptr<IborIndex> &iborIndex)
    def AddFRA(self, rate, monthsToStart, iborIndex):
        helper = ql.FraRateHelper(rate, monthsToStart, iborIndex)
        self.helpers.append(helper)
    
    # (Real price, const Date &iborStartDate, const ext::shared_ptr<IborIndex> &iborIndex) 
    def AddFuture(self, price, iborStartDate, lengthInMonths, convention):
        qlPrice = ql.QuoteHandle(ql.SimpleQuote(price))
        
        helper = ql.FuturesRateHelper(qlPrice, 
                                      iborStartDate, 
                                      lengthInMonths, 
                                      self.calendar, 
                                      convention, 
                                      False, 
                                      self.dayCounter)
        #helper = ql.FuturesRateHelper(qlPrice, iborStartDate, iborIndex)
         
        self.helpers.append(helper)
    
    # SwapRateHelper(Rate rate, const Period &tenor, const Calendar &calendar, 
    # Frequency fixedFrequency, BusinessDayConvention fixedConvention, const DayCounter &fixedDayCount, 
    # const shared_ptr<IborIndex> &iborIndex)
    def AddSwap(self, rate, periodLength, fixedCalendar, fixedFrequency, 
                fixedConvention, fixedDayCount, floatIndex):
        rate_asqh = ql.QuoteHandle(ql.SimpleQuote(rate))
        if self.discount_curve == None:
            helper = ql.SwapRateHelper(rate_asqh, periodLength, 
                                       fixedCalendar, fixedFrequency, 
                                       fixedConvention, fixedDayCount, 
                                       floatIndex)
        else:
            helper = ql.SwapRateHelper(
                rate_asqh,
                periodLength, 
                fixedCalendar,
                fixedFrequency, 
                fixedConvention,
                fixedDayCount, 
                floatIndex, 
                ql.QuoteHandle(), 
                ql.Period(2, ql.Days),
                self.discount_curve)

        self.helpers.append(helper)
    
    # PiecewiseYieldCurve <ZeroYield, Cubic>
    def GetCurveHandle(self):
        yts = ql.PiecewiseNaturalLogCubicDiscount(self.settleDays, 
                                                  self.calendar, 
                                                  self.helpers, 
                                                  self.dayCounter)
        ryts = ql.RelinkableYieldTermStructureHandle()
        ryts.linkTo(yts)
        return ryts
    
    
    
#%% Check
import pandas as pd
from datetime import date as dt

# Function to import input data tickers from trading file
def import_data_trading(str_file):
    dic_data = {}
    lst_names = ['USD_OIS', 'USD_SOFR','USDMXN_Fwds',
                 'USDMXN_XCCY_Basis', 'MXN_TIIE']
    for sheet in lst_names:
        dic_data[sheet] = pd.read_excel(str_file, sheet)

    return dic_data

def pull_data(str_file, dt_today):
    #dic_data = import_data(str_file)
    dic_data = import_data_trading(str_file)
    # db_cme = pd.read_excel(r'E:\db_cme' + r'.xlsx', 'db').set_index('TENOR')
    db_cme = pd.read_excel(r'\\tlaloc\tiie\db_cme' + r'.xlsx', 'db').set_index('TENOR')
    db_cme.columns = db_cme.columns.astype(str)
    # db_crvs = pd.read_excel(r'E:\db_Curves_mkt' + r'.xlsx', 
    db_crvs = pd.read_excel(r'\\tlaloc\tiie\db_Curves_mkt' + r'.xlsx', 
                            'bgnPull', 
                            skiprows=3).drop([0,1]).\
        reset_index(drop=True).set_index('Date')
    # # USD Curves Data
    datakeys = ['USD_OIS', 'USD_SOFR','USDMXN_Fwds', 'USDMXN_XCCY_Basis']
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
        
    dic_data['USDMXN_XCCY_Basis'].loc[0,'Quotes'] = \
        db_cme.loc['FX.USD.MXN', str(dt_today)+' 00:00:00']
    idx_qts = np.where(dic_data['USDMXN_XCCY_Basis'].columns == 'Quotes')[0][0]
    dic_data['USDMXN_XCCY_Basis'].iloc[-9:,idx_qts] = \
        db_cme.iloc[-9:, 
                    np.where(
                        db_cme.columns == str(dt_today)+' 00:00:00')[0][0]].\
            values*100
    dic_data['MXN_TIIE'] = dic_data['MXN_TIIE'].iloc[0:14,0:5]
    dic_data['MXN_TIIE']['Quotes'] = \
        db_cme.iloc[:14, 
                    np.where(
                        db_cme.columns == str(dt_today)+' 00:00:00')[0][0]].\
                        values
            
    return dic_data

if __name__ != '__main__':
    # globals
    tenor2ql = {'B':ql.Days,'D':ql.Days,'M':ql.Months,'W':ql.Weeks,'Y':ql.Years}
    futMty2Month = {'U': 9, 'Z': 12, 'H': 3, 'M': 6}
    # market data path
    xpath = r'\\tlaloc\cuantitativa\Fixed Income\TIIE IRS Valuation Tool\Arnua'
    fname = r'\TIIE_CurveCreate_Inputs.xlsx'
    # market data
    dic_mkt = pull_data(xpath+fname, dt.today())
    # general parameters    
    tradeDate = ql.Date(dt.today().day, dt.today().month, dt.today().year)
    cal_us, cal_mx = ql.UnitedStates(0), ql.Mexico(0)
    dc_A360 = ql.Actual360()
    convention = ql.ModifiedFollowing
    settlementDate = cal_us.advance(tradeDate, ql.Period(2, ql.Days), convention)  
    sidxFF = ql.FedFunds()
    frequency = ql.Annual
    
    # valuation date
    ql.Settings.instance().evaluationDate = tradeDate
    
    ###########################################################################
    # USDOIS
    # Curve builder
    USDOIS_PCB = PiecewiseCurveBuilder(0, cal_us, dc_A360)
    # market
    mkt_per = dic_mkt['USD_OIS']['Period'].tolist()
    mkt_per_type = dic_mkt['USD_OIS']['Tenors'].str[-1].map(tenor2ql).to_list()
    mkt_quote = dic_mkt['USD_OIS']['Quotes'].tolist()
    
    # cash deposit
    depos = []
    depos.append((mkt_quote[0]/100, ql.Period(1,ql.Days), ql.ModifiedFollowing))
    [USDOIS_PCB.AddDeposit(d[0], d[1], d[2]) for d in depos]
    # swaps
    swaps = []
    for i in range(1,len(mkt_quote)):
        tmpPer = ql.Period(mkt_per[i], mkt_per_type[i])
        tmptpl = (mkt_quote[i]/100, tmpPer, cal_us, frequency, convention, dc_A360, sidxFF)
        swaps.append(tmptpl)
    
    [USDOIS_PCB.AddSwap(s[0], s[1], s[2], s[3], s[4], s[5], s[6]) for s in swaps]
    
    # get relinkable curve handle from builder
    crvUSDOIS = USDOIS_PCB.GetCurveHandle()
    crvUSDOIS.enableExtrapolation()
    
    # USDOIS Fwd curve
    enddts = []
    fwdrates = []
    qldt1 = cal_us.advance(tradeDate,ql.Period(2,ql.Days))
    for d in range(0,30):
        qldt2 = cal_us.advance(qldt1,ql.Period(1,ql.Years))
        fwdrates.append(crvUSDOIS.forwardRate(qldt1,qldt2,dc_A360,ql.Compounded,ql.Annual).rate())
        enddts.append(qldt2.to_date())
        qldt1 = qldt2
    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(pd.DataFrame(fwdrates, index=enddts, columns=['USDOIS']),
             marker='o', mfc = 'w', mec = 'darkcyan')
    plt.tight_layout(); plt.show()
    
    ###########################################################################
    # SOFR
    # Curve builder
    sidxSOFR = ql.Sofr()
    # SOFR Ibor Index
    sofrPCB = PiecewiseCurveBuilder(0, cal_us, dc_A360, crvUSDOIS)
    # settlement date
    #qdtSTTL = cal_us.advance(tradeDate ,ql.Period('2D'))
    
    # market
    idx_fut = np.where(~dic_mkt['USD_SOFR']['Period'].str.find('F').isna())
    mkt_fut = dic_mkt['USD_SOFR'].iloc[idx_fut[0]]
    mkt_ = dic_mkt['USD_SOFR'].drop(idx_fut[0])
    
    # Deposit
    depos = []
    depos.append((mkt_['Quotes'].iloc[0]/100, ql.Period(3,ql.Months), ql.ModifiedFollowing))
    [sofrPCB.AddDeposit(d[0], d[1], d[2]) for d in depos]
    
    # Futures
    futures = []
    for i,r in mkt_fut.iterrows():
        # Fetch IMM maturity
        tmpM = futMty2Month[r['Tenors'][-2:][0]]
        tmpY = int(r['Tenors'][-2:][1])+2020
        imm_stDate = ql.IMM.nextDate(ql.Date(1,tmpM,tmpY))
        # Future tuple characteristics
        #tmpTpl = (1-r['Quotes']/100, imm_mty, sidxSOFR)
        tmpTpl = (r['Quotes'], imm_stDate, 3, ql.ModifiedFollowing)
        futures.append(tmpTpl)

    [sofrPCB.AddFuture(f[0], f[1], f[2], f[3]) for f in futures]
    
    # Swaps
    swaps = []
    for i,r in mkt_.iloc[1:].iterrows():
        ql_per = r['Period']
        ql_per_type = tenor2ql[r['Tenors'][-1]]
        tmpQuote = r['Quotes']/100
        tmpPer = ql.Period(ql_per, ql_per_type)
        tmptpl = (tmpQuote, tmpPer, cal_us, frequency, convention, dc_A360, sidxSOFR)
        swaps.append(tmptpl)
    
    [sofrPCB.AddSwap(s[0], s[1], s[2], s[3], s[4], s[5], s[6]) for s in swaps]
    
    # get relinkable curve handle from builder
    crvSOFR = sofrPCB.GetCurveHandle()
    crvSOFR.enableExtrapolation()
    
    # SOFR Fwd curve
    enddts = []
    fwdrates = []
    qldt1 = cal_us.advance(tradeDate,ql.Period(2,ql.Days))
    for d in range(0,30):
        qldt2 = cal_us.advance(qldt1,ql.Period(1,ql.Years))
        fwdrates.append(crvSOFR.forwardRate(qldt1,qldt2,dc_A360,ql.Simple,ql.Annual).rate())
        enddts.append(qldt2.to_date())
        qldt1 = qldt2
    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(pd.DataFrame(fwdrates, index=enddts, columns=['SOFR']),
             marker='o', mfc = 'w', mec = 'darkcyan')
    plt.tight_layout(); plt.show()
    
