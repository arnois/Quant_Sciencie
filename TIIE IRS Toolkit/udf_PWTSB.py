
# -*- coding: utf-8 -*-
"""
Builder for Piecewise Term Structure


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
#%% Convert Class
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

#%% Piecewise Curve Builder Class
class PiecewiseCurveBuilder_OIS:
    
    # Constructor
    def __init__(self, market_data: pd.DataFrame,
                 market_calendar: object = ql.UnitedStates(5),
                 dayCount: object = ql.Actual360(),
                 fixingDays: int = 2,
                 busDayAdj: int = ql.ModifiedFollowing,
                 rollConvEOM: bool = False,
                 iborIdx: object = ql.FedFunds()) -> None:
        """
        Parameters
        ----------
        market_data : pd.DataFrame
            USDOIS Market data with at least Tenors, Quotes and Period feats.
            Tenors as str for maturity: days(B), weeks(W), months(M), years(Y).
            Period as int for maturity amount.
            Quotes as float for swap rate (%)
            -------------------------
            Example:
            -------------------------
               Tenors  Period  Quotes
                   1W       1  5.3295
                   6M       6  5.2976
                  12M      12  5.1458
                   2Y       2  4.7394
                   5Y       5  4.2293
                  10Y      10  4.0795
            -------------------------
        market_calendar : object, optional
            Calendar to follow. Default is ql.UnitedStates(5)
        dayCount : object, optional
            Day count convention for swap coupons. Default is ql.Actual360().
        fixingDays : int, optional
            Fixing days convention for floating leg coupons. Default is 2.
        busDayAdj : int, optional
            Business day adjustment in holiday. Defaults2ql.ModifiedFollowing.
        rollConvEOM : bool, optional
            To use or not end of month convetion for MTY. Default is False.
        iborIdx : object, optional
            Floating leg rate index. The default is ql.FedFunds().

        Returns
        -------
        None
        """
        # Constructor attributes
        self.marketData = market_data
        self.marketCal = market_calendar
        self.dayCount = dayCount
        self.fixingDays = fixingDays
        self.busDayAdj = busDayAdj
        self.rollConvEOM = rollConvEOM
        self.iborIdx = iborIdx
        # Method's attributes
        self.helpers =  None
        self.crv = None
        
        return None        
    
    # Bootstrapping objects for USDOIS curve
    def set_qlHelper_USDOIS(self) -> None:
        """Creates helpers to bootstrap USDOIS curve."""
        # Mkt data
        df = self.marketData
        dcols = df.columns.tolist()
        areCols = all([elem in dcols for elem in ['Tenors','Period','Quotes']])
        if not areCols:
            print("Market data not provided as needed! [Tenors, Period, Quotes]")
            return None
        
        # Mkt calendar
        cal = self.marketCal
        
        # Input mkt data segregated
        tenor = df['Tenors'].str[-1].map(tenor2ql).to_list()
        period = df['Period'].to_list()
        data = (df['Quotes']/100).tolist()
        
        # Deposit rates
        deposits = {(period[0], tenor[0]): data[0]}
        # OIS rates
        n = len(period)
        swaps = {}
        for i in range(1,n):
            swaps[(period[i], tenor[i])] = data[i]
            
        # Rate Qauntlib.Quote objects
        ## desposits
        for n, unit in deposits.keys():
            deposits[(n, unit)] = ql.SimpleQuote(deposits[(n, unit)])
        ## ois rates
        for n, unit in swaps.keys():
            swaps[(n, unit)] = ql.SimpleQuote(swaps[(n, unit)])
            
        # Conventions
        dayCounter = self.dayCount
        fixingDays = self.fixingDays
        businessDayAdj = self.busDayAdj
        endOfMonth = self.rollConvEOM
        OIS_Index = self.iborIdx
        
        # Rate helpers deposits
        ## deposits
        depositHelpers = [
            ql.DepositRateHelper(
                ql.QuoteHandle(deposits[(n, unit)]), ql.Period(int(n), unit), 
                fixingDays, cal, businessDayAdj, endOfMonth, dayCounter
            )
            for (n, unit) in deposits.keys()]
        ## ois rates
        OISHelpers = [
            ql.OISRateHelper(
                fixingDays, ql.Period(int(n), unit), 
                ql.QuoteHandle(swaps[(n,unit)]), OIS_Index
            ) 
            for n, unit in swaps.keys()]
        
        # Helpers
        self.helpers = depositHelpers + OISHelpers
        
        return None
    
    # Curve Boostrapping
    def btstrap_USDOIS(self, interp_method: str = 'Linear') -> None:
        """Bootstraps the USDOIS curve.
        Parameters
        ----------
        interp_method : str, optional
            Type of interpolation. It can be 'Linear' or 'Cubic' over DFs. 
            The default is 'Linear'.

        Returns
        -------
        crvUSDOIS : ql.DiscountCurve
            Bootstrapped USDOIS curve.
            
        See Also
        --------
        set_qlHelper_USDOIS: Creates the helpers needed to bootstrap USDOIS.
        """
        # Helpers
        if self.helpers is None:
            print("Helpers still not set! See set_qlHelper_USDOIS.")
            return None
        
        # Curve bootsrapping methods
        if interp_method == 'Linear':
            crvUSDOIS = ql.PiecewiseLogLinearDiscount(0, self.marketCal, 
                                                      self.helpers, 
                                                      self.dayCount)
        else: 
            crvUSDOIS = ql.PiecewiseNaturalLogCubicDiscount(0, self.marketCal, 
                                                            self.helpers, 
                                                            self.dayCount)
        # Extrapolation enabled by default
        crvUSDOIS.enableExtrapolation()
        
        self.crv = crvUSDOIS
        
        return None
    
#%%############################################################################
# Create piecewise yield curve from swaps market
class PiecewiseCurveBuilder_SWAP:
    def __init__(self, market_data: pd.DataFrame,
                 market_calendar: object = ql.UnitedStates(5),
                 dayCount: object = ql.Actual360(),
                 pmtFreq: object = ql.Annual,
                 fixingDays: int = 2,
                 busDayAdj: int = ql.ModifiedFollowing,
                 rollConvEOM: bool = False,
                 swapIdx: object = ql.Sofr(),
                 discCrv: object = ql.YieldTermStructureHandle()) -> None:
        
        # Constructor attributes
        self.marketData = market_data
        self.marketCal = market_calendar
        self.dayCount = dayCount
        self.pmtFreq = pmtFreq
        self.fixingDays = fixingDays
        self.busDayAdj = busDayAdj
        self.rollConvEOM = rollConvEOM
        self.swapIdx = swapIdx
        self.discCrv = discCrv
        # Method's attributes
        self.helpers =  None
        self.crv = None
        
        return None
        
    
    # Bootstrapping objects for SOFR swaps curve
    def set_qlHelper_SOFR(self) -> None: 
        """Creates helpers to bootstrap USDSOFR curve."""
        # Mkt data
        df = self.marketData # dic_df['USD_SOFR']
        dcols = df.columns.tolist()
        areCols = all([elem in dcols for elem in ['Tenors','Period','Quotes']])
        if not areCols:
            print("Market data not provided as needed! [Tenors, Period, Quotes]")
            return None
        
        # Mkt calendar
        cal = self.marketCal
        # Mkt disc crv
        if type(self.discCrv) is not ql.RelinkableYieldTermStructureHandle:
            discCrv = ql.RelinkableYieldTermStructureHandle() 
            discCrv.linkTo(self.discCrv)
        else:
            discCrv = self.discCrv
        
        # Settlement date
        dt_settlement = cal.advance(ql.Settings.instance().evaluationDate, ql.Period('2D'))
        # Non-futures idx
        idx_nonfut = (df['Types'] != 'FUT')
        # Input mkt data segregated
        tenor = df['Tenors'][idx_nonfut].str[-1].map(tenor2ql).to_list()
        period = df['Period'][idx_nonfut].to_list()
        data_nonfut = (df['Quotes'][idx_nonfut]/100).tolist()
        data_fut = (df['Quotes'][~idx_nonfut]/100).tolist()
        
        # Deposit rates
        deposits = {(period[0], tenor[0]): data_nonfut[0]}
       
        # Futures rates
        n_fut = len(data_fut)
        imm = ql.IMM.nextDate(dt_settlement)
        imm = dt_settlement
        futures = {}
        for i in range(n_fut):
            imm = ql.IMM.nextDate(imm)
            futures[imm] = data_fut[i]*100 
            # futures[imm] = 100 - data_fut[i]*100  
        
        # Swap rates
        n = len(period)
        swaps = {}
        for i in range(1, n):
            swaps[(period[i], tenor[i])] = data_nonfut[i]
            
        # Rate Qauntlib.Quote objects
        ## desposits
        for n, unit in deposits.keys():
            deposits[(n, unit)] = ql.SimpleQuote(deposits[(n, unit)])
        ## futures
        for d in futures.keys():
            futures[d] = futures[d]
        ## swap rates
        for n, unit in swaps.keys():
            swaps[(n, unit)] = ql.SimpleQuote(swaps[(n, unit)])
            
        # Rate helpers deposits
        ## deposits
        depositHelpers = [ql.DepositRateHelper(ql.QuoteHandle(deposits[(n, unit)]),
                                               ql.Period(int(n), unit), 
                                               self.fixingDays,
                                               cal, 
                                               self.busDayAdj, 
                                               self.rollConvEOM, 
                                               self.dayCount) 
                          for n, unit in deposits.keys()]
        ## futures
        months = 3
        futuresHelpers = [ql.FuturesRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(futures[d])), d, months, cal, 
            self.busDayAdj, True, self.dayCount) 
            for d in futures.keys()
            ]
        
        ## OIS Helpers
        swapHelpers = [ql.OISRateHelper(
            self.fixingDays, ql.Period(n, unit), ql.QuoteHandle(swaps[(n,unit)]), 
            self.swapIdx, discCrv, False, 2, self.busDayAdj, ql.Annual, cal) 
               for n, unit in swaps.keys()]
        
        ## swaphelper
        # swapHelpers = [ql.SwapRateHelper(
        #     ql.QuoteHandle(swaps[(n,unit)]),
        #     ql.Period(int(n), unit), 
        #     cal,
        #     self.pmtFreq, 
        #     self.busDayAdj,
        #     self.dayCount, 
        #     self.swapIdx, 
        #     ql.QuoteHandle(), 
        #     ql.Period(),
        #     discCrv,
        #     2)
        #     for n, unit in swaps.keys()]

        ## helpers merge
        self.helpers = depositHelpers + futuresHelpers + swapHelpers

        return None
    
    # Curve Boostrapping
    def btstrap_USDSOFR(self, interp_method: str = 'Linear') -> None:
        """Bootstraps the USDSOFR curve.
        Parameters
        ----------
        interp_method : str, optional
            Type of interpolation. It can be 'Linear' or 'Cubic' over DFs. 
            The default is 'Linear'.

        Returns
        -------
        crvUSDSOFR : ql.YieldTermStructure
            Bootstrapped USDSOFR curve.
            
        See Also
        --------
        set_qlHelper_USDSOFR: Creates the helpers needed to bootstrap USDSOFR.
        """
        # Helpers
        if self.helpers is None:
            print("Helpers still not set! See set_qlHelper_USDSOFR.")
            return None
        
        # Curve bootsrapping methods
        if interp_method == 'Linear':
            crvUSDSOFR = ql.PiecewiseLogLinearDiscount(0, self.marketCal, 
                                                      self.helpers, 
                                                      self.dayCount)
        else: 
            crvUSDSOFR = ql.PiecewiseNaturalLogCubicDiscount(0, self.marketCal, 
                                                            self.helpers, 
                                                            self.dayCount)
        # Extrapolation enabled by default
        crvUSDSOFR.enableExtrapolation()
        
        self.crv = crvUSDSOFR
        
        return None
    
#%%############################################################################
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
    dic_data = import_data_trading(str_file)
    dbflpath = r'\\tlaloc\cuantitativa\Fixed Income\File Dump\Database'
    db_cme = pd.read_excel(dbflpath+r'\db_cme'+r'.xlsx', 'db').set_index('TENOR')
    db_cme.columns = db_cme.columns.astype(str)
    db_crvs = pd.read_excel(dbflpath+r'\db_Curves_mkt' + r'.xlsx', 
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

###############################################################################
# QuantLib's Helper object for Implied MXNOIS Crv Bootstrapping
def qlHelper_MXNOIS(dic_df, discount_curve, crv_usdswp, crvType = 'SOFR'):
    # Disc Crv YieldTermStructure
    if type(discount_curve) is not ql.RelinkableYieldTermStructureHandle:
        ryts = ql.RelinkableYieldTermStructureHandle() 
        ryts.linkTo(discount_curve)
        discount_curve = ryts
    
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
    # USD Forc Crv YieldTermStructure
    if type(crv_usdswp) is not ql.RelinkableYieldTermStructureHandle:
        rytsForc = ql.RelinkableYieldTermStructureHandle() 
        rytsForc.linkTo(crv_usdswp)
        crv_usdswp = rytsForc
        
    # Swap rate helpers
    #settlementDays = 2
    fixedLegFrequency = ql.EveryFourthWeek
    fixedLegAdjustment = ql.Following
    fixedLegDayCounter = ql.Actual360()
    if crvType == 'SOFR':
        # fxIborIndex = ql.Sofr(rytsForc) # if you set the iborIndex it should not carry any curve
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
    helpers = depositHelpers + fxSwapHelper + swapHelpers

    return(helpers)

###############################################################################
# QuantLib's Helper object for TIIE Crv Bootstrapping
def qlHelper_MXNTIIE(df, crv_MXNOIS):
    # calendar
    calendar_mx = ql.Mexico(0)
    # data
    tenor = ql.EveryFourthWeek
    period = df['Period'].astype(int).to_list()
    data = (np.round(df['Quotes']/100,8)).tolist()
    
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

    # MXNOIS YieldTermStructure
    if type(crv_MXNOIS) is not ql.RelinkableYieldTermStructureHandle:
        ryts = ql.RelinkableYieldTermStructureHandle() 
        ryts.linkTo(crv_MXNOIS)
        crv_MXNOIS = ryts

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


#%% MAIN
if __name__ == '__main__':
    
    # market data path
    xpath = r'\\tlaloc\cuantitativa\Fixed Income\TIIE IRS Valuation Tool\Arnua'
    fname = r'\TIIE_CurveCreate_Inputs.xlsx'
    # market data
    dic_mkt = pull_data(xpath+fname, dt.today())
    # general parameters    
    tradeDate = ql.Date(dt.today().day, dt.today().month, dt.today().year)
    cal_us, cal_mx = ql.UnitedStates(5), ql.Mexico(0)
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
    pcbUSDOIS = PiecewiseCurveBuilder_OIS(dic_mkt['USD_OIS'])
    pcbUSDOIS.set_qlHelper_USDOIS()
    pcbUSDOIS.btstrap_USDOIS('NLC')
    
    # USDOIS Fwd curve
    crvUSDOIS = pcbUSDOIS.crv
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
    
    # Mkt Repricing
    ois_type = -1
    ois_nom = 10e6
    ois_DC = ql.Actual360()
    qldt1 = cal_us.advance(tradeDate,ql.Period(2, ql.Days))
    rtysOIS = ql.RelinkableYieldTermStructureHandle()
    rtysOIS.linkTo(crvUSDOIS)
    ois_swp_engine = ql.DiscountingSwapEngine(rtysOIS)
    df_model_ois = pd.DataFrame()
    for i,row in dic_mkt['USD_OIS'].iterrows():
        ois_tenor = ql.Period(row['Period'], tenor2ql[row['Tenors'][-1]])
        ois_swap = ql.MakeOIS(ois_tenor, ql.FedFunds(rtysOIS), 0.04)
        ois_swap.setPricingEngine(ois_swp_engine)
        newDrow = [row['Tenors'], 100*ois_swap.fairRate()]
        df_model_ois = pd.concat([df_model_ois, pd.DataFrame(newDrow).T])
    
    # Market repricing df
    df_model_ois.columns = ['Tenors','Model']
    tmp = df_model_ois.merge(dic_mkt['USD_OIS'],
                       left_on='Tenors',
                       right_on='Tenors').set_index('Tenors').\
        apply(lambda x: np.round(100*(x['Quotes'] - x['Model']),4), axis=1)
    print("\nOIS Market Repricing\n")
    print(tmp)
    
    ###########################################################################
    # SOFR
    # PCB USDSOFR
    pcbSOFR = PiecewiseCurveBuilder_SWAP(market_data=dic_mkt['USD_SOFR'], 
                                         discCrv=pcbUSDOIS.crv)
    pcbSOFR.set_qlHelper_SOFR()
    pcbSOFR.btstrap_USDSOFR('NLC')
    
    # SOFR Fwd curve
    crvSOFR = pcbSOFR.crv
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
    
    # SOFR Fixings
    rtysSOFR = ql.RelinkableYieldTermStructureHandle()
    rtysSOFR.linkTo(crvSOFR)
    sofrIdx = ql.Sofr(rtysSOFR)
    tmpdf = pd.read_excel('C:/Users/jquintero/Downloads/fixings_SOFR.xlsx', 
                          usecols='A:B', skiprows=5)
    sofrIdx.addFixings(tmpdf.apply(lambda x: ql.Date(x['Date'].day, x['Date'].month, x['Date'].year), axis=1).tolist(), 
                       tmpdf.apply(lambda x: x['PX_LAST']/100, axis=1).tolist())
    
    # SOFR Swaps Repricing
    sofr_type = -1
    sofr_nom = 10e6
    sofr_DC = ql.Actual360()
    qldt1 = cal_us.advance(tradeDate,ql.Period(2, ql.Days))
    df_model_sofr = pd.DataFrame()
    for i,row in dic_mkt['USD_SOFR'][dic_mkt['USD_SOFR']['Types'] != 'FUT'].iterrows():
        sofr_tenor = ql.Period(row['Period'], tenor2ql[row['Tenors'][-1]])
        sofr_swap = ql.MakeOIS(sofr_tenor, ql.Sofr(rtysSOFR), 0.04)
        sofr_swap.setPricingEngine(ois_swp_engine)
        newDrow = [row['Tenors'], 100*sofr_swap.fairRate()]
        df_model_sofr = pd.concat([df_model_sofr, pd.DataFrame(newDrow).T])
        
    # SOFR Futures Repricing
    sofrFutMtyCode = {'H':6, 'M':9, 'U':12, 'Z':1}
    df_model_sofrF = pd.DataFrame()
    for i, row in dic_mkt['USD_SOFR'][dic_mkt['USD_SOFR']['Types'] == 'FUT'].iterrows():
        imm_start = ql.IMM.date(row['Tenors'][-2:], ql.Date(1,1,tradeDate.year()))
        if imm_start < qldt1: imm_start = qldt1
        imm_M = int(sofrFutMtyCode[row['Tenors'][-2:][0]])
        imm_Y = 2020+int(row['Tenors'][-1])
        imm_qldt = ql.IMM.nextDate(ql.Date(1,imm_M,imm_Y))
        imm_end = ql.IMM.nextDate(imm_start)
        #sofr_fut = ql.OvernightIndexFuture(ql.Sofr(rtysSOFR), qldt1, imm_qldt)
        #sofr_fut.setPricingEngine(ois_swp_engine)
        #sofr_fut = crvSOFR.forwardRate(imm_start, imm_end, sofr_DC, ql.Compounded, ql.Daily)
        sofr_fut = ql.ForwardRateAgreement(sofrIdx, imm_start, imm_end, -1, 0.04, 100).forwardRate()
        newDrow = [row['Tenors'], 100-100*sofr_fut.rate()]
        df_model_sofrF = pd.concat([df_model_sofrF, pd.DataFrame(newDrow).T])
        
    
    # Market repricing df
    df_model_sofrF.columns = ['Tenors','Model']
    df_model_sofrF.merge(dic_mkt['USD_SOFR'],
                       left_on='Tenors',
                       right_on='Tenors').set_index('Tenors')[['Model','Quotes']]
    df_model_sofr.columns = ['Tenors','Model']
    tmp = df_model_sofr.merge(dic_mkt['USD_SOFR'],
                       left_on='Tenors',
                       right_on='Tenors').set_index('Tenors').\
        apply(lambda x: np.round(100*(x['Quotes'] - x['Model']),4), axis=1)
    print("\nSOFR Market Repricing\n")
    print(tmp)

    ###########################################################################
    # MXNOIS
    # Curve builder
    hlprMXNOIS = qlHelper_MXNOIS(dic_mkt, crvUSDOIS, crvSOFR)
    crvMXNOIS = ql.PiecewiseNaturalLogCubicDiscount(0, ql.Mexico(0), 
                                                   hlprMXNOIS, 
                                                   ql.Actual360())
    crvMXNOIS.enableExtrapolation()
    
    # MXNOIS Fwd curve
    enddts = []
    fwdrates = []
    settleDate = cal_mx.advance(tradeDate,ql.Period(1,ql.Days))
    qldt1 = cal_mx.advance(tradeDate,ql.Period(1,ql.Days))
    for d in range(0,130):
        qldt2 = cal_mx.advance(qldt1, ql.Period(4, ql.Weeks))
        fwdrates.append(crvMXNOIS.forwardRate(qldt1,qldt2,dc_A360,ql.Simple).rate())
        enddts.append(qldt2.to_date())
        qldt1 = qldt2
    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(pd.DataFrame(fwdrates, index=enddts, columns=['MXNOIS']),
             marker='o', mfc = 'w', mec = 'darkcyan')
    plt.tight_layout(); plt.show()
    
    ###########################################################################
    # MXN TIIE
    # Vanilla Swap for TIIE
    def tiieSwap(start, maturity, notional, ibor_tiie, rate, typ, rule):
        # TIIE Swap Schedule Specs
        cal = ql.Mexico(0)
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
    
    # Curve builder
    hlprTIIE = qlHelper_MXNTIIE(dic_mkt['MXN_TIIE'], crvMXNOIS)
    crvTIIE = ql.PiecewiseNaturalLogCubicDiscount(0, cal_mx, hlprTIIE, 
                                                  ql.Actual360())
    crvTIIE.enableExtrapolation()
    
    # TIIE Fwd curve
    enddts = []
    fwdrates = []
    qldt1 = cal_mx.advance(tradeDate,ql.Period(1,ql.Days))
    for d in range(0,130):
        qldt2 = cal_mx.advance(qldt1, ql.Period(4, ql.Weeks))
        fwdrates.append(crvTIIE.forwardRate(qldt1,qldt2,dc_A360,ql.Simple).rate())
        enddts.append(qldt2.to_date())
        qldt1 = qldt2
    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(pd.DataFrame(fwdrates, index=enddts, columns=['MXNOIS']),
             marker='o', mfc = 'w', mec = 'darkcyan')
    plt.tight_layout(); plt.show()
    
    # Repricing Mkt
    tiie_disc_engine = ql.DiscountingSwapEngine(ql.YieldTermStructureHandle(crvMXNOIS))
    ibor_tiie_crv = ql.RelinkableYieldTermStructureHandle()
    ibor_tiie_crv.linkTo(crvTIIE)
    _start = cal_mx.advance(tradeDate,ql.Period(1,ql.Days))
    ibor_tiie = ql.IborIndex('TIIE', ql.Period(13), 1, ql.MXNCurrency(), cal_mx,
                             ql.Following, False, ql.Actual360(), ibor_tiie_crv)
    df_model_tiie = pd.DataFrame()
    for i, row in dic_mkt['MXN_TIIE'].iterrows():
        _end = cal_mx.advance(_start, ql.Period(4*row['Period'], ql.Weeks))
        tiie_swap = tiieSwap(_start, _end, 100e6, ibor_tiie, 0.04, -1, 0)
        tiie_swap.setPricingEngine(tiie_disc_engine)
        newDrow = [row['Tenor'], 100*tiie_swap.fairRate()]
        df_model_tiie = pd.concat([df_model_tiie, pd.DataFrame(newDrow).T])
        
    df_model_tiie.columns = ['Tenor','Model']
    df_mkt = df_model_tiie.merge(dic_mkt['MXN_TIIE'],
                       left_on='Tenor',
                       right_on='Tenor').set_index('Tenor')
    df_mkt['Mispx'] = df_mkt.apply(lambda x: np.round(100*(x['Quotes'] - x['Model']),6), axis=1)
    
    # TIIE Curve Plot
    plt.figure(figsize=(10,6))
    df_mkt[['Quotes','Model']].plot(kind='line', style=['-.', '-^'], 
                                    color=['darkcyan', 'orange'])
    plt.tight_layout(); plt.show()
    
    print(f"\nTIIE SWAPS\n{df_mkt[['Quotes','Model','Mispx']]}")

    
   
        
        
        
        
    