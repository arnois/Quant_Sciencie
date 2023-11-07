#%% MODULES
import numpy as np
import pandas as pd
import datetime as dt
import QuantLib as ql
from matplotlib import pyplot as plt
import holidays
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter

#%%############################################################################
# SWAPS
###############################################################################

# Function to create a fixed vs float swap cash flows
def swap_fxd_flt(start_date: dt.date = dt.date(2022,3,8), 
                 mty_date: dt.date = dt.date(2024,3,8), freq: str = '90D', 
                 notional: float = 100, swap_rate: float = 0.03, 
                 yearbase_conv: int = 360, cal = holidays.US()) -> pd.DataFrame:
    """
    Create swap cash flows.

    Args:
    - start_date (dt.date): Start date of the swap.
    - mty_date (dt.date): Maturity date of the swap.
    - freq (str): Payment frequency; D for days.
    - notional (float): Notional or principal value in millions.
    - swap_rate (float): Fixed rate.
    - yearbase_conv (int): Days in a year convention for periods.
    - cal: Holidays calendar to use.

    Returns:
        (pd.DataFrame) Swap characteristics table.
    """
   
    # Payment dates
    CF_range = pd.bdate_range(start=start_date, end=mty_date, freq=freq, 
                              holidays=cal).sort_values()
    CF_EndDates = [x.date() for x in CF_range.tolist()[1:]]
    CF_StartDates = [x.date() for x in CF_range.tolist()[:-1]]

    # Swap specs dataframe
    df_swp_specs = pd.DataFrame({
                           'AccStartDate': CF_StartDates,
                           'AccEndDate': CF_EndDates})
    df_swp_specs.iloc[-1,1] = mty_date
    df_swp_specs['AccDays'] = df_swp_specs.\
        apply(lambda x: (x['AccEndDate'] - x['AccStartDate']).days, axis=1)
    df_swp_specs['Notional'] = notional
    df_swp_specs['T'] = df_swp_specs['AccDays']/yearbase_conv
    df_swp_specs['FixedRate'] = swap_rate
    df_swp_specs['FixedPmt'] =  swap_rate*df_swp_specs['T']*notional

    return df_swp_specs

"""
Lets create a swap table info
"""
swap_info = swap_fxd_flt()
print(swap_info)

# Lets assume we have the following zero curve for SOFR
df_SOFR3M_Z = pd.DataFrame({'mtyDate': [dt.date(2022,6,6), dt.date(2022,9,4),
                                        dt.date(2022,12,3), dt.date(2023,3,3),
                                        dt.date(2023,6,1), dt.date(2023,8,30),
                                        dt.date(2023,11,28), dt.date(2024,3,8)],
                            'Z': [0.022, 0.0241, 0.0255, 0.027, 
                                  0.0284, 0.0295, 0.0307, 0.0316]})
df_SOFR3M_Z.insert(0,'dtm', [(x - dt.date(2022,3,8)).days for x in df_SOFR3M_Z['mtyDate']])
df_SOFR3M_Z.plot(x='mtyDate', y='Z')

# We then have the following forward rates # df_SOFR3M = np.array([0.022, 0.026, 0.028, 0.031, 0.033, 0.034, 0.036, 0.038])
df_SOFR3M_F = (((1+df_SOFR3M_Z['Z']*df_SOFR3M_Z['dtm']/360)/\
               ((1+df_SOFR3M_Z['Z']*df_SOFR3M_Z['dtm']/360).shift().fillna(1))-1)*\
                360/90).rename('FwdRate').to_frame()

df_SOFR3M_F.insert(0,'AccStartDate', [dt.date(2022,3,8)]+df_SOFR3M_Z['mtyDate'].iloc[:-1].tolist())
df_SOFR3M_F.insert(1,'AccEndDate', df_SOFR3M_Z['mtyDate'].tolist())
df_SOFR3M_F.plot(x='AccStartDate', y='FwdRate')
# Now lets assume we have the following fwd rates as the swap's floating rates
swap_info['FloatRate'] = df_SOFR3M_F['FwdRate']

# Lets compute floating payments
swap_info['FloatPmt'] = swap_info['Notional']*swap_info['FloatRate']*swap_info['T']

# Lets compute net cash flows
swap_info['NetCF'] = swap_info['FixedPmt'] - swap_info['FloatPmt']
print(swap_info)

# Lets plot net cash flows
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(swap_info['AccEndDate'], 1e6*swap_info['NetCF'], width = 45)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(0,3,6,9,12)))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))
plt.title('Swap Net Payments',size=16)
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.tight_layout(); plt.show()

"""
Lets add discount factors for computing the swap's NPV
"""
# Lets assume we have the following zero curve for OIS
crv_Z = pd.DataFrame({'dtm': [1,30,90,180,360,520,1800],
                      'z': [0.015, 0.01798, 0.02, 0.0218, 
                            0.0247, 0.0299, 0.0404]})

# Lets add dtm to swap info df
valuation_date = dt.date(2022,3,7)
swap_info['dtm'] = swap_info['AccEndDate'].\
    apply(lambda x: (x-valuation_date).days)

# Lets interpolate corresponding zero rates and add the discount factors
interpol_z = np.interp(swap_info['dtm'].to_numpy(), crv_Z['dtm'], crv_Z['z'])
swap_info['DF'] = 1/(1+interpol_z*swap_info['dtm']/360)

# Swap's NPV
print(f"Swap NPV: {(swap_info['DF']*swap_info['NetCF']).sum(): ,.3f} millions")

#%% SWAPS w/QuantLib

###############################################################################
# USDOIS
# Example market for USDOIS swaps
mkt_OIS = pd.DataFrame({
    'Ticker': ['FEDL01', 'USSOC', 'USSOF', 'USSOI', 'USSO1', 'USSO2',
               'USSO3', 'USSO4', 'USSO5', 'USSO10'],
    'Type': ['DEPO', 'OIS', 'OIS', 'OIS', 'OIS', 'OIS', 'OIS', 'OIS', 'OIS', 'OIS'],
    'Tenor': [ql.Days, ql.Months, ql.Months, ql.Months, ql.Months, ql.Years, 
              ql.Years, ql.Years, ql.Years, ql.Years],
    'Period': [1,3,6,9,12,2,3,4,5,10],
    'Quote': [1.25, 1.34, 1.36, 1.37, 1.40, 1.57, 1.972, 2.23, 2.30, 2.75]
    })


# QuantLib's Helper object for USDOIS Crv Bootstrapping
def qlHelper_USDOIS(market_data: pd.DataFrame = mkt_OIS) -> list:
    """
    Create object to bootstrap discount curve from USDOIS market.

    Args:
    - market_data (pd.DataFrame): Dataframe with the market rates data. It 
    should come at least with 4 columns: 
        Type = Instrument type
        Tenor = Tenor unit.
        Period = Amount of tenor units.
        Quote = Market quote.

    Returns:
        (list) List with the rate helpers to bootstrap USDOIS curve.
    """
    
    # Calendar
    calendar = ql.UnitedStates(0)
    # Market data
    mkt_depo = market_data[['Period','Tenor','Quote']][market_data['Type'] == 'DEPO']
    mkt_ois = market_data[['Period','Tenor','Quote']][market_data['Type'] == 'OIS']
    
    # Rates from Deposit markets
    deposits = {(int(mkt_depo['Period'][0]), int(mkt_depo['Tenor'][0])): mkt_depo['Quote'][0]/100}
    # Swap rates
    n = mkt_ois.shape[0]
    swaps = {}
    for i in range(1,n):
        swaps[(int(mkt_ois.iloc[i]['Period']), int(mkt_ois.iloc[i]['Tenor']))] = mkt_ois.iloc[i]['Quote']/100
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

# Example of a discount curve for USDSOFR swaps
eg_crv_disc = ql.PiecewiseLogLinearDiscount(0,ql.UnitedStates(0), qlHelper_USDOIS(), ql.Actual360())
crv_disc = ql.RelinkableYieldTermStructureHandle()
crv_disc.linkTo(eg_crv_disc)


###############################################################################
# USDSOFR

# Example market for USDSOFR swaps
mkt_SOFR = pd.DataFrame({
    'Ticker': ['SOFR Index', 'SFR1 Comdty', 'SFR2 Comdty', 'SFR3 Comdty', 
               'SFR4 Comdty', 'USOSFR2 Curncy', 'USOSFR3 Curncy', 
               'USOSFR4 Curncy', 'USOSFR5 Curncy', 'USOSFR10 Curncy'],
    'Type': ['DEPO', 'FUT', 'FUT', 'FUT', 'FUT', 'SWAP', 'SWAP', 'SWAP', 'SWAP', 'SWAP'],
    'Tenor': [ql.Days, ql.Months, ql.Months, ql.Months, ql.Months, ql.Years, 
              ql.Years, ql.Years, ql.Years, ql.Years],
    'Period': [1,3,6,9,12,2,3,4,5,10],
    'Quote': [1.28, 1.35, 1.37, 1.40, 1.985, 2.21, 2.28, 2.37, 2.45, 3.205]
    })


# QuantLib's Helper object for USDSOFR Crv Bootstrapping 
def qlHelper_SOFR(market_data: pd.DataFrame = mkt_SOFR, 
                  discount_curve = crv_disc) -> list:
    """
    Create object to bootstrap discount curve from USDOIS market.

    Args:
    - market_data (pd.DataFrame): Dataframe with the market rates data. It 
    should come at least with 4 columns: 
        Type = Instrument type
        Tenor = Tenor unit.
        Period = Amount of tenor units.
        Quote = Market quote.
    - discount_curve (): Discount curve for discounting cash flows of USDSOFR
        swaps.

    Returns:
        (list) List with the rate helpers to bootstrap USDSOFR curve.
    """
    # Calendar
    calendar = ql.UnitedStates(0)
    # Market data
    mkt_depo = market_data[['Period','Tenor','Quote']][market_data['Type'] == 'DEPO']
    mkt_fut = market_data[['Period','Tenor','Quote']][market_data['Type'] == 'FUT']
    mkt_swap = market_data[['Period','Tenor','Quote']][market_data['Type'] == 'SWAP']
    
    # Valuation date
    ql_val_date = ql.Settings.instance().evaluationDate
    
    ## Rates from Deposit markets
    deposits = {(int(mkt_depo['Period'][0]), int(mkt_depo['Tenor'][0])): mkt_depo['Quote'][0]/100}
    ## Rates from Futures markets
    n_fut = mkt_fut.shape[0]
    imm = ql_val_date
    futures = {}
    for i in range(n_fut):
        imm = ql.IMM.nextDate(imm)
        futures[imm] = 100 - mkt_fut.iloc[i]['Quote']/1
    ## Rates from Swaps markets
    n_swps = mkt_swap.shape[0]
    swaps = {}
    for i in range(1,n_swps):
        swaps[(int(mkt_swap.iloc[i]['Period']), int(mkt_swap.iloc[i]['Tenor']))] = mkt_swap.iloc[i]['Quote']/100
        
    ## Rate Qauntlib.Quote objects
    ### Deposits
    for n, unit in deposits.keys():
        deposits[(n,unit)] = ql.SimpleQuote(deposits[(n,unit)])
    ### Futures
    for d in futures.keys():
        futures[d] = futures[d]
    ### Swaps
    for n,unit in swaps.keys():
        swaps[(n,unit)] = ql.SimpleQuote(swaps[(n,unit)])
    
    # IborIndex
    swapIndex = ql.Sofr()
        
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
###############################################################################
# QL Docs: https://quantlib-python-docs.readthedocs.io/en/latest/index.html
"""
Lets create a USDSOFR swap
"""
# Set valuation date
ql_val_date = ql.Date(4,3,2022)
ql.Settings.instance().evaluationDate = ql_val_date

# First, we need our discount curve from the USDOIS market
## Get USDOIS Rates Helpers
helpers_USDOIS = qlHelper_USDOIS() 
## Bootstrap the USDOIS Market Curve
bootCrv_USDOIS = ql.\
    PiecewiseNaturalLogCubicDiscount(0, # referenceDate
                                     ql.UnitedStates(0), # cal
                                     helpers_USDOIS, # helpers - mkt
                                     ql.Actual360()) # dayCounter
## Set the bootstrapped curve ready to be used
crv_USDOIS = ql.RelinkableYieldTermStructureHandle()
crv_USDOIS.linkTo(bootCrv_USDOIS)

# Second, we need our forward rates curve from the USDSOFR swap's market
## Get USDSOFR Rates Helpers
helpers_USDSOFR = qlHelper_SOFR() # mkt_SOFR_2 = mkt_SOFR.copy()
## Bootstrap the USDSOFR Market Curve with OIS discounting
bootCrv_USDSOFR = ql.\
    PiecewiseNaturalLogCubicDiscount(0,
                                     ql.UnitedStates(0), 
                                     helpers_USDSOFR, 
                                     ql.Actual360())
## Set the bootstrapped curve ready to be used
crv_USDSOFR = ql.RelinkableYieldTermStructureHandle()
crv_USDSOFR.linkTo(bootCrv_USDSOFR)


# Create SOFR swap schedules
ql_cal = ql.UnitedStates(0)
ql_startDate = ql.Date(8,3,2022)
ql_mtyDate = ql.Date(8,3,2028)
fxd_leg_tenor = ql.Period(1, ql.Years)
flt_leg_tenor = ql.Period(1, ql.Years)
non_workday_adj = ql.ModifiedFollowing
mtyDate_adj = ql.ModifiedFollowing
dates_gen_rule = ql.DateGeneration.Backward
eom_adj = False
fxd_schdl = ql.Schedule(ql_startDate, # settleDate
                        ql_mtyDate, # maturityDate
                        fxd_leg_tenor, # Fixed Leg Pmt Freq
                        ql_cal, # Fixed Leg Calendar
                        non_workday_adj, # day adj convention
                        mtyDate_adj, # mtyDate convention
                        dates_gen_rule, # pmt dates generation
                        eom_adj) # adj EoM
flt_schdl = ql.Schedule(ql_startDate, 
                        ql_mtyDate, 
                        flt_leg_tenor, 
                        ql_cal, 
                        non_workday_adj, 
                        mtyDate_adj,
                        dates_gen_rule, 
                        eom_adj)

# Create SOFR swap floating rate index
swap_index = ql.Sofr(crv_USDSOFR)

# Create SOFR IRS object
notional = 100e6
fxd_rate = 0.0295
fxd_leg_daycount = ql.Actual360()
flt_spread = 0
flt_leg_daycount = ql.Actual360()
swap_position = ql.VanillaSwap.Receiver
## USDSOFR IRS
ql_swap = ql.VanillaSwap(swap_position, # position type
                         notional, # Nominal value
                         fxd_schdl, # Fixed Leg Pmt Dates
                         fxd_rate, # Fixed Leg Rate
                         fxd_leg_daycount, # Fixed Leg DayCount convention
                         flt_schdl, # Float Leg Pmt Dates
                         swap_index, # Float Leg Rate Index
                         flt_spread, # Float Rate Spread
                         flt_leg_daycount) # Float Leg DayCount convention

# Swap fixed leg cash flows
ql_swap_cf_fxd = pd.DataFrame({
    'accStartDate': cf.accrualStartDate().ISO(),
    'accEndDate': cf.accrualEndDate().ISO(),
    'accDays': cf.accrualDays(),
    'T': cf.accrualPeriod(),
    'Notional': cf.nominal(),
    'FxdRate': cf.rate(),
    'FxdPmt': cf.amount()
    } for cf in map(ql.as_coupon, ql_swap.leg(0)))

# Swap floating leg cash flows
ql_swap_cf_flt = pd.DataFrame({
    'accStartDate': cf.accrualStartDate().ISO(),
    'accEndDate': cf.accrualEndDate().ISO(),
    'accDays': cf.accrualDays(),
    'T': cf.accrualPeriod(),
    'Notional': cf.nominal(),
    'FltRate': cf.rate(),
    'FltPmt': cf.amount()
    } for cf in map(ql.as_coupon, ql_swap.leg(1)))


# Swap cash flow details
df_swap_des = ql_swap_cf_fxd.merge(ql_swap_cf_flt[['accEndDate','FltRate','FltPmt']], 
                                   how='outer',
                                   left_on='accEndDate', right_on='accEndDate', 
                                   suffixes=('_Fxd', '_Flt'))
print(df_swap_des)

# Swap Net CF
df_swap_des['NetCF'] = df_swap_des['FxdPmt'] - df_swap_des['FltPmt']

# Swap discounting factors
tmp_DF = []
for i,r in df_swap_des.iterrows():
    tmp_qlDt = ql.DateParser.parseFormatted(r['accEndDate'], '%Y-%m-%d')
    df = crv_USDOIS.discount(tmp_qlDt)
    tmp_DF.append(df)
df_swap_des['DF'] = tmp_DF
print(df_swap_des)

"""
Lets price our previous swap
"""
# We need to set the pricing engine as discounting with the USDOIS curve
swap_pricing_eng = ql.DiscountingSwapEngine(crv_USDOIS)

# Swap pricing...
ql_swap.setPricingEngine(swap_pricing_eng)

print(f'\nUSDSOFR 6Y Receiver Swap at {fxd_rate:.3%}')
print(f'\tNPV: {ql_swap.NPV():,.2f}')

# Swap fair rate (Swap rate)
print(f'\tSwap Rate: {ql_swap.fairRate():.3%}')


#%% MXN CASE -- TIIE IRS --- ##################################################

# Example market for TIIE swaps
mkt_TIIE = pd.DataFrame({
    'Ticker': ['TIIE Index', 'MPSWC', 'MPSWF', 'MPSWI', 'MPSW1A', 'MPSW2B', 
               'MPSW3C', 'MPSW4D', 'MPSW5E', 'MPSW7G', 'MPSW10J', 'MPSW16C',
               'MPSW21H', 'MPSW32F'],
    'Type': ['DEPO', 'SWAP', 'SWAP', 'SWAP', 'SWAP', 'SWAP', 'SWAP', 
             'SWAP', 'SWAP', 'SWAP', 'SWAP', 'SWAP', 'SWAP', 'SWAP'],
    'Tenor': np.repeat(ql.EveryFourthWeek,14),
    'Period': [1, 3, 6, 9, 13, 26, 39, 52, 65, 91, 130, 195, 260, 390],
    'Quote': [11.507, 11.52, 11.5125, 11.4425, 11.245, 10.355, 9.785, 9.4675, 
              9.33, 9.25, 9.24, 9.33, 9.345, 9.335]
    })

# QuantLib's Helper object for TIIE Crv Bootstrapping
def qlHelper_TIIE(market_data: pd.DataFrame = mkt_TIIE) -> list:
    """
    Create object to bootstrap discount curve from TIIE market.

    Args:
    - market_data (pd.DataFrame): Dataframe with the market rates data. It 
    should come at least with 4 columns: 
        Type = Instrument type
        Tenor = Tenor unit.
        Period = Amount of tenor units.
        Quote = Market quote.
        
    Returns:
        (list) List with the rate helpers to bootstrap TIIE curve.
    """
    # calendar
    calendar_mx = ql.Mexico(0)
    # Market data
    mkt_depo = market_data[['Period','Tenor','Quote']][market_data['Type'] == 'DEPO']
    mkt_swap = market_data[['Period','Tenor','Quote']][market_data['Type'] == 'SWAP']
    
    # Rates from Deposit markets
    deposits = {(int(mkt_depo['Period'][0]), int(mkt_depo['Tenor'][0])): mkt_depo['Quote'][0]/100}
    # Swap rates
    n = mkt_swap.shape[0]
    swaps = {}
    for i in range(1,n):
        swaps[(int(mkt_swap.iloc[i]['Period']), 
               int(mkt_swap.iloc[i]['Tenor']))] = mkt_swap.iloc[i]['Quote']/100
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

    swapHelpers = [ql.SwapRateHelper(
        ql.QuoteHandle(swaps[(n,unit)]),
        ql.Period(n*4, ql.Weeks), 
        calendar_mx,
        fixedLegFrequency, 
        fixedLegAdjustment,
        fixedLegDayCounter, 
        ibor_MXNTIIE)
        for n, unit in swaps.keys()
    ]

    # helpers merge
    helpers = depositHelpers + swapHelpers
    
    return(helpers)

# QL Docs: https://quantlib-python-docs.readthedocs.io/en/latest/index.html
"""
Lets create a TIIE swap
"""
# Set valuation date
ql_val_date = ql.Date(7,11,2023)
ql.Settings.instance().evaluationDate = ql_val_date

# Get TIIE Rates Helpers
helpers_TIIE = qlHelper_TIIE() 
## Bootstrap the USDOIS Market Curve
bootCrv_TIIE = ql.PiecewiseNaturalLogCubicDiscount(0, ql.Mexico(0), 
                                                     helpers_TIIE, 
                                                     ql.Actual360())
## Set the bootstrapped curve ready to be used
crv_TIIE = ql.RelinkableYieldTermStructureHandle()
crv_TIIE.linkTo(bootCrv_TIIE)


# Create TIIE swap schedules
ql_cal = ql.Mexico(0)
# ql_startDate, ql_mtyDate = ql.Date(8,11,2023), ql.Date(26,10,2033)
ql_startDate = ql.Date(8,11,2023) + ql.Period(13*4, ql.Weeks)
ql_mtyDate = ql_startDate + ql.Period(13*4, ql.Weeks)
fxd_leg_tenor = ql.Period(ql.EveryFourthWeek)
flt_leg_tenor = ql.Period(ql.EveryFourthWeek)
non_workday_adj = ql.Following
mtyDate_adj = ql.Following
dates_gen_rule = ql.DateGeneration.Backward
eom_adj = False
fxd_schdl = ql.Schedule(ql_startDate, 
                        ql_mtyDate, 
                        fxd_leg_tenor, 
                        ql_cal, 
                        non_workday_adj, 
                        mtyDate_adj,
                        dates_gen_rule, 
                        eom_adj)
flt_schdl = ql.Schedule(ql_startDate, 
                        ql_mtyDate, 
                        flt_leg_tenor, 
                        ql_cal, 
                        non_workday_adj, 
                        mtyDate_adj,
                        dates_gen_rule, 
                        eom_adj)

# Create TIIE swap floating rate index
ibor_tiie = ql.IborIndex('TIIE', # name
                         ql.Period(ql.EveryFourthWeek), # tenor
                         1, # settlement days
                         ql.MXNCurrency(), # currency
                         ql.Mexico(), # fixing calendar
                         non_workday_adj, # convention
                         False, # endOfMonth
                         ql.Actual360(), # dayCounter
                         crv_TIIE) # handle YieldTermStructure

# Create TIIE IRS object
notional = 2e9
fxd_rate = 0.0906
fxd_leg_daycount = ql.Actual360()
flt_spread = 0
flt_leg_daycount = ql.Actual360()
swap_position = ql.VanillaSwap.Payer
## TIIE IRS
ql_swap_tiie = ql.VanillaSwap(swap_position, 
                              notional, 
                              fxd_schdl, 
                              fxd_rate, 
                              fxd_leg_daycount, 
                              flt_schdl, 
                              ibor_tiie, 
                              flt_spread, 
                              flt_leg_daycount)

# Swap fixed leg cash flows
ql_swap_cf_fxd = pd.DataFrame({
    'accStartDate': cf.accrualStartDate().ISO(),
    'accEndDate': cf.accrualEndDate().ISO(),
    'accDays': cf.accrualDays(),
    'T': cf.accrualPeriod(),
    'Notional': cf.nominal(),
    'FxdRate': cf.rate(),
    'FxdPmt': cf.amount()
    } for cf in map(ql.as_coupon, ql_swap_tiie.leg(0)))

# Swap floating leg cash flows
ql_swap_cf_flt = pd.DataFrame({
    'accStartDate': cf.accrualStartDate().ISO(),
    'accEndDate': cf.accrualEndDate().ISO(),
    'accDays': cf.accrualDays(),
    'T': cf.accrualPeriod(),
    'Notional': cf.nominal(),
    'FltRate': cf.rate(),
    'FltPmt': cf.amount()
    } for cf in map(ql.as_coupon, ql_swap_tiie.leg(1)))


# Swap cash flow details
df_swap_des = ql_swap_cf_fxd.merge(ql_swap_cf_flt[['accEndDate','FltRate','FltPmt']], 
                                   how='outer',
                                   left_on='accEndDate', right_on='accEndDate', 
                                   suffixes=('_Fxd', '_Flt'))
print(df_swap_des)

# Swap Net CF
df_swap_des['NetCF'] = df_swap_des['FxdPmt'] - df_swap_des['FltPmt']

# Swap discounting factors
tmp_DF = []
for i,r in df_swap_des.iterrows():
    tmp_qlDt = ql.DateParser.parseFormatted(r['accEndDate'], '%Y-%m-%d')
    df = crv_USDOIS.discount(tmp_qlDt)
    tmp_DF.append(df)
df_swap_des['DF'] = tmp_DF
print(df_swap_des)

"""
Lets price our TIIE swap
"""
# We need to set the pricing engine as discounting with the USDOIS curve
swap_pricing_eng = ql.DiscountingSwapEngine(crv_TIIE)

# Swap pricing...
ql_swap_tiie.setPricingEngine(swap_pricing_eng)

print(f'\nTIIE 1Y1Y Payer Swap at {fxd_rate:.3%}')
print(f'\tNPV: {ql_swap_tiie.NPV():,.2f}')

# Swap fair rate (Swap rate)
print(f'\tSwap Rate: {ql_swap_tiie.fairRate():.3%}')
