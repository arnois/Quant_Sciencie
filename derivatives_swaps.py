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

# Now lets assume we have the following fwd rates as the swap's floating rates
df_SOFR3M = np.array([0.022, 0.026, 0.028, 0.031, 0.033, 0.034, 0.036, 0.038])
swap_info['FloatRate'] = df_SOFR3M

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
    
    # Settlement date
    dt_settlement = ql.UnitedStates(0).advance(
            ql.Settings.instance().evaluationDate, ql.Period('2D'))
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
bootCrv_USDOIS = ql.PiecewiseLogLinearDiscount(0,ql.UnitedStates(0), 
                                               helpers_USDOIS, ql.Actual360())
## Set the bootstrapped curve ready to be used
crv_USDOIS = ql.RelinkableYieldTermStructureHandle()
crv_USDOIS.linkTo(bootCrv_USDOIS)

# Second, we need our forward rates curve from the USDSOFR swap's market
## Get USDSOFR Rates Helpers
helpers_USDSOFR = qlHelper_SOFR() # mkt_SOFR_2 = mkt_SOFR.copy()
## Bootstrap the USDSOFR Market Curve with OIS discounting
bootCrv_USDSOFR = ql.PiecewiseLogLinearDiscount(0,ql.UnitedStates(0), 
                                               helpers_USDSOFR, ql.Actual360())
## Set the bootstrapped curve ready to be used
crv_USDSOFR = ql.RelinkableYieldTermStructureHandle()
crv_USDSOFR.linkTo(bootCrv_USDSOFR)


# Create swap schedules
ql_cal = ql.UnitedStates(0)
ql_startDate = ql.Date(8,3,2022)
ql_mtyDate = ql.Date(8,3,2028)
fxd_leg_tenor = ql.Period(1, ql.Years)
flt_leg_tenor = ql.Period(1, ql.Years)
non_workday_adj = ql.ModifiedFollowing
mtyDate_adj = ql.ModifiedFollowing
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

# Create swap floating rate index
swap_index = ql.Sofr(crv_USDSOFR)

# Create IRS object
notional = 100e6
fxd_rate = 0.0295
fxd_leg_daycount = ql.Actual360()
flt_spread = 0
flt_leg_daycount = ql.Actual360()
swap_position = ql.VanillaSwap.Receiver
## USDSOFR IRS
ql_swap = ql.VanillaSwap(swap_position, notional, 
                         fxd_schdl, fxd_rate, fxd_leg_daycount, 
                         flt_schdl, swap_index, flt_spread, flt_leg_daycount)

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
print(f'\tSwap Rate: {ql_swap.fairRate():.3%}')






# Swap
