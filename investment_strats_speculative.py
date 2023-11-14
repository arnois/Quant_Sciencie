#%% MODULES
import numpy as np
import pandas as pd
import QuantLib as ql
from matplotlib import pyplot as plt
from scipy.optimize import minimize

import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter

#%% FX CARRY

# Function to compute fwd points for FX futures contracts
def fwd_points(spot_price: float = 17.50, T: float = 3/12, 
               r_d: float = 0.1120, r_f: float = 0.052) -> float:
    """
    Calculates the forward points for currency forwards.

    Args:
    - spot_price (float): Current FX rate or spot price. QUOTE/BASE
    - T: Forward period or maturity in years.
    - r_d: Domestic or base currency rate.
    - r_f: Foreign or quote currency rate.
    NOTE: FX rate or spot price as QUOTE/BASE

    Returns:
        (float) Forward points.
    """
    # Interest return factor in the base currency
    domestic_ret = (1+r_d*T)
    
    # Interest return factor in the quote currency
    foreign_ret = (1+r_f*T)
    
    return (domestic_ret/foreign_ret - 1)*spot_price*1e4

# Function to compute fx forward price
def fwd_price_FX(spot_price: float = 17.50, T: float = 3/12, 
                 r_d: float = 0.1120, r_f: float = 0.052) -> float:
    """
    Calculates the price for an FX Forward contract.

    Args:
    - spot_price (float): Current FX rate or spot price. QUOTE/BASE
    - T: Forward period or maturity in years.
    - r_d: Domestic or base currency rate.
    - r_f: Foreign or quote currency rate.

    Returns:
        (float) Forward price.
    """
    # Fwd factor
    fwd_f = (1+r_d*T)/(1+r_f*T)
    
    return spot_price*fwd_f

"""
Lets compute the 1M carry for a 3M-Period USDMXN fwd contract
"""
# Fwd price for USDMXN maturing in 3M
USDMXN_F3M = fwd_price_FX(spot_price=17.50, T=3/12)
# Fwd price after 1M with min changes in spot price 
USDMXN_F2M = fwd_price_FX(spot_price=17.50, T=2/12)
# Carry for 1M
USDMXN_carry_1M = USDMXN_F2M - USDMXN_F3M
print(f'\n1M-Carry in USDMXN Fwd: {USDMXN_carry_1M:,.4f}')


#%% FIXED INCOME CARRY

# Function to compute zero-coupon bond price
def zcb_price(nominal: float = 10.0, r: float = 0.1120, 
              dtm: int = 182, ybc: int = 360) -> float:
    
    """
    Calculate the price of a Zero-Coupon Bond (ZCB).

    Args:
    - nominal: Bond's face value
    - r (float): Yield rate (as a decimal, for example, 0.05 for 5%).
    - dtm (int): Days to maturity.
    - ybc (int): Convention for year base. For example: 360, 365, 252.

    Returns:
    float: ZCB price rounded to 6 decimal places.
    """
    # Discount factor
    DF = 1/(1+r*dtm/ybc)
    
    # Face value present value
    P = nominal*DF
    
    return round(P, 6)

"""
Lets compute the 1M-carry of a long position in a ZCB with 6M maturity
"""
# ZCB specs
zcb_fv, zcb_y, zcb_dtm = 10, 11.25/100, 180
# ZCB price
zcb_px_6m = zcb_price(nominal=zcb_fv, r=zcb_y, dtm=zcb_dtm)
# ZCB price 30 days after
zcb_px_5m = zcb_price(nominal=zcb_fv, r=zcb_y, dtm=zcb_dtm-30)
# ZCB 30 days carry
zcb_6m_carry_30D = zcb_px_5m - zcb_px_6m
print(f'\nZCB 1M Carry: ${zcb_6m_carry_30D:,.4f}')

"""
Lets now compute carry from funding the ZCB position for 1M.
"""
# Daily funding costs
zcb_fund_rate = 0.1120

# Borrowed amount
zcb_fund_cost_1M = zcb_px_6m*zcb_fund_rate*30/360

# 1M Carry for repo'ng a 6M-ZCB
print(f'\n6M-ZCB Repo Carry for 1M: ${zcb_6m_carry_30D-zcb_fund_cost_1M:,.4f}')

# Function to compute carry from funding a ZCB
def zcb_carry(nominal: float = 10.0, r: float = 0.1120, dtm: int = 182, 
              ybc: int = 360, carry_days: int = 30, funding_r: float = 0.11) -> float:
    """
    Calculate the carry for funding a ZCB.

    Args:
    - nominal: Bond's face value
    - r (float): Yield rate (as a decimal, for example, 0.05 for 5%).
    - dtm (int): Days to maturity.
    - ybc (int): Convention for year base. For example: 360, 365, 252.
    - carry_days (int): Days to hold the ZCB.
    - funding_r (float): Funding rate.

    Returns:
    float: ZCB carry rounded to 6 decimal places.
    """
    # ZCB price
    zcb_px0 = zcb_price(nominal,r,dtm,ybc)
    
    # ZCB price after holding period
    zcb_px1 = zcb_price(nominal,r,dtm-carry_days,ybc)
    
    # ZCB gain from holding period
    zcb_gain = zcb_px1 - zcb_px0
    
    # ZCB cost from funding the position during the holding period
    zcb_cost = zcb_px0*funding_r*carry_days/ybc
    
    # Carry
    carry = zcb_gain - zcb_cost
    
    return round(carry, 6)

# Recreate previous example with our function
print('\n 1M Carry from funding a 6M-ZCB: '+\
      f'${zcb_carry(zcb_fv, zcb_y, zcb_dtm, 360, 30, zcb_fund_rate):,.4f}')
"""
Lets find our breakeven funding rate
"""
print(f'\nCarry with {0.1120:.4%} funding rate: ${zcb_carry(zcb_fv, zcb_y, zcb_dtm, 360, 30, 0.1120):.8f}')
print(f'Carry with {0.1020:.4%} funding rate: ${zcb_carry(zcb_fv, zcb_y, zcb_dtm, 360, 30, 0.1020):.8f}')
print(f'Carry with {0.1060:.4%} funding rate: ${zcb_carry(zcb_fv, zcb_y, zcb_dtm, 360, 30, 0.1060):.8f}')
print(f'Carry with {0.1080:.4%} funding rate: ${zcb_carry(zcb_fv, zcb_y, zcb_dtm, 360, 30, 0.1080):.8f}')
print(f'Carry with {0.1070:.4%} funding rate: ${zcb_carry(zcb_fv, zcb_y, zcb_dtm, 360, 30, 0.1070):.8f}')
print(f'Carry with {0.1075:.4%} funding rate: ${zcb_carry(zcb_fv, zcb_y, zcb_dtm, 360, 30, 0.1075):.8f}')
print(f'Carry with {0.107462:.4%} funding rate: ${zcb_carry(zcb_fv, zcb_y, zcb_dtm, 360, 30, 0.107462):.8f}')

print('\n 1M Carry from funding a 6M-ZCB: '+\
      f'${zcb_carry(zcb_fv, zcb_y, zcb_dtm, 360, 30, 0.107):,.4f}')

# Solving for breakeven funding rate
max_funding_rate = (zcb_px_5m/zcb_px_6m-1)*360/30
print(f'Funding Rate BE for 1M of previous 6M-ZCB: {max_funding_rate:,.6%}')


#%% FIXED INCOME YC ROLL STRATS

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

"""
Lets compute the carry (roll) for 1y1y TIIE swap
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
fxd_rate = 0.09367826861518108
fxd_leg_daycount = ql.Actual360()
flt_spread = 0
flt_leg_daycount = ql.Actual360()
swap_position = ql.VanillaSwap.Receiver
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
# Set TIIE pricing engine
swap_pricing_eng = ql.DiscountingSwapEngine(crv_TIIE)
ql_swap_tiie.setPricingEngine(swap_pricing_eng)

# 1Y1Y TIIE prices
T1y1y_p0 = ql_swap_tiie.fairRate()
T1y1y_npv0 =  ql_swap_tiie.NPV()

# 12m1Y TIIE Swap Rate
ql_val_date = ql.Date(8,11,2023) + ql.Period(4,ql.Weeks)
ql.Settings.instance().evaluationDate = ql_val_date
T1y1y_p1 = ql_swap_tiie.fairRate()
T1y1y_npv1 = ql_swap_tiie.NPV()

print(f'\n1y1y TIIE Swap Rate: {T1y1y_p0:.4%}')
print(f'12m1Y TIIE Swap Rate: {T1y1y_p1:.4%}')
print(f'1y1y TIIE 28D Carry(roll): {1e4*(T1y1y_p0-T1y1y_p1):.1f} bp')

print(f'\n1y1y TIIE NPV: {T1y1y_npv0:,.0f}')
print(f'12m1Y TIIE NPV: {T1y1y_npv1:,.0f}')
print(f'1y1y TIIE 28D NPV Carry(roll): ${(T1y1y_npv1-T1y1y_npv0):,.0f}')

#%% FIXED INCOME CURVE STRATS

# Lets assume that the yield curve is going to flatten
mkt_TIIE_delta = np.repeat(np.array([600,585,564,525,380,300,200,115,80,75,50]), 
                           np.array([2,1,1,1,1,1,1,1,1,1,3]))/100
mkt_TIIE_shift = mkt_TIIE.copy()
mkt_TIIE_shift['Quote'] -= mkt_TIIE_delta
# Lets plot YC shift
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(mkt_TIIE['Period'], mkt_TIIE['Quote'], 
        color = 'b', marker='o', label=f'{ql.Date(8,11,2023)}')
ax.plot(mkt_TIIE['Period'], mkt_TIIE_shift['Quote'], 
        color = 'orange', marker='o', label=f'{ql.Date(8,11,2023) + ql.Period(4,ql.Weeks)}')
plt.title('TIIE Swaps Market', size=17)
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.legend()
plt.tight_layout(); plt.show()

# Set the shifted curve for pricing
crv_TIIE_shift = ql.RelinkableYieldTermStructureHandle()
crv_TIIE_shift.linkTo(
    ql.PiecewiseNaturalLogCubicDiscount(0, ql.Mexico(0), 
                                        qlHelper_TIIE(market_data=mkt_TIIE_shift) , 
                                        ql.Actual360()))

# Set TIIE pricing engine
ibor_tiie = ql.IborIndex('TIIE', 
                         ql.Period(ql.EveryFourthWeek),
                         1,
                         ql.MXNCurrency(), 
                         ql.Mexico(), 
                         non_workday_adj,
                         False,
                         ql.Actual360(), 
                         crv_TIIE_shift)
ql_swap_tiie = ql.VanillaSwap(swap_position, 
                              notional, 
                              fxd_schdl, 
                              fxd_rate, 
                              fxd_leg_daycount, 
                              flt_schdl, 
                              ibor_tiie, 
                              flt_spread, 
                              flt_leg_daycount)
swap_pricing_eng = ql.DiscountingSwapEngine(crv_TIIE_shift)
ql_swap_tiie.setPricingEngine(swap_pricing_eng)

# 1Y1Y TIIE prices
T1y1y_p_shift = ql_swap_tiie.fairRate()
T1y1y_npv_shift =  ql_swap_tiie.NPV()

print(f'1y1y TIIE Swap Rate After Shift: {T1y1y_p_shift:.4%}')





