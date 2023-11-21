#%% MODULES
import numpy as np
import pandas as pd
import QuantLib as ql
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
import yfinance as yf

#%% FX CARRY

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
Lets compute the basis risk from a 3M FX Fwd hedge
"""
# Fwd price for USDMXN maturing in 3M
spot = 17.20
USDMXN_F3M = fwd_price_FX(spot_price=spot, T=3/12)

# Fwd price 5 days before maturity or closed out 5D before mty
spot_sc1, spot_sc2 = 17.51, 16.98
USDMXN_F5D_sc1 = fwd_price_FX(spot_price=spot_sc1, T=5/360)
USDMXN_F5D_sc2 = fwd_price_FX(spot_price=spot_sc2, T=5/360)

# Results table
df_fxHedge = pd.DataFrame({
    'Spot': [spot_sc1, spot_sc2],
    'Hedge Price': USDMXN_F3M,
    'Fwd PnL': USDMXN_F3M-np.array([USDMXN_F5D_sc1, USDMXN_F5D_sc2])
    }, index = [1,2])
df_fxHedge['Real Hedge Price'] = df_fxHedge['Spot'] + df_fxHedge['Fwd PnL']
df_fxHedge['Basis Risk'] = df_fxHedge['Real Hedge Price'] - df_fxHedge['Hedge Price']

# Basis risk analysis
tmp_t = np.linspace(90,5,15)/360
tmp_S = np.array([17.2, 17.27094682, 17.37443244, 17.53056546, 17.89933073,
       18.04014132, 18.0196103 , 17.93230431, 18.08575092, 18.14446586,
       18.02227577, 18.29089546, 18.31472552, 18.67761283, 18.74422863])
tmp_S_d = np.concatenate(([17.2],17.2-np.diff(tmp_S)/2))
tmp_df = pd.DataFrame({
    'Spot':tmp_S,
    'Fwd':fwd_price_FX(spot_price=tmp_S, T=tmp_t)
    })
tmp_df_d = pd.DataFrame({
    'Spot':tmp_S_d,
    'Fwd':fwd_price_FX(spot_price=tmp_S_d, T=tmp_t)
    })
tmp_df.plot()
tmp_df_d.plot()

#%% BASIS RISK, HEDGE RATIO, CROSS HEDGE
from pandas_datareader import data as pdr
yf.pdr_override()

"""
Lets compute a cross hedge for jet fuel with heating oil futures
"""
# Date points
t_month = np.arange(1,16,1)

# Change in jet fuel price per gallon
fuel_chg = np.array([
    0.029, 0.02, -0.044, 0.008, 0.026, -0.019, 0.010, -0.007, 0.043,
    0.011, -0.036, -0.018, 0.009, -0.032, 0.023])
# Change in jet fuel price per gallon
hoil_chg = np.array([
    0.021, 0.035, -0.046, 0.001, 0.044, -0.029, -0.026, -0.029, 0.048,
    -0.006, -0.036, -0.011, 0.019, -0.027, 0.029])

# Dataframe resuming
df_fuel_hoil_chg = pd.DataFrame({'Chg F': hoil_chg,
                                 'Chg S': fuel_chg}, index = t_month)

# Price changes std
sigma_F, sigma_S = df_fuel_hoil_chg.std()

# Price changes corr
rho = df_fuel_hoil_chg.corr('spearman').iloc[0,1]

# Hedge ratio
hstar = rho*sigma_S/sigma_F

# Optimal Contract Number
Q_S = 2e6 # we need to purchase 2 million gallons of jet fuel in a month
Q_F = 42000 # contract size for heating oil future
Nstar = np.floor(hstar*Q_S/Q_F)

# Hedge dynamics
df_fuel_hoil_chg.cumsum().plot()


""" 
Lets compute the hedge for a portfolio of stocks
""" 

# We choose a 1-stock pfolio comprised of MSFT
data = pdr.get_data_yahoo(["ES=F","MSFT"], 
                          start="2019-11-20", 
                          end="2023-11-20")
data = data.fillna(method='ffill')
train_data, test_data = data['2019':'2023-06'], data['2023-06-30':'2023-09']

# Correlation between our pfolio and the hedging instrument
rho = train_data['Adj Close'].diff().corr('spearman').iloc[0,1]

# Pfolio and hedging instrument volatilities
sigma_F = train_data['Adj Close']['ES=F'].apply(np.log).diff().std()
sigma_S = train_data['Adj Close']['MSFT'].apply(np.log).diff().std()

# Hedge ratio for our pfolio
hstar = rho*sigma_S/sigma_F

# Optimal Contract Number
Q_S = 2e3*train_data['Adj Close']['MSFT'].iloc[-1]
Q_F = 50*train_data['Adj Close']['ES=F'].iloc[-1]
Nstar = np.floor(hstar*Q_S/Q_F)

# Pfolio and Hedge value
test_pfolio = test_data['Adj Close']['MSFT']*2e3
test_hedge = test_data['Adj Close']['ES=F']*50*Nstar
df_test = pd.concat([test_hedge, test_pfolio], axis=1)
df_test.plot()

# Pfolio changes with hedge
df_pfolio_hedge = df_test.diff().fillna(0)['MSFT'] - df_test.diff().fillna(0)['ES=F']
pd.DataFrame({
    'No Hedge': df_test.diff().fillna(0)['MSFT'].cumsum(), 
    'Hedge': df_pfolio_hedge.cumsum()
           }).plot()


#%% FIXED INCOME HEDGING

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

# Function to get cashflows from Ql Swap
def get_cashflows(ql_swap_tiie: ql.VanillaSwap) -> pd.DataFrame:
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

    # Swap Net CF
    df_swap_des['NetCF'] = df_swap_des['FxdPmt'] - df_swap_des['FltPmt']
    
    return df_swap_des

"""
Lets compute the carry (roll) for 1y1y TIIE swap
"""
# Set valuation date
ql_val_date = ql.Date(7,11,2023)
ql.Settings.instance().evaluationDate = ql_val_date

# Get TIIE Rates Helpers
helpers_TIIE = qlHelper_TIIE() 
## Bootstrap the Market Curve
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
ql_val_date = ql.Date(7,11,2023) + ql.Period(4,ql.Weeks)
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

# Set pricing date
ql_val_date = ql.Date(7,11,2023)
ql.Settings.instance().evaluationDate = ql_val_date

# We set a 2s3s steepener trade
ql_schdl_26m = ql.Schedule(ql.Date(8,11,2023), 
                           ql.Date(8,11,2023) + ql.Period(4*26,ql.Weeks), 
                           fxd_leg_tenor, ql_cal, non_workday_adj, mtyDate_adj,
                           dates_gen_rule, eom_adj)
ql_schdl_39m = ql.Schedule(ql.Date(8,11,2023), 
                           ql.Date(8,11,2023) + ql.Period(4*39,ql.Weeks), 
                           fxd_leg_tenor, ql_cal, non_workday_adj, mtyDate_adj,
                           dates_gen_rule, eom_adj)
## tiie index
ibor_tiie = ql.IborIndex('TIIE', ql.Period(ql.EveryFourthWeek), 1, 
                         ql.MXNCurrency(), ql.Mexico(), non_workday_adj, False,
                         ql.Actual360(), crv_TIIE)

## 2s3s steepnr at -56bp
ql_26m_rec = ql.VanillaSwap(-1, 968.004e6, ql_schdl_26m, 0.1035, fxd_leg_daycount, 
                            ql_schdl_26m, ibor_tiie, flt_spread, flt_leg_daycount)
ql_39m_pay = ql.VanillaSwap(1, 675.04e6, ql_schdl_39m, 0.0979, fxd_leg_daycount, 
                            ql_schdl_39m, ibor_tiie, flt_spread, flt_leg_daycount)
swap_pricing_eng = ql.DiscountingSwapEngine(crv_TIIE)
ql_26m_rec.setPricingEngine(swap_pricing_eng)
ql_39m_pay.setPricingEngine(swap_pricing_eng)

## Check dv01 neutral
ql_26m_rec.legBPS(0) + ql_39m_pay.legBPS(0)

# Strategy Portfolio
pfolio_strat = [ql_26m_rec, ql_39m_pay]

# Pfolio Starting NPV
pfolio_strat_npv0 = np.sum(np.array([x.NPV() for x in pfolio_strat]))
print(f'Strategy NPV as of {ql_val_date}:\n\t2y: {ql_26m_rec.NPV():,.0f}\n\t3y: {ql_39m_pay.NPV():,.0f}')
print(f'\t2s3s: {pfolio_strat_npv0:,.0f}')

# Pfolio Expected CF
ql_26m_cf, ql_39m_cf = get_cashflows(ql_26m_rec), get_cashflows(ql_39m_pay)
pfolio_CF = ql_39m_cf[['accStartDate', 'accEndDate']].copy()
pfolio_CF['NetCF'] = ql_26m_cf['NetCF']+ql_39m_cf['NetCF']*-1
pfolio_CF['NetCF'].iloc[26:] = ql_39m_cf['NetCF'].iloc[26:]*-1
print(pfolio_CF.iloc[:3])

# Lets plot net cash flows
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(pfolio_CF['accEndDate'].apply(pd.to_datetime), pfolio_CF['NetCF'], width = 10)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(0,3,6,9,12)))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
plt.title('TIIE 2s3s Net Payments',size=16)
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.tight_layout(); plt.show()

# Lets assume that the yield curve is going to steepen
mkt_TIIE_delta = np.repeat(np.array([600,585,564,525,380,300,200,115,80,75,50]), 
                           np.array([2,1,1,1,1,1,1,1,1,1,3]))/600
mkt_TIIE_shift = mkt_TIIE.copy()
mkt_TIIE_shift['Quote'] -= mkt_TIIE_delta

# Lets plot YC shift
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(mkt_TIIE['Period'], mkt_TIIE['Quote'], 
        color = 'b', marker='o', label=f'{ql.Date(7,11,2023)}')
ax.plot(mkt_TIIE['Period'], mkt_TIIE_shift['Quote'], 
        color = 'orange', marker='o', label=f'{ql.Date(7,11,2023) + ql.Period(4,ql.Weeks)}')
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

# Pfolio after market steepnd 
ibor_tiie_shift = ql.IborIndex('TIIE', ql.Period(ql.EveryFourthWeek), 1,
                         ql.MXNCurrency(), ql.Mexico(), non_workday_adj, False,
                         ql.Actual360(), crv_TIIE_shift)
ibor_tiie_shift.addFixings([ql.Date(7,11,2023), ql.Date(5,12,2023)], 
                          [crv_TIIE_shift.forwardRate(ql.Date(8,11,2023), 
                                                     ql.Date(6,12,2023), 
                                                     fxd_leg_daycount, 
                                                     ql.Simple).rate(),
                           crv_TIIE_shift.forwardRate(ql.Date(6,12,2023), 
                                                      ql.Date(3,1,2024), 
                                                      fxd_leg_daycount, 
                                                      ql.Simple).rate()])
ql_26m_rec = ql.VanillaSwap(-1, 968.004e6, ql_schdl_26m, 0.1035, fxd_leg_daycount, 
                            ql_schdl_26m, ibor_tiie_shift, flt_spread, flt_leg_daycount)
ql_39m_pay = ql.VanillaSwap(1, 675.04e6, ql_schdl_39m, 0.0979, fxd_leg_daycount, 
                            ql_schdl_39m, ibor_tiie_shift, flt_spread, flt_leg_daycount)
swap_pricing_eng = ql.DiscountingSwapEngine(crv_TIIE_shift)
ql_26m_rec.setPricingEngine(swap_pricing_eng)
ql_39m_pay.setPricingEngine(swap_pricing_eng)
pfolio_strat = [ql_26m_rec, ql_39m_pay]

# If the steepenning move is instantaneous
pfolio_strat_npv1 = np.sum(np.array([x.NPV() for x in pfolio_strat]))
print('Market Steepened Instantaneously')
print(f'Strategy NPV as of {ql_val_date}:\n\t2y: {ql_26m_rec.NPV():,.0f}\n\t3y: {ql_39m_pay.NPV():,.0f}')
print(f'\t2s3s: {pfolio_strat_npv1:,.0f}')

# If the steepenning move is after 28D
ql.Settings.instance().evaluationDate = ql.Date(7,11,2023) + ql.Period(4*1,ql.Weeks)
pfolio_strat_npv1 = np.sum(np.array([x.NPV() for x in pfolio_strat]))
print('Market Steepened 1L Later')
print(f'Strategy NPV as of {ql.Settings.instance().evaluationDate}:\n\t2y: {ql_26m_rec.NPV():,.0f}\n\t3y: {ql_39m_pay.NPV():,.0f}')
print(f'\t2s3s: {pfolio_strat_npv1:,.0f}')

# Pfolio Expected CF
ql_26m_cf, ql_39m_cf = get_cashflows(ql_26m_rec), get_cashflows(ql_39m_pay)
pfolio_CF = ql_39m_cf[['accStartDate', 'accEndDate']].copy()
pfolio_CF['NetCF'] = ql_26m_cf['NetCF']+ql_39m_cf['NetCF']*-1
pfolio_CF['NetCF'].iloc[26:] = ql_39m_cf['NetCF'].iloc[26:]*-1
print(pfolio_CF.iloc[:3])

# Lets plot net cash flows
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(pfolio_CF['accEndDate'].apply(pd.to_datetime), pfolio_CF['NetCF'], width = 10)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(0,3,6,9,12)))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
plt.title('TIIE 2s3s Net Payments',size=16)
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.tight_layout(); plt.show()

# 2s3s spread t0, t1
mkt_TIIE['Quote'].iloc[5:7].diff().dropna().values*100, mkt_TIIE_shift['Quote'].iloc[5:7].diff().dropna().values*100


