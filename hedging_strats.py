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

train_data['2023-01-01':'2023-06']['Adj Close']['MSFT'].plot()

# Correlation between our pfolio and the hedging instrument
rho = train_data['Adj Close'].apply(np.log).diff().corr().iloc[0,1]

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
(100*df_test/df_test.iloc[0]).plot(title='Pfolio & Future Performance')

# Pfolio changes with hedge
df_pfolio_hedge = df_test.diff().fillna(0)['MSFT'] - df_test.diff().fillna(0)['ES=F']
pd.DataFrame({
    'No Hedge': df_test.diff().fillna(0)['MSFT'].cumsum(), 
    'Hedge': df_pfolio_hedge.cumsum()
           }).plot(title='Equity Curve')

# PROJECT IDEA: 
## Improve hedge dynamics with CAPM; change hedge ratio for market beta


"""
Lets compute the cross hedge for a portfolio of swaps
"""
# Data
fidata = pd.DataFrame({
    'TIIE2Y': [10.6900,10.7650,10.5900,10.5850,10.5350,10.5300,10.4350,10.4850,
               10.5850,10.5900,10.6350,10.7050,10.7400,10.7350,10.6700,10.6700,
               10.6900,10.7700,10.6450,10.6050,10.6550],
    'US2Y': [5.1041, 5.1502, 5.0518, 5.0183, 5.0813, 5.0813, 4.9696, 4.9821, 
             5.0687, 5.0538, 5.0985, 5.2095, 5.2225, 5.1585, 5.0731, 5.0474, 
             5.1119, 5.1206, 5.0395, 5.0021, 5.0540],
    'TU1': [101.222656, 101.152344, 101.343750, 101.390625, 101.277344, 
            101.542969, 101.468750, 101.421875, 101.300781, 101.332031,
            101.246094, 101.023438, 101.015625, 101.105469, 101.265625,
            101.312500, 101.226563, 101.140625, 101.285156, 101.332031,
            101.277344]
    }, index = np.arange(1,22,1))

# Data for hedging parameters
fidata_train, fidata_test = fidata.iloc[:16], fidata.iloc[15:]

# Plotting what we know and one expected path
fidata_train.diff()[['TIIE2Y', 'US2Y']].plot()
fidata_test.diff()[['TIIE2Y', 'US2Y']].plot()

# Data changes
sigma_TIIE, sigma_TU1 = fidata_train.diff().std()[['TIIE2Y', 'TU1']]
# Correlation between tiie and hedge
rho_fi = fidata_train.diff().corr()['TIIE2Y']['TU1']

# Hedge ratio
hstar = rho_fi*100*sigma_TIIE/sigma_TU1

# Optimal TU Contract Size
Q_TIIE = 5000
Q_TU1 = 32*62.5
Nstar = np.floor(hstar*Q_TIIE/Q_TU1)

# Hedge dynamics
fidata_test_dynamics = pd.concat([
    fidata_test.diff()[['TIIE2Y']].fillna(0)*-5000*100,
    fidata_test.diff()[['TU1']].fillna(0)*2000*Nstar], axis=1)
fidata_test_dynamics['Hedge'] = fidata_test_dynamics['TIIE2Y'] + fidata_test_dynamics['TU1']
fidata_test_dynamics[['TIIE2Y', 'Hedge']].plot(title='Equity Curve')

# Alternative hedge ratio with DV01
dv01_TU1 = -0.69*62.5
Nstar = np.floor(Q_TIIE/dv01_TU1)
# Hedge dynamics
fidata_test_dynamics = pd.concat([
    fidata_test.diff()[['TIIE2Y']].fillna(0)*-5000*100,
    fidata_test.diff()[['TU1']].fillna(0)*2000*Nstar], axis=1)
fidata_test_dynamics['Hedge'] = fidata_test_dynamics['TIIE2Y'] + fidata_test_dynamics['TU1']
fidata_test_dynamics[['TIIE2Y', 'Hedge']].plot(title='Equity Curve')



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
Lets compute bucket risk for a 1y1y TIIE swap
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
schdl_1y1y = ql.Schedule(ql_startDate, 
                        ql_mtyDate, 
                        fxd_leg_tenor, 
                        ql_cal, 
                        non_workday_adj, 
                        mtyDate_adj,
                        dates_gen_rule, 
                        eom_adj)

# Create TIIE swap floating rate index
ibor_tiie = ql.IborIndex('TIIE', 
                         ql.Period(ql.EveryFourthWeek), 
                         1, 
                         ql.MXNCurrency(), 
                         ql.Mexico(), 
                         non_workday_adj, 
                         False,
                         ql.Actual360(), 
                         crv_TIIE) 

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
                              schdl_1y1y, 
                              fxd_rate, 
                              fxd_leg_daycount, 
                              schdl_1y1y, 
                              ibor_tiie, 
                              flt_spread, 
                              flt_leg_daycount)
# Set TIIE pricing engine
swap_pricing_eng = ql.DiscountingSwapEngine(crv_TIIE)
ql_swap_tiie.setPricingEngine(swap_pricing_eng)

# 1Y1Y TIIE
T1y1y_p0 = ql_swap_tiie.fairRate()
T1y1y_npv0 =  ql_swap_tiie.NPV()
T1y1y_dv01 = ql_swap_tiie.legBPS(0)*ql_swap_tiie.type()
print(f'\n1y1y TIIE \n\tSwap Rate: {T1y1y_p0:.4%}')
print(f'\tNPV: {T1y1y_npv0:,.0f}')
print(f'\tOutright DV01: {T1y1y_dv01:,.0f}')


"""
Lets hedge our pfolio of swaps by outright risk
"""
# Assume we hedge with the 1y tenor
hedge_pos = ql.VanillaSwap.Payer
hedge_fv = 1.75e9
schdl_1y = ql.Schedule(ql_cal.advance(ql_val_date,ql.Period(1,ql.Days)), 
                       ql_cal.advance(ql_val_date,ql.Period(1,ql.Days))+ql.Period(4*13,ql.Weeks), 
                       fxd_leg_tenor, ql_cal, non_workday_adj, mtyDate_adj, dates_gen_rule, eom_adj)
ql_swap_hedge_1y = ql.VanillaSwap(hedge_pos, hedge_fv, schdl_1y, 0.1125, fxd_leg_daycount, schdl_1y, 
                              ibor_tiie, flt_spread, flt_leg_daycount)
swap_pricing_eng = ql.DiscountingSwapEngine(crv_TIIE)
ql_swap_hedge_1y.setPricingEngine(swap_pricing_eng)
hedge_1y_npv0 = ql_swap_hedge_1y.NPV()

# Lets see what happens with parallel shifts
mkt_TIIE_shift = mkt_TIIE.copy()
mkt_TIIE_shift['Quote'] += 20/100

crv_TIIE_shift = ql.RelinkableYieldTermStructureHandle()
crv_TIIE_shift.linkTo(
    ql.PiecewiseNaturalLogCubicDiscount(0, ql.Mexico(0), 
                                        qlHelper_TIIE(market_data=mkt_TIIE_shift) , 
                                        ql.Actual360()))

ibor_tiie_shift = ql.IborIndex('TIIE', ql.Period(ql.EveryFourthWeek), 1,
                         ql.MXNCurrency(), ql.Mexico(), non_workday_adj, False,
                         ql.Actual360(), crv_TIIE_shift)

ql_swap_tiie = ql.VanillaSwap(swap_position, notional, schdl_1y1y, fxd_rate, 
                              fxd_leg_daycount, schdl_1y1y, ibor_tiie_shift, 
                              flt_spread, flt_leg_daycount)
ql_swap_hedge_1y = ql.VanillaSwap(hedge_pos, hedge_fv, schdl_1y, 0.1125, 
                                  fxd_leg_daycount, schdl_1y, ibor_tiie_shift, 
                                  flt_spread, flt_leg_daycount)
swap_pricing_eng = ql.DiscountingSwapEngine(crv_TIIE_shift)
ql_swap_tiie.setPricingEngine(swap_pricing_eng)
ql_swap_hedge_1y.setPricingEngine(swap_pricing_eng)

npv_shift_1y1y = ql_swap_tiie.NPV()
npv_shift_hedge_1y = ql_swap_hedge_1y.NPV()

print('NPV Change Afer Market Sold Off:'+\
      f'\n\tNo Hedge: {npv_shift_1y1y-T1y1y_npv0: ,.0f}'+\
      f'\n\tHedge: {npv_shift_1y1y-T1y1y_npv0 + npv_shift_hedge_1y-hedge_1y_npv0:,.0f}')
    
# Lets see what happens with slope shifts
mkt_TIIE_shift = mkt_TIIE.copy()
mkt_TIIE_shift.loc[:4,'Quote'] += 20/100
mkt_TIIE_shift.loc[5:,'Quote'] += 30/100
rv_TIIE_shift = ql.RelinkableYieldTermStructureHandle()
crv_TIIE_shift.linkTo(
    ql.PiecewiseNaturalLogCubicDiscount(0, ql.Mexico(0), 
                                        qlHelper_TIIE(market_data=mkt_TIIE_shift) , 
                                        ql.Actual360()))

ibor_tiie_shift = ql.IborIndex('TIIE', ql.Period(ql.EveryFourthWeek), 1,
                         ql.MXNCurrency(), ql.Mexico(), non_workday_adj, False,
                         ql.Actual360(), crv_TIIE_shift)

ql_swap_tiie = ql.VanillaSwap(swap_position, notional, schdl_1y1y, fxd_rate, 
                              fxd_leg_daycount, schdl_1y1y, ibor_tiie_shift, 
                              flt_spread, flt_leg_daycount)
ql_swap_hedge_1y = ql.VanillaSwap(hedge_pos, hedge_fv, schdl_1y, 0.1125, 
                                  fxd_leg_daycount, schdl_1y, ibor_tiie_shift, 
                                  flt_spread, flt_leg_daycount)
swap_pricing_eng = ql.DiscountingSwapEngine(crv_TIIE_shift)
ql_swap_tiie.setPricingEngine(swap_pricing_eng)
ql_swap_hedge_1y.setPricingEngine(swap_pricing_eng)

npv_shift_1y1y = ql_swap_tiie.NPV()
npv_shift_hedge_1y = ql_swap_hedge_1y.NPV()

print('NPV Change Afer Market Bear Steepened:'+\
      f'\n\tNo Hedge: {npv_shift_1y1y-T1y1y_npv0: ,.0f}'+\
      f'\n\tHedge: {npv_shift_1y1y-T1y1y_npv0 + npv_shift_hedge_1y-hedge_1y_npv0:,.0f}')
    

"""
Lets hedge our pfolio of swaps by bucket risk
"""

# Risk by Tradeable Tenor
br_T1y1y = pd.DataFrame(columns=['dv01'], index=mkt_TIIE['Period'])
for i,r in mkt_TIIE.iterrows():
    mkt_TIIE_br = mkt_TIIE.copy()
    mkt_TIIE_br.loc[i,'Quote'] = mkt_TIIE_br.loc[i,'Quote']+1/100
    ## Curve with tenor shift
    crv_TIIE_shift = ql.RelinkableYieldTermStructureHandle()
    crv_TIIE_shift.linkTo(
        ql.PiecewiseNaturalLogCubicDiscount(0, ql.Mexico(0), 
                                            qlHelper_TIIE(market_data=mkt_TIIE_br) , 
                                            ql.Actual360()))
    # ibor index shifted
    ibor_tiie_shift = ql.IborIndex('TIIE', ql.Period(ql.EveryFourthWeek), 1,
                             ql.MXNCurrency(), ql.Mexico(), non_workday_adj, False,
                             ql.Actual360(), crv_TIIE_shift)
    # Swap price effect by curve shift
    ql_swap_tiie = ql.VanillaSwap(swap_position, 
                                  notional, 
                                  schdl_1y1y, 
                                  fxd_rate, 
                                  fxd_leg_daycount, 
                                  schdl_1y1y, 
                                  ibor_tiie_shift, 
                                  flt_spread, 
                                  flt_leg_daycount)
    swap_pricing_eng = ql.DiscountingSwapEngine(crv_TIIE_shift)
    ql_swap_tiie.setPricingEngine(swap_pricing_eng)
    swap_npv_chg = ql_swap_tiie.NPV() - T1y1y_npv0
    br_T1y1y.loc[r['Period']] = swap_npv_chg
    
# Lets see the dv01 risk by bucket
br_T1y1y_fmt = br_T1y1y.apply(lambda x: "{:,.0f}".format(x['dv01']), axis=1)
print(br_T1y1y_fmt)

# We need to hedge two buckets
## 1y bucket hedge
ql_swap_hedge_1y = ql.VanillaSwap(-1, 1.88e9, schdl_1y, 0.112425, 
                                  fxd_leg_daycount, schdl_1y, ibor_tiie, 
                                  flt_spread, flt_leg_daycount)
## 2y bucket hedge
schdl_2y = ql.Schedule(ql_cal.advance(ql_val_date,ql.Period(1,ql.Days)), 
                       ql_cal.advance(ql_val_date,ql.Period(1,ql.Days))+ql.Period(4*26,ql.Weeks), 
                       fxd_leg_tenor, ql_cal, non_workday_adj, mtyDate_adj, dates_gen_rule, eom_adj)
ql_swap_hedge_2y = ql.VanillaSwap(1, 1.95e9, schdl_2y, 0.10356, 
                                  fxd_leg_daycount, schdl_2y, ibor_tiie, 
                                  flt_spread, flt_leg_daycount)
swap_pricing_eng = ql.DiscountingSwapEngine(crv_TIIE)
ql_swap_hedge_1y.setPricingEngine(swap_pricing_eng)
ql_swap_hedge_2y.setPricingEngine(swap_pricing_eng)
hedge_1y_npv0 = ql_swap_hedge_1y.NPV()
hedge_2y_npv0 = ql_swap_hedge_2y.NPV()

# Lets see bucket risk after hedge pfolio
br_T1y1y_hedge = pd.DataFrame(columns=['dv01'], index=mkt_TIIE['Period'])
for i,r in mkt_TIIE.iterrows():
    mkt_TIIE_br = mkt_TIIE.copy()
    mkt_TIIE_br.loc[i,'Quote'] = mkt_TIIE_br.loc[i,'Quote']+1/100
    ## Curve with tenor shift
    crv_TIIE_shift = ql.RelinkableYieldTermStructureHandle()
    crv_TIIE_shift.linkTo(
        ql.PiecewiseNaturalLogCubicDiscount(0, ql.Mexico(0), 
                                            qlHelper_TIIE(market_data=mkt_TIIE_br) , 
                                            ql.Actual360()))
    # ibor index shifted
    ibor_tiie_shift = ql.IborIndex('TIIE', ql.Period(ql.EveryFourthWeek), 1,
                             ql.MXNCurrency(), ql.Mexico(), non_workday_adj, False,
                             ql.Actual360(), crv_TIIE_shift)
    # Swap price effect by curve shift
    ql_tmp_tiie = ql.VanillaSwap(swap_position, notional, schdl_1y1y, fxd_rate, 
                                  fxd_leg_daycount, schdl_1y1y, ibor_tiie_shift, 
                                  flt_spread, flt_leg_daycount)
    ql_tmp_tiie1y = ql.VanillaSwap(-1, 1.88e9, schdl_1y, 0.112425, 
                                   fxd_leg_daycount, schdl_1y, ibor_tiie_shift, 
                                   flt_spread, flt_leg_daycount)
    ql_tmp_tiie2y = ql.VanillaSwap(1, 1.95e9, schdl_2y, 0.10356, 
                                   fxd_leg_daycount, schdl_2y, ibor_tiie_shift, 
                                   flt_spread, flt_leg_daycount)
    tmp_px_eng = ql.DiscountingSwapEngine(crv_TIIE_shift)
    ql_tmp_tiie.setPricingEngine(tmp_px_eng)
    ql_tmp_tiie1y.setPricingEngine(tmp_px_eng)
    ql_tmp_tiie2y.setPricingEngine(tmp_px_eng)
    swap_npv_chg = ql_tmp_tiie.NPV() - T1y1y_npv0
    hedge1y_npv_chg = ql_tmp_tiie1y.NPV() - hedge_1y_npv0
    hedge2y_npv_chg = ql_tmp_tiie2y.NPV() - hedge_2y_npv0
    br_T1y1y_hedge.loc[r['Period']] = swap_npv_chg+hedge1y_npv_chg+hedge2y_npv_chg

# Lets see the dv01 risk by bucket after hedge
br_T1y1y_hedge_fmt = br_T1y1y_hedge.apply(lambda x: "{:,.0f}".format(x['dv01']), axis=1)
print(pd.concat([br_T1y1y_fmt, br_T1y1y_hedge_fmt],axis=1))


# Lets see what happens with parallel shifts
mkt_TIIE_shift = mkt_TIIE.copy()
mkt_TIIE_shift['Quote'] += 20/100

crv_TIIE_shift = ql.RelinkableYieldTermStructureHandle()
crv_TIIE_shift.linkTo(
    ql.PiecewiseNaturalLogCubicDiscount(0, ql.Mexico(0), 
                                        qlHelper_TIIE(market_data=mkt_TIIE_shift) , 
                                        ql.Actual360()))

ibor_tiie_shift = ql.IborIndex('TIIE', ql.Period(ql.EveryFourthWeek), 1,
                         ql.MXNCurrency(), ql.Mexico(), non_workday_adj, False,
                         ql.Actual360(), crv_TIIE_shift)

ql_swap_tiie = ql.VanillaSwap(swap_position, notional, schdl_1y1y, fxd_rate, 
                              fxd_leg_daycount, schdl_1y1y, ibor_tiie_shift, 
                              flt_spread, flt_leg_daycount)
ql_swap_hedge_1y = ql.VanillaSwap(-1, 1.88e9, schdl_1y, 0.112425, 
                               fxd_leg_daycount, schdl_1y, ibor_tiie_shift, 
                               flt_spread, flt_leg_daycount)
ql_swap_hedge_2y = ql.VanillaSwap(1, 1.95e9, schdl_2y, 0.10356, 
                               fxd_leg_daycount, schdl_2y, ibor_tiie_shift, 
                               flt_spread, flt_leg_daycount)
swap_pricing_eng = ql.DiscountingSwapEngine(crv_TIIE_shift)
ql_swap_tiie.setPricingEngine(swap_pricing_eng)
ql_swap_hedge_1y.setPricingEngine(swap_pricing_eng)
ql_swap_hedge_2y.setPricingEngine(swap_pricing_eng)

npv_shift_1y1y = ql_swap_tiie.NPV()
npv_shift_hedge_1y = ql_swap_hedge_1y.NPV()
npv_shift_hedge_2y = ql_swap_hedge_2y.NPV()
npvChg_hedge = npv_shift_hedge_1y-hedge_1y_npv0 + npv_shift_hedge_2y-hedge_2y_npv0

print('NPV Change Afer Market Sold Off:'+\
      f'\n\tNo Hedge: {npv_shift_1y1y-T1y1y_npv0: ,.0f}'+\
      f'\n\tHedge: {npv_shift_1y1y-T1y1y_npv0 + npvChg_hedge:,.0f}')


# Lets see what happens with slope shifts
mkt_TIIE_shift = mkt_TIIE.copy()
mkt_TIIE_shift.loc[:4,'Quote'] += 20/100
mkt_TIIE_shift.loc[5:,'Quote'] += 30/100
rv_TIIE_shift = ql.RelinkableYieldTermStructureHandle()
crv_TIIE_shift.linkTo(
    ql.PiecewiseNaturalLogCubicDiscount(0, ql.Mexico(0), 
                                        qlHelper_TIIE(market_data=mkt_TIIE_shift) , 
                                        ql.Actual360()))

ibor_tiie_shift = ql.IborIndex('TIIE', ql.Period(ql.EveryFourthWeek), 1,
                         ql.MXNCurrency(), ql.Mexico(), non_workday_adj, False,
                         ql.Actual360(), crv_TIIE_shift)

ql_swap_tiie = ql.VanillaSwap(swap_position, notional, schdl_1y1y, fxd_rate, 
                              fxd_leg_daycount, schdl_1y1y, ibor_tiie_shift, 
                              flt_spread, flt_leg_daycount)
ql_swap_hedge_1y = ql.VanillaSwap(-1, 1.88e9, schdl_1y, 0.112425, 
                               fxd_leg_daycount, schdl_1y, ibor_tiie_shift, 
                               flt_spread, flt_leg_daycount)
ql_swap_hedge_2y = ql.VanillaSwap(1, 1.95e9, schdl_2y, 0.10356, 
                               fxd_leg_daycount, schdl_2y, ibor_tiie_shift, 
                               flt_spread, flt_leg_daycount)
swap_pricing_eng = ql.DiscountingSwapEngine(crv_TIIE_shift)
ql_swap_tiie.setPricingEngine(swap_pricing_eng)
ql_swap_hedge_1y.setPricingEngine(swap_pricing_eng)
ql_swap_hedge_2y.setPricingEngine(swap_pricing_eng)

npv_shift_1y1y = ql_swap_tiie.NPV()
npv_shift_hedge_1y = ql_swap_hedge_1y.NPV()
npv_shift_hedge_2y = ql_swap_hedge_2y.NPV()
npvChg_hedge = npv_shift_hedge_1y-hedge_1y_npv0 + npv_shift_hedge_2y-hedge_2y_npv0

print('NPV Change Afer Market Bear Steepened:'+\
      f'\n\tNo Hedge: {npv_shift_1y1y-T1y1y_npv0: ,.0f}'+\
      f'\n\tHedge: {npv_shift_1y1y-T1y1y_npv0 + npvChg_hedge:,.0f}')

