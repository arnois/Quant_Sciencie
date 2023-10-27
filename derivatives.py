#%% MODULES
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import norm
import QuantLib as ql
#import holidays
#%% EQUITY OPTIONS

# Function to price equity call options with Black-Scholes formula
def call_option_price_blackscholes(S: float, X: float, T: float, sigma: float,
                                   r: float, y: float) -> float:
    """
    Calculate equity call option price.

    Args:
    - S (float): Stock spot price.
    - X (float): Option strike price.
    - T (float): Option maturity or period (in years).
    - sigma (float): Stock's volatility.
    - r (float): Interest rate.
    - y (float): Stock's dividend yield.

    Returns:
        (float) Call option price.
    """
    # d
    d1 = (np.log(S/X)+(r-y+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # StdNormal CDF
    Phi = norm(0,1).cdf
    
    # shares term
    spot = S*np.exp(-y*T)*Phi(d1)
    
    # investment term
    strike = X*np.exp(-r*T)*Phi(d2)
    
    return spot - strike

# Function to price equity put options with Black-Scholes formula
def put_option_price_blackscholes(S: float, X: float, T: float, sigma: float,
                                  r: float, y: float) -> float:
    """
    Calculate equity put option price.

    Args:
    - S (float): Stock spot price.
    - X (float): Option strike price.
    - T (float): Option maturity or period (in years).
    - sigma (float): Stock's volatility.
    - r (float): Interest rate.
    - y (float): Stock's dividend yield.

    Returns:
        (float) Put option price.
    """
    # d
    d1 = (np.log(S/X)+(r-y+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # StdNormal CDF
    Phi = norm(0,1).cdf
    
    # shares term
    spot = S*np.exp(-y*T)*(1-Phi(d1))
    
    # investment term
    strike = X*np.exp(-r*T)*(1-Phi(d2))
    
    return strike - spot

"""
Lets price 1M ATM call/put options for UNH
"""
# UNH stock specs
S, X, T, sigma, r, y = 530, 530, 30/360, 20/100, 5.33/100, 1.5/100

# Options pricing
call_UNH = call_option_price_blackscholes(S, X, T, sigma, r, y)
put_UNH = put_option_price_blackscholes(S, X, T, sigma, r, y)

# Call-Put Parity: C-P = Se^(-yt) - Ke^(-rt)
cpp = np.round(np.array([call_UNH - put_UNH, S*np.exp(-y*T)-X*np.exp(-r*T)]),8)

print(f'\nUNH Options:\n\tCall: {np.round(call_UNH,6)}\n\tPut: {np.round(put_UNH,6)}')
print(f'\nUNH Call-Put Parity:\n\tC-P: {cpp[0]}\n\tS-K: {cpp[1]}')

###############################################################################
## Idem. using QuantLib

# Option specs
settle_qdt = ql.Date().todaysDate()
mty_qdt = settle_qdt + ql.Period(30, ql.Days)
payoff_call = ql.PlainVanillaPayoff(ql.Option.Call, X)
payoff_put = ql.PlainVanillaPayoff(ql.Option.Put, X)
europeanExercise = ql.EuropeanExercise(mty_qdt)
europeanOption_C = ql.VanillaOption(payoff_call, europeanExercise)
europeanOption_P = ql.VanillaOption(payoff_put, europeanExercise)
## Pricing engine
dayCount = ql.Actual360()
cal = ql.UnitedStates()
rTS = ql.YieldTermStructureHandle(ql.FlatForward(settle_qdt, r, dayCount))
yTS = ql.YieldTermStructureHandle(ql.FlatForward(settle_qdt, y, dayCount))
model_sigma = ql.BlackConstantVol(settle_qdt, cal, sigma, dayCount)
volatility = ql.BlackVolTermStructureHandle(model_sigma)
initialValue = ql.QuoteHandle(ql.SimpleQuote(S))
process = ql.BlackScholesMertonProcess(initialValue, yTS, rTS, volatility)
engine = ql.AnalyticEuropeanEngine(process)
europeanOption_C.setPricingEngine(engine)
europeanOption_P.setPricingEngine(engine)
## Options pricing
c,p = np.round(np.array([europeanOption_C.NPV(), europeanOption_P.NPV()]),6)
print(f'UNH Options (QL):\n\tCall: {c}\n\tPut: {p}')

# Function to create vanilla european equity options using BSM model w/QuantLib
def european_option(S: float, X: float, DTM: int, sigma: float, r: float, 
                    y: float, option_type: str = 'Call') -> ql.VanillaOption:
    """
    Create vanilla european equity option object.

    Args:
    - S (float): Stock spot price.
    - X (float): Option strike price.
    - DTM (int): Option days to maturity or period in days.
    - sigma (float): Stock's volatility.
    - r (float): Interest rate.
    - y (float): Stock's dividend yield.
    - option_type (str): Call or Put.

    Returns:
        (float) Vanilla european equity option.
    """
    # Conventions
    dayCount = ql.Actual360()
    cal = ql.UnitedStates()
    
    # Option maturity period
    settle_qdt = ql.Date().todaysDate()
    mty_qdt = settle_qdt + ql.Period(DTM, ql.Days)
    
    # Option payoff type
    europeanExercise = ql.EuropeanExercise(mty_qdt)
    if option_type.lower()[0] == 'c':
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, X)
    elif option_type.lower()[0] == 'p':
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, X)
    else:
        print("Incorrect option type specified!")
        return None

    # BSM option pricing model
    rTS = ql.YieldTermStructureHandle(ql.FlatForward(settle_qdt, r, dayCount))
    yTS = ql.YieldTermStructureHandle(ql.FlatForward(settle_qdt, y, dayCount))
    model_sigma = ql.BlackConstantVol(settle_qdt, cal, sigma, dayCount)
    volatility = ql.BlackVolTermStructureHandle(model_sigma)
    initialValue = ql.QuoteHandle(ql.SimpleQuote(S))
    process = ql.BlackScholesMertonProcess(initialValue, yTS, rTS, volatility)
    engine = ql.AnalyticEuropeanEngine(process)
    
    # Option
    europeanOption = ql.VanillaOption(payoff, europeanExercise)
    europeanOption.setPricingEngine(engine)
    
    return europeanOption

# Function to create vanilla american equity options using BSM model w/QuantLib
def american_option(S: float, X: float, val_qdt: ql.Date, mty_qdt: ql.Date, 
                    sigma: float, r: float, y: float, option_type: str = 'Call') -> ql.VanillaOption:
    """
    Create vanilla european equity option object.

    Args:
    - S (float): Stock spot price.
    - X (float): Option strike price.
    - val_qdt (ql.Date): Valuation date.
    - mty_qdt (ql.Date): Expiry date.
    - sigma (float): Stock's volatility.
    - r (float): Interest rate.
    - y (float): Stock's dividend yield.
    - option_type (str): Call or Put.

    Returns:
        (float) Vanilla european equity option.
    """
    # Conventions
    ql.Settings.instance().evaluationDate = val_qdt
    dayCount = ql.Actual360()
    cal = ql.UnitedStates()
    
    # Option maturity period
    settle_qdt = ql.Date().todaysDate()
    
    # Option payoff type
    amerExercise = ql.AmericanExercise(val_qdt, mty_qdt)
    if option_type.lower()[0] == 'c':
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, X)
    elif option_type.lower()[0] == 'p':
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, X)
    else:
        print("Incorrect option type specified!")
        return None

    # BSM option pricing model
    rTS = ql.YieldTermStructureHandle(ql.FlatForward(settle_qdt, r, dayCount))
    yTS = ql.YieldTermStructureHandle(ql.FlatForward(settle_qdt, y, dayCount))
    model_sigma = ql.BlackConstantVol(settle_qdt, cal, sigma, dayCount)
    volatility = ql.BlackVolTermStructureHandle(model_sigma)
    initialValue = ql.QuoteHandle(ql.SimpleQuote(S))
    process = ql.BlackScholesMertonProcess(initialValue, yTS, rTS, volatility)
    ## grid
    steps = 200
    rng = "pseudorandom" # could use "lowdiscrepancy"
    numPaths = 100000
    engine = ql.MCAmericanEngine(process, rng, steps, requiredSamples=numPaths)
    
    # Option
    amerOption = ql.VanillaOption(payoff, amerExercise)
    amerOption.setPricingEngine(engine)
    
    return amerOption

"""
Lets create 1M ATM call/put options for UNH
"""
# Option objects
UNH_Call = european_option(530, 530, 30, 20/100, 5.33/100, 0*1.5/100, 'call')
UNH_Put = european_option(530, 530, 30, 20/100, 5.33/100, 0*1.5/100, 'P')

# Option prices
print(f'\nUNH:\n\t Call: {UNH_Call.NPV():,.6f}\n\t Put: {UNH_Put.NPV():,.6f}')

# Option greeks
print(f'\nUNH Options Greeks:\n\tCall/Put\n\t\tDelta: {UNH_Call.delta():,.4f}'+\
      f'/{UNH_Put.delta():,.4f}\n\t\tGamma: {UNH_Call.gamma():,.4f}'+\
      f'/{UNH_Put.gamma():,.4f}\n\t\tVega: {UNH_Call.vega():,.4f}'+\
      f'/{UNH_Put.vega():,.4f}\n\t\tTheta: {UNH_Call.theta():,.4f}'+\
      f'/{UNH_Put.theta():,.4f}\n\t\tRho: {UNH_Call.rho():,.4f}/{UNH_Put.rho():,.4f}')

# American option verison
UNH_aCall = american_option(530, 530, ql.Date(27,10,2023), ql.Date(26,11,2023), 
                            20/100, 5.33/100, 0*1.5/100, 'call')
UNH_aPut = american_option(530, 530, ql.Date(27,10,2023), ql.Date(26,11,2023), 
                           20/100, 5.33/100, 0*1.5/100, 'P')
print(f'\nUNH:\n\t Call: {UNH_aCall.NPV():,.6f}\n\t Put: {UNH_aPut.NPV():,.6f}')



