#%% MODULES
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import norm
import QuantLib as ql
from matplotlib import pyplot as plt
#import holidays
#%% EQUITY OPTIONS PAYOFFS
# Function to compute equity option payoff at maturity
def option_payoff(S: float = 110, X: float = 100, op_type: str = 'call') -> float:
    """
    Calculate equity option payoff.

    Args:
    - S (float): Stock spot price.
    - X (float): Option strike price.
    - op_type (str): Option type.

    Returns:
        (float) Option payoff.
    """
    if op_type.lower()[0] == 'c':
        f = np.maximum(S-X,0)
    elif op_type.lower()[0] == 'p':
        f = np.maximum(X-S,0)
    else:
        print('Incorrect option type specified!')
    return f


#%% EQUITY OPTIONS VALUATION BOUNDS

# Function to get price bounds for an equity option
def option_bounds(S: float, X: float, T: float, r: float, op_type: str = 'call') -> np.array:
    """
    Calculate equity option price boundaries.

    Args:
    - S (float): Stock spot price.
    - X (float): Option strike price.
    - T (float): Option maturity or period (in years).
    - r (float): Interest rate.
    - op_type (str): Option type.

    Returns:
        (np.array) Option price bounds.
    """
    if op_type.lower()[0] == 'p':
        ub = X*np.exp(-r*T)
        lb = np.max([X*np.exp(-r*T) - S, 0])
    elif op_type.lower()[0] == 'c':
        ub = S
        lb = np.max([S-X*np.exp(-r*T),0])
    else:
        print('Incorrect option type!')
        return None
    
    return np.array([lb, ub])

# Function to get arbitrage portfolio if boundary conditions are not met
def option_offbounds_arb(S: float = 37, X: float = 40, T: float = 0.5, 
                         r: float = 0.05, op_type: str = 'put', op_price: float = 2) -> float:
    """
    Calculate arbitrage portfolio for out of bounds option price.

    Args:
    - S (float): Stock spot price.
    - X (float): Option strike price.
    - T (float): Option maturity or period (in years).
    - r (float): Interest rate.
    - op_type (str): Option type.
    - op_price (float): Option price.

    Returns:
        (float) Minimum arbitrage amount.
    """
    # Option bounds
    op_bounds = option_bounds(S,X,T,r,op_type)
    if op_bounds is None: return None
    # Out of bounds conditions
    isLBOff = op_price < op_bounds[0]
    isUBOff = op_price > op_bounds[1]
    isOOB = isLBOff or isUBOff
    
    # If out of bounds check side
    if isOOB:
        if op_type.lower()[0] == 'c':
            # LB/UB off
            if isLBOff: # LB off
                print(f'\nShort Stock at {S:,.2f} + Buy Call at {op_price:,.3f}')
                print(f'Invest {S-op_price:,.3f} at {r:,.2%} for {T:,.2f} years\n')
                future_invest = (S-op_price)*np.exp(r*T)
                min_arb_amt = future_invest - X
            else: # UB off
                print(f'\nSell Call at {op_price:,.3f} + Buy Stock at {S:,.2f}\n')
                min_arb_amt = op_price - S
        elif op_type.lower()[0] == 'p':
            # LB/UB off
            if isLBOff: # LB off
                print('\nBuy Stock + Put')
                print(f'Borrow {S+op_price:,.3f} at {r:,.2%} for {T:,.2f} years\n')
                future_liab = -1*(S+op_price)*np.exp(r*T)
                min_arb_amt = future_liab + X
            else: # UB off
                print(f'\nSell Put at {op_price:,.3f}')
                print(f'Invest premium at {r:,.2%} for {T:,.2f} years\n')
                min_arb_amt = op_price*np.exp(r*T) - X
        else:
            print('Incorrect option type!')
            return None
    else:
        print('Option price is between non-arbitrage bounds.')
        return None
    
    return min_arb_amt

#%% EQUITY OPTIONS PRICING BINOMIAL
# Function to compute the risk-neutral probability
def risk_neutral_p(S: float = 100, Su: float = 110, Sd: float = 90, 
                   r: float = 0.05, T: float = 1) -> float:
    """
    Calculate risk-neutral probability.

    Args:
    - S (float): Stock spot price.
    - Su (float): Stock spot price at option's maturity to the upside.
    - Sd (float): Stock spot price at option's maturity to the downside.
    - T (float): Option maturity or period (in years).
    - r (float): Interest rate.

    Returns:
        (float) Risk-neutral probability.
    """
    # Terminal movements for stock
    u, d = Su/S, Sd/S
    
    return (np.exp(r*T)-d)/(u-d)

# Function to price equity options using binomial model
def option_price_binomial(S: float = 100, X: float = 105, Su: float = 110, 
                          Sd: float = 90, r: float = 0.05, T: float = 1, 
                          op_type: str = 'call') -> float:
    """
    Calculate option price based on a one-period binomial model.

    Args:
    - S (float): Stock spot price.
    - X (float): Strike price.
    - Su (float): Stock spot price at option's maturity to the upside.
    - Sd (float): Stock spot price at option's maturity to the downside.
    - T (float): Option maturity or period (in years).
    - r (float): Interest rate.
    - op_type (str): Option type.

    Returns:
        (float) Option price.
    """
    # Risk-neutral probability
    p = risk_neutral_p(S, Su, Sd, r, T)
    # Option binomial payoffs
    if op_type.lower()[0] == 'c':
        payoff_u = np.max([Su-X, 0])
        payoff_d = np.max([Sd-X, 0])
    elif op_type.lower()[0] == 'p':
        payoff_u = np.max([X-Su, 0])
        payoff_d = np.max([X-Sd, 0])
    else:
        print('Incorrect option type specified.')
        return None
    # Option payoff risk-neutral expected value
    fF = p*payoff_u+(1-p)*payoff_d
    
    return fF*np.exp(-r*T)

#%% EQUITY OPTIONS PRICING BSM

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

#%% EQUITY OPTIONS PRICING BSM
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
    rng = "pseudorandom"
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
UNH_Call = european_option(530, 530, 30, 20/100, 5.33/100, 1*1.5/100, 'call')
UNH_Put = european_option(530, 530, 30, 20/100, 5.33/100, 1*1.5/100, 'P')

# Option prices
print(f'\nUNH:\n\t Call: {UNH_Call.NPV():,.6f}\n\t Put: {UNH_Put.NPV():,.6f}')

# Option greeks
print(f'\nUNH Euro Options Greeks:\n\tCall/Put\n\t\tDelta: {UNH_Call.delta():,.4f}'+\
      f'/{UNH_Put.delta():,.4f}\n\t\tGamma: {UNH_Call.gamma():,.4f}'+\
      f'/{UNH_Put.gamma():,.4f}\n\t\tVega: {UNH_Call.vega():,.4f}'+\
      f'/{UNH_Put.vega():,.4f}\n\t\tTheta: {UNH_Call.theta():,.4f}'+\
      f'/{UNH_Put.theta():,.4f}\n\t\tRho: {UNH_Call.rho():,.4f}/{UNH_Put.rho():,.4f}')

# American option verison
UNH_aCall = american_option(530, 530, ql.Date(27,10,2023), ql.Date(26,11,2023), 
                            20/100, 5.33/100, 1*1.5/100, 'call')
UNH_aPut = american_option(530, 530, ql.Date(27,10,2023), ql.Date(26,11,2023), 
                           20/100, 5.33/100, 1*1.5/100, 'P')
print(f'\nUNH:\n\t Call: {UNH_aCall.NPV():,.6f}\n\t Put: {UNH_aPut.NPV():,.6f}')


#%% EQUITY OPTIONS STRATS
###############################################################################
# PAYOFF CURVES
# Function to compute bull spread payoff using calls
def option_bullspread_calls(S: float = 110, X1: float = 100, X2: float = 150) -> float:
    """
    Calculate bullspread option strategy payoff.

    Args:
    - S (float): Stock spot price.
    - X1 (float): Long call strike price.
    - X2: (float): Short call strike price

    Returns:
        (float) Bullspread strategy payoff.
    """
    # Long call payoff
    long_c = option_payoff(S, X1, op_type = 'call')
    # Short call payoff
    short_c = -1*option_payoff(S, X2, op_type = 'call')
    
    return long_c + short_c

# Function to compute bear spread payoff using calls
def option_bearspread_calls(S: float = 110, X1: float = 100, X2: float = 150) -> float:
    """
    Calculate bearspread option strategy payoff.

    Args:
    - S (float): Stock spot price.
    - X1 (float): Short call strike price.
    - X2: (float): Long call strike price.

    Returns:
        (float) Bearspread strategy payoff.
    """
    # Long call payoff
    long_c = option_payoff(S, X2, op_type = 'call')
    # Short call payoff
    short_c = -1*option_payoff(S, X1, op_type = 'call')
    
    return long_c + short_c

# Function to compute straddle payoff
def option_straddle(S: float = 110, X: float = 100) -> float:
    """
    Calculate straddle option strategy payoff.

    Args:
    - S (float): Stock spot price.
    - X (float): Long call & put strike price.

    Returns:
        (float) Straddle strategy payoff.
    """
    # Long call payoff
    long_c = option_payoff(S, X, op_type = 'call')
    # long put payoff
    long_p = option_payoff(S, X, op_type = 'put')
    
    return long_c + long_p

# Function to compute straddle payoff
def option_strangle(S: float = 110, X1: float = 100, X2: float = 150) -> float:
    """
    Calculate straddle option strategy payoff.

    Args:
    - S (float): Stock spot price.
    - X1 (float): Long put strike price.
    - X2: (float): Long call strike price.

    Returns:
        (float) Strangle strategy payoff.
    """
    # Long call payoff
    long_c = option_payoff(S, X2, op_type = 'call')
    # long put payoff
    long_p = option_payoff(S, X1, op_type = 'put')
    
    return long_c + long_p

# Function to compute butterfly payoff
def option_butterfly(S: float = 110, X1: float = 100, X2: float = 120, X3: float = 140) -> float:
    """
    Calculate butterfly option strategy payoff.

    Args:
    - S (float): Stock spot price.
    - X1 (float): Long 1 call strike price.
    - X2: (float): Short 2 calls strike price.
    - X3: (float): Long 1 calls strike price

    Returns:
        (float) Butterfly strategy payoff.
    """
    # Long 1 call payoff at X1
    long_1c_X1 = option_payoff(S, X1, op_type = 'call')
    # Short 2 calls payoff
    short_2c = -2*option_payoff(S, X2, op_type = 'call')
    # Long 1 call payoff at X3
    long_1c_X3 = option_payoff(S, X3, op_type = 'call')
    
    return long_1c_X1 + short_2c + long_1c_X3

# Function to compute iron condor payoff
def option_ironcondor(S: float = 110, X1: float = 100, X2: float = 120, X3: float = 140, X4: float = 160) -> float:
    """
    Calculate iron condor option strategy payoff.

    Args:
    - S (float): Stock spot price.
    - X1 (float): Long 1 call strike price.
    - X2: (float): Short 1 call strike price.
    - X3: (float): Short 1 call strike price.
    - X4: (float): Long 1 call strike price

    Returns:
        (float) Iron condor strategy payoff.
    """
    # Long call payoff at X1
    long_c_X1 = option_payoff(S, X1, op_type = 'call')
    # Short call payoff at X2
    short_c_X2 = -1*option_payoff(S, X2, op_type = 'call')
    # Short call payoff at X3
    short_c_X3 = -1*option_payoff(S, X3, op_type = 'call')
    # Long call payoff at X4
    long_c_X4 = option_payoff(S, X4, op_type = 'call')
    
    return long_c_X1 + short_c_X2 + short_c_X3 + long_c_X4



###############################################################################
# STRAT PRICES
# Function to compute bull spread price using calls
def price_bullspread_calls(S: float = 110, X1: float = 100, X2: float = 150, 
                           DTM: float = 30, sigma: float = 0.30, r: float = 0.05,
                           y: float = 0.015) -> float:
    """
    Calculate bullspread option strategy price.

    Args:
    - S (float): Stock spot price.
    - X1 (float): Long call strike price.
    - X2: (float): Short call strike price.
    - DTM (int): Option days to maturity or period in days.
    - sigma (float): Stock's volatility.
    - r (float): Interest rate.
    - y (float): Stock's dividend yield.

    Returns:
        (float) Bullspread strategy price.
    """
    # Long call
    long_c = european_option(S, X1, DTM, sigma, r, y, option_type = 'Call') 
    # Short call
    short_c = european_option(S, X2, DTM, sigma, r, y, option_type = 'Call') 
    
    return long_c.NPV() - short_c.NPV()

# Function to compute bear spread price using calls
def price_bearspread_calls(S: float = 110, X1: float = 100, X2: float = 150, 
                           DTM: float = 30, sigma: float = 0.30, r: float = 0.05,
                           y: float = 0.015) -> float:
    """
    Calculate bearspread option strategy price.

    Args:
    - S (float): Stock spot price.
    - X1 (float): Short call strike price.
    - X2: (float): Long call strike price.
    - DTM (int): Option days to maturity or period in days.
    - sigma (float): Stock's volatility.
    - r (float): Interest rate.
    - y (float): Stock's dividend yield.

    Returns:
        (float) Bearspread strategy price.
    """
    # Long call
    long_c = european_option(S, X2, DTM, sigma, r, y, option_type = 'Call') 
    # Short call
    short_c = european_option(S, X1, DTM, sigma, r, y, option_type = 'Call') 
    
    return long_c.NPV() - short_c.NPV()

# Function to compute strangle price
def price_strangle(S: float = 110, X1: float = 100, X2: float = 150, 
                   DTM: float = 30, sigma: float = 0.30, r: float = 0.05,
                   y: float = 0.015) -> float:
    """
    Calculate strangle option strategy price.

    Args:
    - S (float): Stock spot price.
    - X1 (float): Long 1 put strike price.
    - X2: (float): Long 1 call strike price.
    - DTM (int): Option days to maturity or period in days.
    - sigma (float): Stock's volatility.
    - r (float): Interest rate.
    - y (float): Stock's dividend yield.

    Returns:
        (float) Strangle strategy price.
    """
    # Long 1 call at X2
    long_1c_X2 = european_option(S, X2, DTM, sigma, r, y) 
 
    # Long 1 put at X1
    long_1p_X1 = european_option(S, X1, DTM, sigma, r, y, 'put')
    
    return long_1c_X2.NPV() + long_1p_X1.NPV()

# Function to compute fly price
def price_butterfly(S: float = 110, X1: float = 100, X2: float = 150, 
                    X3: float = 200, DTM: float = 30, sigma: float = 0.30, 
                    r: float = 0.05, y: float = 0.015) -> float:
    """
    Calculate fly option strategy price.

    Args:
    - S (float): Stock spot price.
    - X1 (float): Long 1 call strike price.
    - X2: (float): Short 2 calls strike price.
    - X3: (float): Long 1 call strike price.
    - DTM (int): Option days to maturity or period in days.
    - sigma (float): Stock's volatility.
    - r (float): Interest rate.
    - y (float): Stock's dividend yield.

    Returns:
        (float) Butterfly strategy price.
    """
    # Long 1 call at X1
    long_1c_X1 = european_option(S, X1, DTM, sigma, r, y) 
    
    # Short 2 calls at X2
    short_2c_X2 = european_option(S, X2, DTM, sigma, r, y) 
 
    # Long 1 call at X3
    long_1c_X3 = european_option(S, X3, DTM, sigma, r, y)
    
    return long_1c_X1.NPV() - 2*short_2c_X2.NPV() + long_1c_X3.NPV()


"""
Lets plot a bullspread payoff curve
"""
S_prices = np.arange(80,110,1)
X1, X2 = 90, 100
payoff_bullspread = option_bullspread_calls(S_prices, X1, X2)
plt.plot(S_prices, payoff_bullspread)
plt.title('Bullspread Payoff')
plt.tight_layout(); plt.show()
# Strat price
p_bullsprd = price_bullspread_calls(S=88,X1=X1,X2=X2,DTM=30,sigma=0.30,r=0.05,y=0)
curve_bullsprd = payoff_bullspread - p_bullsprd
plt.plot(S_prices, curve_bullsprd)
plt.axhline(y=0, c='gray')
plt.title('Bullspread PnL at T')
plt.tight_layout(); plt.show()

"""
Lets plot a straddle payoff curve
"""
X = 95
payoff_straddle = option_straddle(S_prices, X)
plt.plot(S_prices, payoff_straddle)
plt.title('Straddle Payoff')
plt.tight_layout(); plt.show()
# Strat price

"""
Lets plot a strangle payoff curve
"""
X1, X2 = 92, 97
payoff_strangle = option_strangle(S_prices, X1, X2)
plt.plot(S_prices, payoff_strangle)
plt.title('Strangle Payoff')
plt.tight_layout(); plt.show()
# Strat price
p_strangle = price_strangle(S=95,X1=X1,X2=X2,DTM=30,sigma=0.30,r=0.05,y=0)
curve_strangle = payoff_strangle - p_strangle
plt.plot(S_prices, curve_strangle)
plt.axhline(y=0, c='gray')
plt.title('Strangle PnL at T')
plt.tight_layout(); plt.show()

"""
Lets plot a butterfly payoff curve
"""
X1, X2, X3 = 90, 95, 100
payoff_fly = option_butterfly(S_prices, X1, X2, X3)
plt.plot(S_prices, payoff_fly)
plt.title('Fly Payoff')
plt.tight_layout(); plt.show()
# Strat rice
p_fly = price_butterfly(S=111,X1=X1,X2=X2,X3=X3,DTM=30,sigma=0.30,r=0.05,y=0)
curve_fly = payoff_fly - p_fly
plt.plot(S_prices, curve_fly)
plt.axhline(y=0, c='gray')
plt.title('Butterfly PnL at T')
plt.tight_layout(); plt.show()

"""
Lets plot an iron condor payoff curve
"""
X1, X2, X3, X4 = 88, 92, 97, 101
payoff_ironc = option_ironcondor(S_prices, X1, X2, X3, X4)
plt.plot(S_prices, payoff_ironc)
plt.title('Iron Condor Payoff')
plt.tight_layout(); plt.show()






