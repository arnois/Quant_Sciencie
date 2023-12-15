#%% MODULES
import numpy as np
import pandas as pd
import datetime as dt
import holidays

#%% DATA IMPORT

# Function to import zcb data from price vendor VALMER
def data_from_valmer_CETE(fpath: str = r'C:\Users\jquintero\Downloads',
                           fdate: str = '20231122') -> pd.DataFrame:
    """
    Import CETE data.

    Args:
    - fpath (str): Path where the VALMER file is located.
    - fdate (str): Date for retreiving closing data from VALMER.

    Returns:
        (pd.DataFrame) Characteristics for each CETE.
    """
    # File default name
    fname = r'\Niveles_de_Valuacion_'
    
    # Whole file path 
    path = fpath+fname+fdate+'.xlsx'
        
    # File read
    tmpdf = pd.read_excel(path)
    
    # Indexes of MBONOS loc
    r,c = np.where(tmpdf == "Directo_Cete's")
    r,c = r[0], c[0]
    
    # First True value
    idx_1stT = tmpdf.iloc[r:, c].isnull().idxmax()
    
    # Second True value
    idx_2ndT = (tmpdf.iloc[r:, c]).loc[idx_1stT:].isnull().idxmax()
    
    # Data
    df_cets = tmpdf.iloc[r:idx_2ndT, 1:4].dropna()
    df_cets.columns = df_cets.iloc[0]
    df_cets = df_cets.drop(df_cets.iloc[0].name).reset_index(drop=True)
    df_cets.columns.rename('', inplace=True)
    
    # Data proc
    df_cets.insert(1,'Mty',df_cets['Instrumento'].\
                    apply(lambda c: dt.datetime.strptime(c[-6:],'%y%m%d')))
    renamedic = {'Instrumento':'ID', 'Dias X Vencer': 'DTM', 'Hoy':'YTM'}
    df_cets = df_cets.rename(columns=renamedic)
    
    return df_cets

# Function to import bond data from price vendor VALMER
def data_from_valmer_MBONO(fpath: str = r'C:\Users\jquintero\Downloads',
                           fdate: str = '20231122') -> pd.DataFrame:
    """
    Import MBONO data.

    Args:
    - fpath (str): Path where the VALMER file is located.
    - fdate (str): Date for retreiving closing data from VALMER.

    Returns:
        (pd.DataFrame) Characteristics for each MBONO.
    """
    # File default name
    fname = r'\Niveles_de_Valuacion_'
    
    # Whole file path 
    path = fpath+fname+fdate+'.xlsx'
    
    # Bond CPN rate by mty
    specs_path = r'H:\mbonos_specs.xlsx'
    df_mbonospecs = pd.read_excel(specs_path)
    df_mbonospecs['Maturity'] = df_mbonospecs['Maturity'].\
        apply(lambda c: dt.datetime.strptime(c,'%d/%m/%Y'))
    df_mbonospecs = df_mbonospecs[~df_mbonospecs[['Maturity']].duplicated()]
    
    # File read
    tmpdf = pd.read_excel(path)
    
    # Indexes of MBONOS loc
    r,c = np.where(tmpdf == 'Bonos de Tasa Fija (BONOS M)')
    r,c = r[0], c[0]
    
    # First True value
    idx_1stT = tmpdf.iloc[r:, c].isnull().idxmax()
    
    # Second True value
    idx_2ndT = (tmpdf.iloc[r:, c]).loc[idx_1stT+1:].isnull().idxmax()
    
    # Data
    df_bonos = tmpdf.iloc[r:idx_2ndT, 1:4].dropna()
    df_bonos.columns = df_bonos.iloc[0]
    df_bonos = df_bonos.drop(df_bonos.iloc[0].name).reset_index(drop=True)
    df_bonos.columns.rename('', inplace=True)
    
    # Data proc
    df_bonos.insert(1,'Mty',df_bonos['Instrumento'].\
                    apply(lambda c: dt.datetime.strptime(c[-6:],'%y%m%d')))
    renamedic = {'Instrumento':'ID', 'Plazos': 'DTM', 'Hoy':'YTM'}
    df_bonos = df_bonos.merge(df_mbonospecs[['Maturity', 'Coupon']], 
                              how='left', left_on='Mty', right_on='Maturity').\
        drop('Mty',axis=1).rename(columns=renamedic)
    
    return df_bonos

#%% Interest Rates

# Function to measure interest 
def calc_interest(VT: float = 100, V0: float = 96, t: float = 0.5, 
                  im: str = 'eff') -> float:
    """
    Calculate interest earned.

    Args:
    - VT (float): Investment at end of period.
    - V0 (float): Investment at beginning of period.
    - t (float): Period (in years).
    - im (str): Interest measure: Effective, Annually.

    Returns:
        (float) Interest amount earned in the specified measure.
    """
    if im.lower()[0] == 'e':
        I = (VT - V0)
    elif im.lower()[0] == 'a':
        I = (VT - V0)/t
    else:
        print("Incorrect measure specified!")
        return np.nan
    
    return I

# Function to measure interest rate
def calc_interest_rate(VT: float = 100, V0: float = 96, t: float = 0.5, 
                       im: str = 'eff') -> float:
    """
    Calculate interest rate earned.

    Args:
    - VT (float): Investment at end of period.
    - V0 (float): Investment at beginning of period.
    - t (float): Period (in years).
    - im (str): Interest measure: Effective, Annually.

    Returns:
        (float) Interest rate earned in the specified measure.
    """
    I = calc_interest(VT,V0,t,im)
    
    return I/V0

# Accumulation factor function given an interest rate 
def accum_f(i: float = 0.04, n: float = 1, itype: str = 'simple', 
            m: float = 1) -> float:
    """
    Calculate end-of-period value of a $1-investment.

    Args:
    - i (float): Interest rate.
    - n (float): Investment period (in years, for example, 1/12 for 1 month).
    - itype (str): Type of interest rate to accrue investment.
    - m (float): Compounding frequency (when applicable, in years).

    Returns:
        (float) Accumulated value at the end of the nth period for
        a $1-investment at the given interest rate (i).
    """
    # Lets set type of interest
    if itype.lower()[0] == 's':
        # Simple interest rate
        a_n = (1+i*n)
    elif itype.lower()[0] == 'c':
        # Compounding interest rate
        a_n = (1+i*m)**(n/m)
    else:
        print("\nInterest type incorrectly specified.\nPlease try again!\n")
        return np.nan
        
    return a_n

# Discount factor function given an interest rate 
def DF(i: float = 0.04, n: float = 1, 
            itype: str = 'simple', m: float = 1) -> float:
    """
    Calculate the beginning-period value of a $1 accumulated at end of period.

    Args:
    - i (float): Interest rate.
    - n (float): Investment period (in years, for example, 1/12 for 1 month).
    - itype (str): Type of interest rate to accrue investment.
    - m (float): Compounding frequency (when applicable, in years).

    Returns:
        (float) Discounted value at the beginning of an nth period for
        a $1 value accumulated for a given interest rate (i) at end of period.
    """
    # Lets set type of interest
    if itype.lower()[0] == 's':
        # Simple interest rate
        DF_n = 1/(1+i*n)
    elif itype.lower()[0] == 'c':
        # Compounding interest rate
        DF_n = (1+i*m)**(-n/m)
    else:
        print("\nInterest type incorrectly specified.\nPlease try again!\n")
        return np.nan
        
    return DF_n
###############################################################################
"""
Lets compute simple and compounding interests
"""
# Simple
V_simple = 100*accum_f(i=0.10, n=np.arange(0,4,1), itype='simple')
# Compounding
V_comp = 100*accum_f(i=0.10, n=np.arange(0,4,1), itype='compound', m=1)
# Comparison
pd.DataFrame({'Simp':V_simple, 'Comp':V_comp}, index=np.arange(0,4,1))

#%% TVM
"""
How do we compute the present value?
"""
# Lets compute the PV of a $100 to be earned in 1.5 years at a simple 11% rate 
DF(0.11, 1.5)*100
# Ibid, but with monthly compounding rate
DF(11/100, 1.5, 'c', 1/12)*100

# Function to compute present value of earned amount at end of period
def PV(cf: float = 100, i: float = 0.04, n: float = 1, 
       itype: str = 'simple', m: float = 1) -> float:
    """
    Calculate the present value of amount to be earned at the end of the period

    Args:
    - cf (float): Cashflow or amount to discount.
    - i (float): Interest rate.
    - n (float): Investment period (in years, for example, 1/12 for 1 month).
    - itype (str): Type of interest rate to accrue investment.
    - m (float): Compounding frequency (when applicable, in years).

    Returns:
        (float) Present value of the cashflow or amount at period's end.
    """
    # Discounting factor
    discount_factor = DF(i=i, n=n, itype=itype, m=m)
    
    return discount_factor*cf
###############################################################################
#%% BOND PRICING

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
    float: ZCB price.
    """
    return PV(cf=nominal,i=r,n=dtm/ybc,itype='simple',m=dtm/ybc)
###############################################################################
"""
Lets price a CETE.
"""
print(f'CETE Price: {zcb_price():,.6f}\n\tr: 11.20%\n\tDTM: 182')

"""
Lets price a 10-year maturity bond yielding 6.25% interest, 
paying yearly a 4.5%-coupon over a par value of $100.
"""
# Bond characteristics
n_maturity = 10
par_value = 100
coupon_rate = 4.5/100
bond_yield = 6.25/100
comp_freq = 1

# We create the vector of yearly payments 
coupon_pmt = coupon_rate*par_value 
cashflows = np.repeat(coupon_pmt, n_maturity)
# At maturity we will recieve the 100 par 
cashflows[-1] += 100

# Making use of the PV function to discount bond's CFs
PV_cf = PV(cashflows, bond_yield, np.arange(1,11,1), 'c', comp_freq)

# Sum of the present value of the bond's cashflows
fair_value = np.sum(PV_cf)
print(f'The price of the bond is: ${fair_value: .6f}')

# Bond as DataFrame
dfb = pd.DataFrame({'period':np.arange(1,11,1), 'CF':cashflows, 'PV_CF':PV_cf})
print(f"Bond Specs:\n{dfb}")

# Function to price a bond
def bond_price(settle_date: dt.date, mty_date: dt.date, coup_freq: int, 
               par_value: float, coupon_rate: float, ytm: float, 
               ybc: int = 360) -> tuple:
    """
    Calculate bond price.

    Args:
    - settle_date (dt.date): Settlement date.
    - mty_date (dt.date): Maturity date.
    - coup_freq (int): Frequency of coupon payment in days.
    - par_value (float): Nominal or face value.
    - coupon_rate (float): Coupon rate
    - ytm (float): Yield to maturity.
    - ybc (int): Days in a year convention for periods.

    Returns:
        (tuple) Bond price with cashflows characteristics table.
    """
    # Maturity in days
    T = (mty_date - settle_date).days
    
    # Coupons left
    n_coup_left = np.ceil(T/coup_freq).astype(int)

    # Payment dates
    m = coup_freq/ybc
    CF_dates = pd.bdate_range(start=mty_date, periods=n_coup_left, 
                              freq='-'+str(coup_freq)+'D', 
                              holidays=holidays.MX()).sort_values()

    # Coupon payments
    c_pmt = coupon_rate*par_value*m
    bond_CF = np.repeat(c_pmt, n_coup_left)
    bond_CF[-1] += par_value

    # Discount Factors
    bond_CF_dtm = np.array([x.days for x in (CF_dates - settle_date)])
    bond_CF_period = bond_CF_dtm/ybc
    bond_DF = DF(ytm, bond_CF_period, 'c', m)

    # Bond specs dataframe
    df_bond_specs = pd.DataFrame({'n_coupon': np.array(range(1,n_coup_left+1)),
                           'date_pmt': CF_dates,
                           'dtm': bond_CF_dtm,
                            'period': bond_CF_period, 
                            'CF': bond_CF, 
                            'DF': bond_DF})

    # Bond pricing
    bond_price = (df_bond_specs['CF']*df_bond_specs['DF']).sum()
    
    return bond_price, df_bond_specs
###############################################################################
"""
Lets price the MBONO May31.
"""
# Trade & Settle dates
trade_date = dt.datetime.now().date()
settle_date = trade_date + pd.offsets.\
    CustomBusinessDay(n=2, calendar=holidays.MX())
# M31 characteristics
mty_str = '2031-05-29'
yearbase_conv = 360
mty_date = dt.datetime.strptime(mty_str, '%Y-%m-%d')
T = (mty_date - settle_date).days
coup_freq = 182
n_coup_left = np.ceil(T/coup_freq).astype(int)
cf_dates = pd.bdate_range(start=mty_date, periods=n_coup_left, 
                          freq='-'+str(coup_freq)+'D', 
                          holidays=holidays.MX()).sort_values()
vn = 100
coupon_rate = 7.75/100
ytm = 9.60/100

# M31 Price & CF table
m31_price, df_m31 = bond_price(settle_date, mty_date, coup_freq, vn, 
                               coupon_rate, ytm, yearbase_conv)
# accrued interest
m31_accInt = (182-df_m31.iloc[0]['dtm'])/360*coupon_rate*vn
m31_price_cln = m31_price - m31_accInt

# M31 pricing output
print(f'The price of the bond is: ${m31_price: .6f}')
print(f'Bond clean price: ${m31_price_cln: .6f}')
print(f'Bond accrued interest: ${m31_accInt: .6f}')
print(f"M31 Specs:\n{df_m31}")


#%% BOND RISK MEASURES

# Function to compute bond duration
def bond_risk_dur(df_bond_specs: pd.DataFrame, m: float) -> float:
    """
    Calculate bond duration in years.

    Args:
    - df_bond_specs (pd.DataFrame): Bond specs with the periods, cashflows and
        discounting factors data.
    - m (float): Compounding frequency.

    Returns:
        (float) Bond duration in years.
    """
    # PV of cashflows
    PV_CF = df_bond_specs['CF']*df_bond_specs['DF']
    # Price
    P = np.sum(PV_CF)
    # Cashflows weights
    w = PV_CF/P
    # Coupon periods
    t = df_bond_specs['period']/m
    # Duration
    dur = (w @ t)*m
    
    return dur

# Function to compute bond modified duration
def bond_risk_mdur(dur: float, y: float, m: float) -> float:
    """
    Calculate bond modified duration.

    Args:
    - dur (float): Bond's duration.
    - y (float): Yield to maturity.
    - m (float): Compounding frequency.

    Returns:
        (float) Bond modified duration.
    """
    return dur/(1+y*m)

# Function to compute bond dv01 risk measure
def bond_risk_dv01(df_bond_specs: pd.DataFrame, y: float, m: float) -> float:
    """
    Calculate bond DV01 risk.

    Args:
    - df_bond_specs (pd.DataFrame): Bond specs with the periods, cashflows and
        discounting factors data.
    - y (float): Yield to maturity.
    - m (float): Compounding frequency.

    Returns:
        (float) Bond DV01 risk.
    """
    # PV of cashflows
    PV_CF = df_bond_specs['CF']*df_bond_specs['DF']
    # Price
    P = np.sum(PV_CF)
    # Duration
    dur = bond_risk_dur(df_bond_specs, m)
    # Modified duration
    mdur = bond_risk_mdur(dur, y, m)
    # Price change wrt ytm
    derivP = -1*mdur*P
    
    return derivP*0.0001
###############################################################################
"""
Lets compute the MBONO May31 Risk Measures.
"""
# Duration
m31_dur = bond_risk_dur(df_m31, coup_freq/yearbase_conv)
# Modified Duration
m31_mdur = bond_risk_mdur(m31_dur, ytm, coup_freq/yearbase_conv)
# DV01 via ModDur
m31_dv01_mdur = bond_risk_dv01(df_m31, ytm, coup_freq/yearbase_conv)
df_m31_risks = pd.DataFrame({'Measure': [m31_dur, m31_mdur, m31_dv01_mdur]}, 
                            index = ['Dur', 'ModDur', 'DV01'])
print('\nM31 Risk Measures\n')
print(df_m31_risks)

###############################################################################
from sympy import symbols, diff

# Function to compute bond price
def B(ytm: float, settle_date: dt.date, mty_date: dt.date, coup_freq: int, 
      par_value: float, coupon_rate: float, ybc: int) -> float:
    # Maturity in days
    T = (mty_date - settle_date).days
    
    # Coupons left
    n_coup_left = np.ceil(T/coup_freq).astype(int)

    # Payment dates
    m = coup_freq/ybc
    CF_dates = pd.bdate_range(start=mty_date, periods=n_coup_left, 
                              freq='-'+str(coup_freq)+'D', 
                              holidays=holidays.MX()).sort_values()

    # Coupon payments
    c_pmt = coupon_rate*par_value*m
    bond_CF = np.repeat(c_pmt, n_coup_left)
    bond_CF[-1] += par_value

    # Discount Factors
    bond_CF_dtm = np.array([x.days for x in (CF_dates - settle_date)])
    bond_CF_period = bond_CF_dtm/ybc
    bond_DF = DF(ytm/100, bond_CF_period, 'c', m)
    
    return (bond_CF*bond_DF).sum()
    
# Function to compute first derivative of bond price wrt yield
def deriv_B(ytm: float, settle_date: dt.date, mty_date: dt.date, coup_freq: int, 
            par_value: float, coupon_rate: float, ybc: int) -> float:
    # Bond price and cashflow table
    price, df_specs = bond_price(settle_date, mty_date, coup_freq, par_value,
                                 coupon_rate, ytm, ybc)
    # Macaulay duration
    b_dur = bond_risk_dur(df_specs, coup_freq/ybc)
    
    # Modified duration
    modD = bond_risk_mdur(b_dur, ytm, coup_freq/ybc)
    
    # Price deriv wrt ytm; for delta ytm = 100 bps
    #derivPrice_y = -1*modD*price/100
    # Price deriv wrt ytm; for delta ytm = 1 bps
    #derivPrice_y = -1*modD*price/10000
    
    #return derivPrice_y
    return -modD*price

# Function to compute first derivative of bond's duration wrt yield
def deriv_D(ytm: float, settle_date: dt.date, mty_date: dt.date, coup_freq: int, 
            par_value: float, coupon_rate: float, ybc: int) -> float:
    # Bond price and cashflow table
    price, df_specs = bond_price(settle_date, mty_date, coup_freq, par_value,
                                 coupon_rate, ytm, ybc)
    # Macaulay duration
    m = coup_freq/ybc
    D = bond_risk_dur(df_specs, m)
    
    # PV of cashflows
    PV_CF = df_specs['CF']*df_specs['DF']
    # Price
    P = np.sum(PV_CF)
    # Cashflows weights
    w = PV_CF/P
    # Coupon periods
    sqrdt = (df_specs['period']/m)**2
    # sumation
    sum1 = (w @ sqrdt)
    
    # Sumation factors
    sF1 = D**2/(1+ytm*m)
    sF2 = sum1*(m**2)/(1+ytm*m)
    
    return sF1 - sF2

# Function to compute first derivative of bond's modified duration wrt yield
def deriv_modD(ytm: float, settle_date: dt.date, mty_date: dt.date, coup_freq: int, 
            par_value: float, coupon_rate: float, ybc: int) -> float:
    # Bond price and cashflow table
    price, df_specs = bond_price(settle_date, mty_date, coup_freq, par_value,
                                 coupon_rate, ytm, ybc)
    # Macaulay duration
    m = coup_freq/ybc
    D = bond_risk_dur(df_specs, m)
    
    # Macaulay duration 1st derivative
    derivD = deriv_D(ytm, settle_date, mty_date, coup_freq, par_value, coupon_rate, ybc)
    
    # Sumation factors
    sF1 = derivD/(1+ytm*m)
    sF2 = m/((1+ytm*m)**2)*D
    
    return sF1 - sF2

# Function to compute convexity of bond price
def bond_convexity(ytm: float, settle_date: dt.date, mty_date: dt.date, coup_freq: int, 
            par_value: float, coupon_rate: float, ybc: int) -> float:
    # Bond price and cashflow table
    price, df_specs = bond_price(settle_date, mty_date, coup_freq, par_value,
                                 coupon_rate, ytm, ybc)
    # Macaulay duration
    b_dur = bond_risk_dur(df_specs, coup_freq/ybc)
    
    # Modified duration
    modD = bond_risk_mdur(b_dur, ytm, coup_freq/ybc)
    
    # Modified duration 1st derivative
    derivmodD = deriv_modD(ytm, settle_date, mty_date, coup_freq, par_value, coupon_rate, ybc)
    
    return (modD**2-derivmodD)

# Function to compute convexity of bond price by approximation
def bond_convexity_approx(ytm: float, settle_date: dt.date, mty_date: dt.date, coup_freq: int, 
            par_value: float, coupon_rate: float, ybc: int) -> float:
    # 1bp yield change
    d = 0.0001
    
    # Bond price 
    V0 = B(100*ytm, settle_date, mty_date, coup_freq, par_value, coupon_rate, ybc)
    
    # Bond price +1/-1 bp
    Vplus = B(100*(ytm+d), settle_date, mty_date, coup_freq, par_value, coupon_rate, ybc)
    Vminus = B(100*(ytm-d), settle_date, mty_date, coup_freq, par_value, coupon_rate, ybc)
    
    return (Vplus+Vminus-2*V0)/(V0*d**2)

# Function to compute second derivative of bond price wrt yield
def deriv2_B(ytm: float, settle_date: dt.date, mty_date: dt.date, coup_freq: int, 
            par_value: float, coupon_rate: float, ybc: int) -> float:
    # Bond price and cashflow table
    price, df_specs = bond_price(settle_date, mty_date, coup_freq, par_value,
                                 coupon_rate, ytm, ybc)
    # Macaulay duration
    b_dur = bond_risk_dur(df_specs, coup_freq/ybc)
    
    # Modified duration
    modD = bond_risk_mdur(b_dur, ytm, coup_freq/ybc)
    
    # Modified duration 1st derivative
    derivmodD = deriv_modD(ytm, settle_date, mty_date, coup_freq, par_value, coupon_rate, ybc)
    
    return price*(modD**2-derivmodD)



# Lets assert the first derivative is well coded/defined
x = symbols('x')
f,_ = bond_price(settle_date, mty_date, coup_freq, vn, coupon_rate, x, yearbase_conv)
f = B(x, settle_date, mty_date, coup_freq, vn, coupon_rate, yearbase_conv)
f_prime = diff(f,x)
def f_prime2(x): return deriv_B(x, settle_date, mty_date, coup_freq, vn, coupon_rate, yearbase_conv)
f_prime.evalf(subs={x:9.55})/100 - f_prime2(0.0955)*0.0001 # checked

###############################################################################

# Function to compute second order approx of bond pct change wrt changes in ytm
def delta_pctB(dy: float = 0.0001, modD: float = 5.29, C: float = 35) -> float:
    """
    Calculate bond percent change wrt changes in the ytm.

    Args:
    - dy (float): Yield to maturity change.
    - modD (float): Bond's modified duration.
    - C (float): Bond's convexity.

    Returns:
        (float) Bond's percent change after a change in the YTM.
    """
    
    return C/2*(dy**2)-modD*dy

"""
Lets compute the price change in the MBONO May31 after a 200bp rally
"""
dy = -200*0.0001
C = bond_convexity(ytm, settle_date, mty_date, coup_freq, vn, coupon_rate, yearbase_conv)
pctChgB = delta_pctB(dy, m31_mdur, C)
chgB = m31_price*pctChgB
realPriceChg = B(100*(ytm+dy),settle_date,mty_date,coup_freq,vn,coupon_rate,yearbase_conv)
print('\nM31 Price after 200bp rally:'+\
      f'\n\t Price: {realPriceChg:,.4f}'+\
      f'\n\t2nd Ord Approx: {m31_price+chgB:,.4f}')
###############################################################################
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
"""
Lets compute viz the MBONO May31's Convexity
"""
# Price-Yield Curve
dyrange = np.arange(-400,401,1)*0.0001
yrange = ytm + dyrange
B_yrange = []
for y in yrange:
    B_yrange.append(B(100*y, settle_date, mty_date, coup_freq, par_value,coupon_rate, yearbase_conv))
m31_price_ytm_df = pd.DataFrame({'YTM':yrange, 'Price': B_yrange})

# 1st Order Approximation
B_yrange_approx = m31_price-dyrange*m31_mdur*m31_price

# 2nd Order Approximation
B_yrange_approx2 = m31_price + m31_price*(C/2*dyrange**2 - m31_mdur*dyrange)

# M31 Price-Yield Curve
ax = m31_price_ytm_df.plot(x='YTM', y='Price',title='M31 Price-YTM')#, marker='.', mfc='w')
ax.plot(m31_price_ytm_df.YTM, B_yrange_approx, linestyle='--')
ax.plot(m31_price_ytm_df.YTM, B_yrange_approx2, linestyle='--')
ax.legend(['Price', '1st Ord Approx', '2nd Ord Approx'])
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1%}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel("YTM")
ax.set_ylabel("Price")
plt.tight_layout()
plt.show()
###############################################################################
"""
Lets get each MBONO Convexity
"""
filedate = '20231205'
df_mbono = data_from_valmer_MBONO(fdate=filedate)
df_mbono[['P', 'modD', 'C']] = 0
for i,r in df_mbono.iterrows():
    B0 = B(r['YTM'], settle_date, r['Maturity'], 182, 100, r['Coupon']/100, yearbase_conv)
    D0 = -deriv_B(r['YTM']/100, settle_date,  r['Maturity'], 182, 100,  r['Coupon']/100, yearbase_conv)/B0
    C = bond_convexity(r['YTM']/100, settle_date,  r['Maturity'], 182, 100,  r['Coupon']/100, yearbase_conv)
    df_mbono.loc[i,['P', 'modD', 'C']] = B0,D0,C
    
# Lets add expected price change over a 100bp rally
dY = -0.02
df_mbono['chgP'] = df_mbono.apply(lambda x: 
                                  (dY**2*x['C']/2-x['modD']*dY),
                                  axis=1)
df_mbono['chgP2'] = df_mbono.apply(lambda x: 
                                  (dY**2*x['C']/2-x['modD']*(-dY)),
                                  axis=1)
    
    
# CETEs
df_cete = data_from_valmer_CETE(fdate=filedate)
df_cete['DTM'] = (df_cete['Mty'] - settle_date).apply(lambda x: x.days)
df_cete[['P','modD', 'C']] = 0
for i,r in df_cete.iterrows():
    Z0 = zcb_price(r = r['YTM']/100, dtm = r['DTM'])
    Zp = zcb_price(r = (r['YTM']+0.01)/100, dtm = r['DTM'])
    Zm = zcb_price(r = (r['YTM']-0.01)/100, dtm = r['DTM'])
    D0 = (r['DTM']/360)/(1+r['YTM']*r['DTM']/36000)
    C = (Zp+Zm-2*Z0)/(Z0*0.0001**2)
    df_cete.loc[i,['P', 'modD', 'C']] = Z0,D0,C


#%% RATES TERM STRUCTURE

# import historic data
tmppath = r'H:\Python\KaxaNuk\FixedIncome'
db_cetes = pd.read_excel(tmppath+r'\HistoricoDeuda.xlsx', sheet_name='Cetes')
db_mbonos = pd.read_excel(tmppath+r'\HistoricoDeuda.xlsx', sheet_name='Bonos')

# Function to extract maturity data from bond serial name
def bond_mty_from_name(serial_name: str) -> dt.datetime:
    """
    Parse bond maturity date from string serial name data.

    Args:
    - serial_name (str): Bond serial name from ticker/name/name_id column data.

    Returns:
        (dt.datetime) Bond maturity date.
    """
    # Get maturity string
    str_mty = serial_name[-6:]
    # Maturity in datetime type
    T = dt.datetime(2000+int(str_mty[:2]), int(str_mty[2:4]), int(str_mty[-2:]))
    return T

# Set dates
dt_date1, dt_date2 = dt.datetime(2023,10,2), dt.datetime(2023,9,4)

# ZCBs
cond_date1 = (db_cetes['dteDate'] == dt_date1)
cond_date2 = (db_cetes['dteDate'] == dt_date2)
df_zcb = db_cetes[cond_date1].\
    sort_values('DTM')[['DTM', 'YTM', 'txtInstrumento']].\
    set_index('txtInstrumento').merge(
        db_cetes[cond_date2][['DTM', 'YTM', 'txtInstrumento']],
        how='inner', left_on='txtInstrumento', right_on='txtInstrumento')
# CETE Yield Curve
ax = df_zcb.set_index('DTM_x')[['YTM_x','YTM_y']].plot(title='CETE Curve', marker='o', mfc='w')
ax.legend([dt_date1.strftime("%d-%b-%Y"), dt_date2.strftime("%d-%b-%Y")])
ax.set_xlabel("Days to Maturity (DTM)")
ax.set_ylabel("Rate(%)")
plt.tight_layout()
plt.show()

# Ms
cond_date1 = (db_mbonos['dteDate'] == dt_date1)
cond_date2 = (db_mbonos['dteDate'] == dt_date2)
df_M = db_mbonos[cond_date1].\
    sort_values('DTM')[['DTM', 'YTM', 'txtInstrumento']].\
    set_index('txtInstrumento').merge(
        db_mbonos[cond_date2][['DTM', 'YTM', 'txtInstrumento']],
        how='inner', left_on='txtInstrumento', right_on='txtInstrumento').\
        drop(range(4)).reset_index(drop=True)
# MBONO Yield Curve
ax = df_M.set_index('DTM_x')[['YTM_x','YTM_y']].plot(title='MBONO Curve', marker='o', mfc='w')
ax.legend([dt_date1.strftime("%d-%b-%Y"), dt_date2.strftime("%d-%b-%Y")])
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel("Days to Maturity (DTM)")
ax.set_ylabel("Rate(%)")
plt.tight_layout()
plt.show()

# MXN Govt YC
sel_cete = ['BI_CETES_231101', 'BI_CETES_231130', 'BI_CETES_231228', 
            'BI_CETES_240404', 'BI_CETES_240627', 'BI_CETES_241003', 
            'BI_CETES_250320']
cond_isSelCete = db_cetes['txtInstrumento'].isin(sel_cete)
cond_isCeteInDt = db_cetes['dteDate'] == dt_date1 #dt_date2
cond_isMinDt = db_mbonos['dteDate'] == dt_date1 #dt_date2
df_crv_mx = pd.concat([db_cetes[cond_isCeteInDt*cond_isSelCete].sort_values('DTM'),
           db_mbonos[cond_isMinDt].sort_values('DTM').reset_index(drop=True).drop(range(4))]).reset_index(drop=True)
df_crv_mx['class'] = 'M'
df_crv_mx['class'][df_crv_mx['txtInstrumento'].apply(lambda x: x[:2]) == 'BI'] = 'Z'

# plot
tmpdf = df_crv_mx.copy()
tmprowZ = tmpdf[tmpdf['class'] == 'M'].iloc[0]; tmprowZ['class'] = 'Z'
tmprowM = tmpdf[tmpdf['class'] == 'Z'].iloc[-1]; tmprowM['class'] = 'M'
tmpdf = tmpdf.append(tmprowM).append(tmprowZ).sort_values('DTM')
grps = tmpdf.groupby('class')
for name, group in grps:
    ax = group.set_index('DTM')['YTM'].plot(marker='o', mfc='w')
plt.title(f'MXN Govt Yield Curve\n{dt_date2.strftime("%d-%b-%Y")}')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel("Days to Maturity (DTM)")
ax.set_ylabel("Rate(%)")
ax.legend(['MBONOs', 'CETEs'])
plt.tight_layout()
plt.show()

#%% BOOTSTRAPPING
from scipy.optimize import minimize
# Market rates
crv_mkt = df_crv_mx.drop(range(9,21))[['txtInstrumento','CPA','DTM','YTM','class']]
crv_mkt = crv_mkt.append(db_cetes[['txtInstrumento','DTM','YTM',]].iloc[-1]).sort_values('DTM').reset_index(drop=True)
crv_mkt.loc[7,'class'] = 'Z'
crv_mkt = crv_mkt.drop([4,6]).reset_index(drop=True)
# plot
tmpdf = crv_mkt.copy()
tmprowZ = tmpdf[tmpdf['class'] == 'M'].iloc[0]; tmprowZ['class'] = 'Z'
tmprowM = tmpdf[tmpdf['class'] == 'Z'].iloc[-1]; tmprowM['class'] = 'M'
tmpdf = tmpdf.append(tmprowM).append(tmprowZ).sort_values('DTM')
grps = tmpdf.groupby('class')
for name, group in grps:
    ax = group.set_index('DTM')['YTM'].plot(marker='o', mfc='w')
plt.title(f'MXN Govt Yield Curve\n{dt_date2.strftime("%d-%b-%Y")}')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel("Days to Maturity (DTM)")
ax.set_ylabel("Rate(%)")
ax.legend(['MBONOs', 'CETEs'])
plt.tight_layout()
plt.show()

###############################################################################
# Zero Crv Init
crv_Z = crv_mkt[crv_mkt['class'] == 'Z'][['DTM','YTM']].rename(columns={'YTM':'Z'})
###############################################################################
# M mar26 bootstrap
mar26_price, df_mar26 = bond_price(dt.date(2023,10,2), dt.date(2026,3,5), 182, 100, 5.75/100, 10.52/100, 182/360)
df_mar26['Z'] = np.interp(df_mar26['dtm'].to_numpy(),crv_Z['DTM'],crv_Z['Z'])
df_mar26['PV_CF_Z'] = df_mar26['CF']/(1+df_mar26['Z']*df_mar26['dtm']/36000)
def opt_f_mar26(r):
    df_mar26['Z'].iloc[-1] = r
    df_mar26['PV_CF_Z'] = df_mar26['CF']/(1+df_mar26['Z']*df_mar26['dtm']/36000)
    delta = 1e4*(mar26_price - df_mar26['PV_CF_Z'].sum())**2
    return delta
opt_res_mar26 = minimize(opt_f_mar26, df_mar26['Z'].iloc[-1], method='BFGS')
###############################################################################
# Zero Crv Update
crv_Z = crv_Z.append(df_mar26[['dtm','Z']].iloc[-1].rename({'dtm':'DTM'}))
###############################################################################
# M sep26 boostrap
sep26_price, df_sep26 = bond_price(dt.date(2023,10,2), dt.date(2026,9,3), 182, 100, 7/100, 10.535/100, 182/360)
df_sep26['Z'] = np.interp(df_sep26['dtm'].to_numpy(),crv_Z['DTM'],crv_Z['Z'])
df_sep26['PV_CF_Z'] = df_sep26['CF']/(1+df_sep26['Z']*df_sep26['dtm']/36000)
def opt_f_sep26(r):
    df_sep26['Z'].iloc[-1] = r
    df_sep26['PV_CF_Z'] = df_sep26['CF']/(1+df_sep26['Z']*df_sep26['dtm']/36000)
    delta = 1e4*(sep26_price - df_sep26['PV_CF_Z'].sum())**2
    return delta
opt_res_sep26 = minimize(opt_f_sep26, df_sep26['Z'].iloc[-1], method='BFGS')
###############################################################################
# Zero Crv Update
crv_Z = crv_Z.append(df_sep26[['dtm','Z']].iloc[-1].rename({'dtm':'DTM'}))
crv_Z = crv_Z.append(pd.DataFrame({'DTM':1,'Z':11.25}, index=range(1)),ignore_index=True)
crv_Z = crv_Z.sort_values('DTM').reset_index(drop=True)
###############################################################################
# Zero Curve Smoothing
zero_rates = np.interp(np.arange(1,crv_Z.iloc[-1]['DTM']+1,1), 
                       crv_Z['DTM'],crv_Z['Z'])
disc_factors = pd.DataFrame(1/(1+zero_rates*np.arange(1,crv_Z.iloc[-1]['DTM']+1,1)/36000), columns=['df1'])
disc_factors['df2'] = disc_factors.shift()
disc_factors.loc[0,'df2'] = 1
fwd_rates = 36000*(disc_factors['df2']/disc_factors['df1']-1)
df_z_rates = pd.DataFrame(zero_rates, index=np.arange(1,crv_Z.iloc[-1]['DTM']+1,1))
df_fwd_rates = pd.DataFrame(fwd_rates.values, index=np.arange(1,crv_Z.iloc[-1]['DTM']+1,1))
# Zero Curve Plot
ax = df_z_rates.plot(title='Zero Rates Curve', legend=False)
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel("Days to Maturity (DTM)")
ax.set_ylabel("Rate(%)")
plt.tight_layout()
plt.show()
# Fwd Curve Plot
ax = df_fwd_rates.plot(title='Forward Rates Curve', legend=False)
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel("Days to Maturity (DTM)")
ax.set_ylabel("Rate(%)")
plt.tight_layout()
plt.show()

#%%############################################################################
# QUANTLIB
import QuantLib as ql

trade_date = dt.datetime(2023,10,10)
ql_valdate = ql.Date(trade_date.day, trade_date.month, trade_date.year)
ql.Settings.instance().evaluationDate = ql_valdate

# Function to parse datetime into ql.Date
def parse_datetime_2_qlDate(date: dt.datetime) -> ql.Date:
    """
    Parse bond maturity date from datetime to ql.Date type.

    Args:
    - date (dt.datetime): Date to parse.

    Returns:
        (ql.Date) Date as ql.Date data type.
    """
    # Extract date atoms
    day, month, yr = date.day, date.month, date.year
    return ql.Date(day, month, yr)

# MBONO Curve
df_crv_m = db_mbonos[db_mbonos['dteDate'] == dt_date1][
    ['txtInstrumento','DTM','YTM','CPA']].reset_index(drop=True).sort_values('DTM')
df_crv_m['MTY'] = df_crv_m['txtInstrumento'].apply(bond_mty_from_name)
df_crv_m['MTY_ql'] = df_crv_m['MTY'].apply(parse_datetime_2_qlDate)

spotDates = df_crv_m['MTY_ql'].tolist()
spotDates.insert(0, ql_valdate)
spotRates = (df_crv_m['YTM']/100).tolist()
spotRates.insert(0, 11.25/100)

dayCount = ql.Actual360()
calendar = ql.Mexico()
interpolation = ql.Linear()
compType= ql.Compounded
compFreq = ql.Annual #ql.OtherFrequency

# M31
issueDate = ql.Date(23,6,2011)
maturityDate = df_crv_m[df_crv_m['MTY']=='2031-05-29']['MTY_ql'].values[0]
tenor = ql.Period('26W')
calendar = ql.Mexico()
bussinessConvention = ql.Following
dateGeneration = ql.DateGeneration.Backward
monthEnd = False
# Bond pmt schedule
schedule = ql.Schedule(issueDate, maturityDate, tenor, calendar, 
                       bussinessConvention, bussinessConvention, 
                       dateGeneration, monthEnd)
list(schedule)
dayCount = ql.Actual360()
couponRate = df_crv_m[df_crv_m['MTY']=='2031-05-29']['CPA'].to_numpy()[0]/100
coupons = [couponRate]

settlementDays = 2
faceValue = 100
fixedRateBond = ql.FixedRateBond(settlementDays, faceValue, schedule, coupons, dayCount)

# Use this curve for pricing via bootstrapped zero curve
spotCurve = ql.ZeroCurve(spotDates, spotRates, dayCount, calendar, 
                         interpolation, compType)
spotCurveHandle = ql.YieldTermStructureHandle(spotCurve)

# Use this workaround curve for pricing with the YTM
#flatCrv = ql.FlatForward(ql.Date(12,10,2023), ql.QuoteHandle(ql.SimpleQuote(ytm)), dayCount, compType, compFreq)
flatCrv = ql.FlatForward(2, ql.Mexico(), ql.QuoteHandle(ql.SimpleQuote(ytm)), dayCount, 1, 2)
ytm_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(flatCrv))
bondEngine = ql.DiscountingBondEngine(spotCurveHandle)
fixedRateBond.setPricingEngine(ytm_engine)

print(fixedRateBond.NPV())
print(fixedRateBond.dirtyPrice())
print(fixedRateBond.cleanPrice())
print(fixedRateBond.accruedAmount())
print(fixedRateBond.dayCounter())
print(fixedRateBond.settlementDate())

# Price with -50bp shift
dv01 = ql.BondFunctions.basisPointValue(fixedRateBond, ql.InterestRate(ytm, dayCount, 1,2))
P0 = fixedRateBond.dirtyPrice()
D = ql.BondFunctions.duration(fixedRateBond, ql.InterestRate(ytm, dayCount, 1,2), ql.Duration.Macaulay)
modD = ql.BondFunctions.duration(fixedRateBond, ql.InterestRate(ytm, dayCount, 1,2))/1
C = ql.BondFunctions.convexity(fixedRateBond, ql.InterestRate(ytm, dayCount, 1,2))
dr = 0.0015
deltaPx = P0*(C/2*(dr**2) - modD*(-dr)) # deltaPx = P0*(dr**2)*C/2 - dv01*dr/0.0001
P1 = P0 + deltaPx
np.round(P1,6), np.round(fixedRateBond.dirtyPrice(ytm-dr,dayCount, 1, 2),6)

for c in fixedRateBond.cashflows():
    print('%20s %12f' % (c.date(), c.amount()))
    
pd.DataFrame([(ql.as_coupon(c).accrualStartDate(), ql.as_coupon(c).accrualEndDate())
              for c in fixedRateBond.cashflows()[:-1]],
             columns = ('start','end'), index = range(1,len(fixedRateBond.cashflows()))
             )

## MANUALLY
ytm = df_crv_m[df_crv_m['MTY']=='2031-05-29']['YTM'].to_numpy()[0]/100
# M31 Price & CF table
m31_price, df_m31 = bond_price(settle_date, mty_date, coup_freq, vn, 
                               coupon_rate, ytm, yearbase_conv)
# accrued interest
m31_accInt = (182-df_m31.iloc[0]['dtm'])/360*coupon_rate*par_value
m31_price_cln = m31_price - m31_accInt
print(f'The price of the bond is: ${m31_price: .6f}')
print(f'Bond clean price: ${m31_price_cln: .6f}')
print(f'Bond accrued interest: ${m31_accInt: .6f}')
print(f"M31 Specs:\n{df_m31}")

#%% 1a. Compound Interest Calculation
def compound_interest(P: float, r: float, n: int) -> dict:
    
    results = {}
    
    """
    Calculate and print the capital plus interest at the end of each year.

    Args:
    - P (float): Initial capital.
    - r (float): Interest rate per period (as a decimal, for example, 0.05 for 5%).
    - n (int): Number of years.

    Returns:
    None
    """
    for i in range(1, n+1):
        A = P * (1 + r) ** i
        print(f"Year {i}: Capital + Interest = {A:.2f}")
        results['Year ' + str(i)] = A
        
    return results

# Example:
comp_interest = compound_interest(1000, 0.05, 3)

#%% 1b. Net Present Value Calculation
def calculate_npv(r: float, C0: float, cashflows: list) -> float:
    
    """
    Calculate and print the Net Present Value and determine if a project is profitable.

    Args:
    - r (float): Discount rate (as a decimal, for example, 0.05 for 5%).
    - C0 (float): Initial investment.
    - cashflows (list of float): List of cash flows per period.

    Returns:
    None
    """
    NPV = -C0
    for i, CF in enumerate(cashflows, 1):
        
        NPV += CF / (1 + r) ** i

    print(f"NPV = {NPV:.2f}")
    if NPV > 0:
        print("The project is profitable.")
    else:
        print("The project is not profitable.")
    
    return NPV

# Example:
npv = calculate_npv(0.05, 10000, [3000, 4000, 5000])


