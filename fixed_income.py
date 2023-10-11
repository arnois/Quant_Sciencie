#%% MODULES
import numpy as np
import pandas as pd
import datetime as dt
import holidays
#%% Interest Rates

# Function to measure interest 
def calc_interest(VT: float = 100, V0: float = 96, t: float = 0.5, im: str = 'eff') -> float:
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
def calc_interest_rate(VT: float = 100, V0: float = 96, t: float = 0.5, im: str = 'eff') -> float:
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
def accum_f(i: float = 0.04, n: float = 1, itype: str = 'simple', m: float = 1) -> float:
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

#%% BOND PRICING
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
def bond_price(settle_date: dt.date, mty_date: dt.date, n_coup_days: int, 
               par_value: float, coupon_rate: float, ytm: float, 
               m: float, yearbase_conv: int = 360) -> tuple:
    """
    Calculate bond price.

    Args:
    - settle_date (dt.date): Settlement date.
    - mty_date (dt.date): Maturity date.
    - n_coup_days (int): Frequency of coupon payment in days.
    - par_value (float): Nominal or face value.
    - coupon_rate (float): Coupon rate
    - ytm (float): Yield to maturity.
    - m (float): Compounding frequency.
    - yearbase_conv (int): Days in a year convention for periods.

    Returns:
        (tuple) Bond price with cashflows characteristics table.
    """
    # Maturity in days
    T = (mty_date - settle_date).days
    
    # Coupons left
    n_coup_left = np.ceil(T/n_coup_days).astype(int)

    # Payment dates
    coup_freq = int(yearbase_conv*m)
    CF_dates = pd.bdate_range(start=mty_date, periods=n_coup_left, 
                              freq='-'+str(coup_freq)+'D', 
                              holidays=holidays.MX()).sort_values()

    # Coupon payments
    c_pmt = coupon_rate*par_value*m
    bond_CF = np.repeat(c_pmt, n_coup_left)
    bond_CF[-1] += par_value

    # Discount Factors
    bond_CF_dtm = np.array([x.days for x in (CF_dates - settle_date)])
    bond_CF_period = bond_CF_dtm/yearbase_conv
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
# M31 characteristics
yearbase_conv = 360
settle_date = dt.datetime(2023,10,12)
mty_date = dt.datetime(2031,5,29)
T = (mty_date - settle_date).days
coup_freq = 182
n_coup_left = np.ceil(T/coup_freq).astype(int)
cf_dates = pd.bdate_range(start=mty_date, periods=n_coup_left, 
                          freq='-'+str(coup_freq)+'D', 
                          holidays=holidays.MX()).sort_values()
vn = 100
coupon_rate = 7.75/100
comp_conv = coup_freq/yearbase_conv
ytm = 9.90/100

# M31 Price & CF table
m31_price, df_m31 = bond_price(settle_date, mty_date, coup_freq, vn, 
                               coupon_rate, ytm, comp_conv, yearbase_conv)
# accrued interest
m31_accInt = (182-df_m31.iloc[0]['dtm'])/360*coupon_rate*par_value
m31_price_cln = m31_price - m31_accInt
print(f'The price of the bond is: ${m31_price: .6f}')
print(f'Bond clean price: ${m31_price_cln: .6f}')
print(f'Bond accrued interest: ${m31_accInt: .6f}')
print(f"M31 Specs:\n{df_m31}")
###############################################################################

#%% BOND RISK MEASURES
# Function to compute bond duration
def bond_risk_dur(df_bond_specs: pd.DataFrame, m: float) -> float:
    """
    Calculate bond duration.

    Args:
    - df_bond_specs (pd.DataFrame): Bond specs with the periods, cashflows and
        discounting factors data.
    - m (float): Compounding frequency.

    Returns:
        (float) Bond duration.
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
def bond_risk_dv01_approx(df_bond_specs: pd.DataFrame, y: float, m: float) -> float:
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
    # DV01 via ModDur
    dv01_approx = P*mdur/10000
    
    return -1*dv01_approx

"""
Lets compute the MBONO May31 Risk Measures.
"""
# Duration
m31_dur = bond_risk_dur(df_m31, comp_conv)
# Modified Duration
m31_mdur = bond_risk_mdur(m31_dur, ytm, comp_conv)
# DV01 via ModDur
m31_dv01_mdur = bond_risk_dv01_approx(df_m31, ytm, comp_conv)
df_m31_risks = pd.DataFrame({'Measure': [m31_dur, m31_mdur, m31_dv01_mdur]}, index = ['Dur', 'ModDur', 'DV01'])
print('\nM31 Risk Measures\n')
print(df_m31_risks)

"""
Lets compute the MBONO May31 DV01 Measure by def.
"""

# Function to compute DV01 risk by def
def bond_risk_dv01(settle_date: dt.date, mty_date: dt.date, n_coup_days: int, 
               par_value: float, coupon_rate: float, ytm: float, m: float) -> tuple:
    """
    Calculate bond DV01.

    Args:
    - settle_date (dt.date): Settlement date.
    - mty_date (dt.date): Maturity date.
    - n_coup_days (int): Frequency of coupon payment in days.
    - par_value (float): Nominal or face value.
    - coupon_rate (float): Coupon rate
    - ytm (float): Yield to maturity.
    - m (float): Compounding frequency.

    Returns:
        (tuple) Bond DV01 risk.
    """
    # Bond price
    P, df_bond_specs = bond_price(settle_date, mty_date, n_coup_days, 
                                  par_value, coupon_rate, ytm, m)
    # Bond price wrt +1bp
    P_plus1bp, df_bond_specs_plus1bp = bond_price(settle_date, mty_date, 
                                                  n_coup_days, par_value, 
                                                  coupon_rate, ytm+0.0001, m)
    # Bond price wrt -1bp
    P_minus1bp, df_bond_specs_minus1bp = bond_price(settle_date, mty_date, 
                                                    n_coup_days, par_value, 
                                                    coupon_rate, ytm-0.0001, m)
    # Bond DV01
    bond_dv01 = -1*np.mean(abs(np.array([P_plus1bp-P, P_minus1bp-P])))
    
    return bond_dv01


# M31 DV01
m31_dv01 = bond_risk_dv01(dt.datetime(2023,10,12), dt.datetime(2031,5,29), 
                           182, 100, 7.75/100, 9.90/100, 182/360)
print(f'M31 DV01 Risk: {m31_dv01: .6f}')

#%% RATES TERM STRUCTURE
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# import historic data
tmppath = r'H:\Python\KaxaNuk\FixedIncome'
db_cetes = pd.read_excel(tmppath+r'\HistoricoDeuda.xlsx', sheet_name='Cetes')
db_mbonos = pd.read_excel(tmppath+r'\HistoricoDeuda.xlsx', sheet_name='Bonos')

# Function to extract matiruty data from bond serial name
def bond_mty_from_name():
    return

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
# If we define it as a function:
def pv(pmt, years, r, fv = 0):
    '''
    Like the BAII Plus Financial calculator, 
    computes the present value of a series of cashflows
    '''
    
    cashflows = np.repeat(pmt, years)
    cashflows[-1] += fv
    d_factor = (np.ones(years) + r).cumprod()
    pv_i = cashflows/d_factor
    return -np.sum(pv_i, axis = 0)

pv(pmt = 5, years = 20, r = 0.06, fv = 100)

pv(pmt = 65, years = 20, r = 0.06, fv = 1000)


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

#%% 1c. Calculation of CETE Price
def cete_price(r: float, t: int):
    
    """
    Calculate the price of a CETE.

    Args:
    - r (float): Yield rate (as a decimal, for example, 0.05 for 5%).
    - t (int): CETE term in days.

    Returns:
    float: CETE price rounded to 6 decimal places.
    """
    P = 10/(1 + (r * t / 360))
    return round(P, 6)

# Example:
print(cete_price(0.1125, 182))
