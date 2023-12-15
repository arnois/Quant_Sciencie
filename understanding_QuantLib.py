# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 13:56:12 2022

Getting familiar with QuantLib

@author: jquintero
"""
# installation command: pip install QuantLib
# load module
import QuantLib as ql
import pandas as pd

############################################################################
# BASICS
############################################################################

# Date instances
dt1, dt2 = ql.Date(24,6,2021), ql.Date(1,4,2021)

# Date operations
dt1 + ql.Period(28,ql.Days)
dt2 - ql.Period(28,ql.Days)

# Calendar instances
mx_cal = ql.Mexico()
us_cal = ql.UnitedStates()
# joint calendar
mxus_cal = ql.JointCalendar(mx_cal, us_cal)

# Date operations based on Calendar instance
period = ql.Period(28*6,ql.Days)
mx_fwdDate = mx_cal.advance(dt1, period)
us_fwdDate = us_cal.advance(dt1, period)
mxus_fwdDate = mxus_cal.advance(dt1, period)
print(mx_fwdDate,us_fwdDate,mxus_fwdDate)
# business days between dates based on Calendar instance
days_mxcal = mx_cal.businessDaysBetween(dt1,dt2); print(days_mxcal)
days_uscal = us_cal.businessDaysBetween(dt1,dt2); print(days_uscal)

# Schedule Class
# udf to build a tiie swap's coupons schedule
def schdl_irs_tiie(eff_dt = ql.Date(6,4,2021), 
                   end_dt = ql.Date(16,9,2021),
                   irrCpnStart = True):
    """
    Parameters
    ----------
    eff_dt : QuantLib.Date, effective date of swap
        DESCRIPTION. The default is QuantLib.Date(6,4,2021).
    end_dt : QuantLib.Date, maturity date of swap
        DESCRIPTION. The default is QuantLib.Date(16,9,2021).
    irrCpnStart : bool, irregular coupon at start
        DESCRIPTION. The default is True.

    Returns
    -------
    QuantLib.Schedule, swap's coupons schedule

    """
    # class parameters for TIIE IRS cpn schedule
    tenor = ql.Period('28D') # period between coupons
    cal = ql.Mexico() # domestic crncy calendar
    busConv = ql.Following
    endBusConv = ql.Following
    if not irrCpnStart:
        date_gen = ql.DateGeneration.Forward # irregular cpn at mty
    else:
        date_gen = ql.DateGeneration.Backward # irregular cpn at start
    eom = False # do not enforce dates to be end of month if eff_dt is one
    # class instance
    schdl = ql.Schedule(eff_dt, end_dt,
                        tenor, cal,
                        busConv, endBusConv,
                        date_gen, eom)
    return(schdl)

# example of tiie irs schedule
list(schdl_irs_tiie(ql.Date(26,8,2022),
                         ql.Date(26,8,2022)+\
                             ql.Period(28*6,ql.Days)))
    
    
# Interest Rate Class
# instance
r = ql.InterestRate(0.0850,
                 ql.Actual360(),
                 ql.Simple,
                 ql.Annual)
# methods
r.compoundFactor(28*3/360), r.discountFactor(28*3/360)
r.equivalentRate(ql.Annual, ql.Monthly, 28*3/360).rate()



############################################################################
# PRICING
############################################################################

# global evaluation date
today = ql.Date(25, ql.August, 2022) # set today's day
ql.Settings.instance().evaluationDate = today






