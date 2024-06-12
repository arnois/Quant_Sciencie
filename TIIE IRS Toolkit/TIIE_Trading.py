# Code Description
"""
The objetive of this code is to quote real time TIIE-IRS.
"""
#%%############################################################################
# Libraries
###############################################################################
import os # portable way of using operating system dependent functionality
import glob
import QuantLib as ql
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
import sys # provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.
import warnings
warnings.filterwarnings("ignore")
str_cwd = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(str_cwd)
import xlwings as xw
import udf_corros as pr
import udf_TIIE_Trading as fn
import udf_TIIE_CurveCreate as udf
import udf_TIIE_PfolioMgmt as udfp
#print('Working in '+os.getcwd())

#%%############################################################################
# XL File
###############################################################################
def fetch_xlBook(str_file):
    xw.App.calculation = 'manual'
    wb = xw.Book(str_file)
    xl_sess_id = xw.apps.keys()
    for pid in xl_sess_id:
        try:
            wb = xw.apps[pid].books(str_file)
            xl_pid = pid 
            #print(f'{str_file} found in {pid} XL session')
            break
        except:
            #print(f'{str_file} not in {pid} XL session')
            continue
    wb_sheetnames = [wb.sheets(i).name for i in range(1,1+wb.sheets.count)]
    if 'shortEndPricing' not in wb_sheetnames:
        wb.sheets.add(name='shortEndPricing', before='Pricer')
        red = (252, 228, 214)
        wb.sheets('shortEndPricing').api.Tab.Color = xw.utils.rgb_to_int(red)
        wb.save()
    xw.App.calculation = 'automatic'
    return xl_pid, wb

def proc_EndSession():
    # Close XL Session
    isCloseGiven = input('End XL Session (Yes/No): ')[0].lower() == 'y'
    if isCloseGiven:
        wb_pricing.range(rng_pricing).color = None
        wb.save() 
        xw.apps[xl_id].api.Quit()
#%%############################################################################     
# Curves Consistency Check
###############################################################################
# TIIE28 Ibor Index
def main_ccheck():
    str_tiiefixings_file = r'C:\Users\jquintero\Downloads' + \
        r'\Consulta_20230313-132346367.xlsx'
    ibor_tiie = udf.set_ibor_TIIE(crvTIIE, str_tiiefixings_file)
    # Discounting Engine
    tiie_swp_engine = udfp.get_pricing_engine(crvMXNOIS)
    # Repricing Vaniall Swaps
    vanilla_tenors = dic_data['MXN_TIIE']['Period'][
        dic_data['MXN_TIIE']['Types']=='SWAP']*ql.Period(13)
    vanilla_swp_rates = {}
    for qlperiod in vanilla_tenors:
        input_rate = dic_data['MXN_TIIE']['Quotes'][
            dic_data['MXN_TIIE']['Tenor'] ==\
                '%'+str(int(qlperiod.length()/4))+'L']/100
        mty = ql_settle_dt+qlperiod
        vswp = udfp.tiieSwap(ql_settle_dt, mty, 1e6, ibor_tiie, 
                             rate=input_rate.values[0])
        vswp0 = vswp[0]
        vswp0.setPricingEngine(tiie_swp_engine)
        vanilla_swp_rates[mty.to_date()] = vswp0.fairRate()*100
    # Vanilla Swap Rates
    df_sancheck = pd.DataFrame(vanilla_swp_rates.values(), 
                 index=dic_data['MXN_TIIE']['Tenor'][
                     dic_data['MXN_TIIE']['Types']=='SWAP'])
    df_sancheck['input'] = dic_data['MXN_TIIE']['Quotes'][
        dic_data['MXN_TIIE']['Types']=='SWAP'].values
    df_sancheck.round(4)
    # Swap payment dates
    tmpfxnglst =  udfp.tiieSwap_FixingDates(ibor_tiie,
                                 settle_date,
                                 mty.to_date(),
                                 0)
    # Swap payment at date
    tmpcf = udfp.get_CF_tiieSwap(vswp0, tmpfxnglst, ql_eval_dt)
    print(tmpcf)
    # Swap FloatLeg Mgmt
    swp_leg1 = vswp0.leg(1)
    df_swpfltleg = pd.DataFrame({'start': 
                        pd.to_datetime(str(cf.accrualStartDate().ISO())), 
                  'mty': pd.to_datetime(str(cf.accrualEndDate().ISO())),
                  'TIIE28': cf.rate()} for cf in map(ql.as_coupon, swp_leg1))
    for i,r in df_swpfltleg.iterrows():
        idt = ql.Date(r['start'].day, r['start'].month, r['start'].year)
        fdt = ql.Date(r['mty'].day, r['mty'].month, r['mty'].year)
        df_swpfltleg.loc[i,'FwdRate'] = crvTIIE.forwardRate(idt, fdt, 
                                                            ql.Actual360(), 
                                                            ql.Simple).rate()
        df_swpfltleg.loc[i,'DF']  = crvMXNOIS.discount(fdt)
    df_swpfltleg['tau'] = (df_swpfltleg['mty'] - \
                           df_swpfltleg['start']).dt.days/360
    df_swpfltleg['w'] = df_swpfltleg['tau']*df_swpfltleg['DF']/\
        (df_swpfltleg['tau']*df_swpfltleg['DF']).sum()
    df_swpfltleg['wFR'] = df_swpfltleg['w']*df_swpfltleg['FwdRate']
#%%############################################################################
# Displaying TIIE Disc and Proj Curves
###############################################################################
def plot_curves_TIIE():
    crv_tiie_ois, crv_tiie = {}, {}
    t = ql_settle_dt
    for i in range(0, 390):
        crv_tiie_ois[t] = crvMXNOIS.forwardRate(t, t+ql.Period(28, ql.Days), 
                                                ql.Actual360(), 
                                                ql.Simple).rate()
        crv_tiie[t] = crvTIIE.forwardRate(t, t+ql.Period(28, ql.Days), 
                                                ql.Actual360(), 
                                                ql.Simple).rate()
        t += ql.Period(28, ql.Days)
    #crv_tiie_ois_df = pd.DataFrame(crv_tiie_ois, index=[0]).T
    #crv_tiie_df = pd.DataFrame(crv_tiie, index=[0]).T
    pd.Series(crv_tiie_ois.values(), 
              index = [s.to_date() 
                       for s in pd.Series(crv_tiie_ois).index]).plot()
    pd.Series(crv_tiie.values(), 
              index = [s.to_date() 
                       for s in pd.Series(crv_tiie).index]).plot()
    plt.tight_layout()
    plt.show()

# Function to pull MXNOIS bootstrapped curve in discount factors
def pull_curve_MXN(wb,strpath,str_type):
    # Read curve data
    tmpcrv = pd.read_excel(strpath)
    # DF
    tmpdf = (1/(1+tmpcrv['PLAZO']*tmpcrv['VALOR']/36000)).iloc[:10921].reset_index()
    tmpdf['index'] = tmpdf['index']+1
    # Print data
    wb_df = wb.sheets('DF')
    if str_type == 'OIS' :
        wb_df.range('B4').value = tmpdf.values
    elif str_type == 'TIIE':
        wb_df.range('G4').value = tmpdf.values
    else:
        print('Curve type wrong. Try again!')
    
    return None

#%%############################################################################
# Trading Blotter File
###############################################################################
# Function to adjust maturity dates for holidays
def dt_adj_hday(dates: list) -> list:
    """
    Parameters
    ----------
    dates : (list) List of datetime dates.

    Returns
    -------
    (list) List of ql.Date dates adjusted for holidays.
    """
    # Parse datetime to ql.Date
    lst_dt = [ql.Date(x.day, x.month, x.year) for x in dates]
    # Adjust for holidays
    for dt in lst_dt:
        if cal_mx.isHoliday(dt): 
            lst_dt[lst_dt.index(dt)] = cal_mx.advance(dt,1,ql.Days)
    
    return lst_dt

# Function to pull trading blotter data
def pull_blotters(path_trading_blotter):
    # Get filename
    lst_files = glob.glob(path_trading_blotter)
    # Get trading data from filename
    try:
        tmpdf1 = pd.read_excel(lst_files[0],'Blotter')
    except ValueError:
        print('No trades yet.')
        return None
    
    tmpdf1 = tmpdf1[tmpdf1['Book']==1814]
    # tmpdf1['Start_Date'] = tmpdf1['Start_Date'].fillna(settle_date)
    # Start Date Mgmt
    tmp_loc_stdt_na = tmpdf1['Start_Tenor'].isna()
    ## vanilla
    tmpdf1['Start_Date'][tmp_loc_stdt_na] = tmpdf1['Start_Date'].fillna(settle_date)
    ## fwd starting
    tmpdf1['Start_Date'][~tmp_loc_stdt_na] = (tmpdf1['Start_Tenor']
        [~tmp_loc_stdt_na]).apply(lambda x: settle_date+timedelta(days=28*x))
    # End Date Mgmt
    tmp_loc_edt_na = tmpdf1['End_Date'].isna()
    if tmp_loc_edt_na.any():
        tmpdf1['End_Date'][tmp_loc_edt_na] = (tmpdf1[tmp_loc_edt_na]).\
        apply(lambda x: timedelta(days=28*x['Fwd_Tenor']) + x['Start_Date'], 
              axis=1)
    # Start/End Date String Format
    tmpdf1[['Start_Date','End_Date']] = tmpdf1[['Start_Date','End_Date']].\
        apply(pd.to_datetime).astype(str)
    # Trading blotter fields
    flds = 'A1'+':'+wb.sheets('Blotter').range('A1').end('right').\
        address.replace('$','')[0]+'1'
    lst_flds = wb.sheets('Blotter').range(flds).value

    # Get last row
    is_data_there = wb.sheets('Blotter').range('A2').value == 1
    if is_data_there:
        last_row = wb.sheets('Blotter').range('A1').end('down').row
    else:
        last_row = 1
    
    # Check and separate own trades
    dfblttr = wb.sheets('Blotter').range('A1').\
        options(pd.DataFrame,header=1,index=True,expand='table').value
    tmplt = dfblttr.index.to_frame().apply(lambda x: x['Trade_#'][:2], axis=1)
    idx_ja = tmplt.to_numpy() == 'JA'
    tmpIDs = dfblttr['ID']
    dfblttr = dfblttr[idx_ja]
    
    # Clear contents
    wb.sheets('Blotter').range('A2:P100').clear_contents()
    
    # Data out
    tmpdf1 = tmpdf1.drop(['Book','Start_Tenor','Fwd_Tenor'],axis=1).fillna(0)
    start_row = last_row+1
    for f in tmpdf1.columns:
        if f in lst_flds:
            n = lst_flds.index(f)
            wb.sheets('Blotter').range('A1').offset(start_row-1,n).\
            value = tmpdf1[[f]].values
    
    # Paste back own trades
    wb.sheets('Blotter').range('A1').end('down').\
        offset(1,0).value = dfblttr.reset_index().to_numpy()
        
    wb.sheets('Blotter').range('M2').value = tmpIDs.to_numpy().reshape(-1,1)
            
    # Dates adjustment
    #lrow = wb.sheets('Blotter').range('E2').end('down').row
    #lst_dt = dt_adj_hday(wb.sheets('Blotter').range('E2:E'+str(lrow)).value)
    #wb.sheets('Blotter').range('E2').value = [[x.serialNumber()] for x in lst_dt]
    
    return None

# Function to pull EoD blotter data
def fetch_blotter(filepath = r'E:\Blotters'):
    date_val = evaluation_date
    # Dates mgmt
    ql_eval_dt = ql.Date(date_val.day, date_val.month, date_val.year)
    ql_settle_dt = cal_mx.advance(ql_eval_dt, 1, ql.Days)
    settle_date = date(ql_settle_dt.year(),
                       ql_settle_dt.month(),
                       ql_settle_dt.dayOfMonth())
    # Get trading/valuation date for file name
    str_month = str(date_val.month)
    str_day = str(date_val.day)
    if len(str_month)<=1:
        str_month = '0'+str_month
    if len(str_day)<=1:
        str_day = '0'+str_day
    str_name = str(date_val.year)[-2:]+str_month+str_day
    
    # Whole path to blotter
    str_filenamepath = filepath+'\\'+str_name+'.xlsx'
    
    # Read blotter file
    blttr = pd.read_excel(str_filenamepath, skiprows=2)
    
    # Filter 1814 swaps
    blttr = blttr.iloc[:,1:]
    blttr = blttr.set_index("#")
    blttr1814 = blttr[blttr['Book']==1814].drop(['User','Book'],axis=1)
    sel_cols = ['Tenor','Yield(Spot)','Size','Fecha Inicio',
                'Fecha vencimiento','Cuota compensatoria / unwind']
    blttr1814 = blttr1814[sel_cols]
    flag_van = blttr1814['Fecha Inicio'].isna()
    
    # Segregate vanilla from broken date swaps
    blttr1814_van = blttr1814[flag_van]
    blttr1814_bd = blttr1814[~flag_van].fillna(0)
    
    # Calculate start/end dates for vanilla swaps
    blttr1814_van['Fecha Inicio'] = settle_date
    s_edt = blttr1814_van['Tenor'].apply(lambda x: int(x[:-1])*28).\
        apply(lambda x: timedelta(days=x)+settle_date)
    blttr1814_van['Fecha vencimiento'] = s_edt
    blttr1814_van = blttr1814_van.fillna(0)
    
    # Output frame
    df_1814_van = blttr1814_van.copy()
    df_1814_bd = blttr1814_bd.copy()
    df_1814_van['Tenor'] = blttr1814_van['Tenor'].apply(lambda x: int(x[:-1]))
    df_1814_van['Yield(Spot)'] /= 100
    df_1814_van['Size'] *= 1e6
    
    df_1814_bd['Tenor'] = blttr1814_bd['Tenor'].apply(lambda x: int(x[:-1]))
    df_1814_bd['Yield(Spot)'] /= 100
    df_1814_bd['Size'] *= 1e6
    
    df_1814 = df_1814_van.append(df_1814_bd)
    
    # Get last row
    #is_data_there = wb.sheets('Blotter').range('A2').value == 1
    #if is_data_there:
    #    last_row = wb.sheets('Blotter').range('A1').end('down').row
    #else:
    #    last_row = 2
    
    # XL File PrintOut
    tmpc1 = ['Fecha Inicio', 'Fecha vencimiento', 'Size', 'Yield(Spot)']
    ## Swaps specs
    #wb.sheets('Blotter').range('D1').offset(last_row,0).value = ...
    wb.sheets('Blotter').range('D2').value = df_1814[tmpc1].values
    ## Swaps fees
    wb.sheets('Blotter').range('O2').value = df_1814[['Cuota compensatoria / unwind']].values
    
    return None
#%%############################################################################
# Evaluation date
###############################################################################
str_file = 'TIIE_IRS_Trading.xlsm'; str_file = 'TIIE_Trading.xlsx'
parameters = pd.read_excel(str_cwd+r'\\'+str_file, 'Pricer', header=None)
cal_mx = ql.Mexico()
evaluation_date = pd.to_datetime(parameters.iloc[0,1]) 
ql_eval_dt = ql.Date(evaluation_date.day,
                     evaluation_date.month, 
                     evaluation_date.year)
ql.Settings.instance().evaluationDate = ql_eval_dt
ql_settle_dt = cal_mx.advance(ql_eval_dt, 1, ql.Days)
settle_date = date(ql_settle_dt.year(), 
                 ql_settle_dt.month(), 
                 ql_settle_dt.dayOfMonth())

# exit
#%%############################################################################
# Global Inputs
###############################################################################
# Dealer market screening file
path_dmsf = r'\\tlaloc\cuantitativa\Fixed Income\TIIE IRS Valuation Tool\Arnua'
path_dmsf = r'\\tlaloc\cuantitativa\Fixed Income\Arnulf'
path_dmsf = r'U:\Fixed Income\Arnulf'
name_dmsf = 'Corros_v2.1.xlsx'
corros_file = path_dmsf + '\\' + name_dmsf

# Trading blotter path

path_trading_blotter =r'\\tlaloc\cuantitativa\Fixed Income\File Dump'+\
    r'\Blotters\TIIE\Desk Blotters'+r'\*J_'+ str(evaluation_date)[:4]+\
        str(evaluation_date)[5:7]+str(evaluation_date)[8:10]+'.xlsx'
# Markets
input_sheets = ['USD_OIS', 'USD_SOFR', 
                'USDMXN_XCCY_Basis', 'USDMXN_Fwds', 
                'MXN_TIIE']
# XL
clr_orange_output = (253, 233, 217)
xl_id, wb = fetch_xlBook(str_file)
wb_pricing = wb.sheets('Pricer')
wb_pxngfwds = wb.sheets('Fwds')
wb_tiie = wb.sheets('MXN_TIIE')
rng_pricing = 'H4:K417'
# Move to XL file path
os.chdir(str_cwd)
input_fxrate = wb_pricing.range('E1').value
# Fixings
banxico_TIIE28 = fn.get_fixings_TIIE28_banxico(evaluation_date)
# Holidays
hdays_mx = cal_mx.holidayList(ql_eval_dt-ql.Period(4*16,ql.Weeks),ql_eval_dt+ql.Period(30,ql.Years))
hdays_us = ql.UnitedStates(0).holidayList(ql_eval_dt-ql.Period(4*16,ql.Weeks),ql_eval_dt+ql.Period(30,ql.Years))
wb.sheets('holidays').range('A2').value = [[t.serialNumber()] for t in hdays_mx]
wb.sheets('holidays').range('B2').value = [[t.serialNumber()] for t in hdays_us]

# T-1 Curves
strpath_prevCrvs = r'\\tlaloc\cuantitativa\Fixed Income\File Dump\Historical OIS TIIE'
strprevdt = cal_mx.advance(ql_eval_dt, -1, ql.Days).to_date().strftime('%Y%m%d')
prevmxnois = r'\OIS_'+strprevdt+'.xlsx'
prevmxntiie = r'\TIIE_'+strprevdt+'.xlsx'
pull_curve_MXN(wb, strpath_prevCrvs+prevmxnois, 'OIS')
pull_curve_MXN(wb, strpath_prevCrvs+prevmxntiie, 'TIIE')

#%%############################################################################
# MAIN
###############################################################################
# fn.proc_CarryCalc(ibor_tiie, tiie_swp_engine, str_cwd, str_file, wb)
if __name__ == '__main__':
    # Closes
    fn.remate_closes(wb, evaluation_date)
    input('\nRemate closes ready. \nContinue? (Yes/No): ')
    wb.save()
    # Init
    str_cwd = str_cwd.replace('tiie irs toolkit', 'TIIE IRS Toolkit')
    dic_data = fn.proc_UpdateCurves_Inputs(input_sheets, str_cwd, str_file, wb)
    crvUSDOIS, crvUSDSOFR, crvMXNOIS, crvTIIE, \
        frCrvs, brCrvs, ibor_tiie, tiie_swp_engine = \
            fn.proc_BuildCurves(dic_data, banxico_TIIE28)
    user_option = 0
    # Trading
    while int(user_option) != 8:
        # Promtp
        user_option = int(fn.displayMenu(ql_eval_dt, ql_settle_dt))
        # User decision
        if user_option==1:
            # Pricer
            print('\nPricing... ')
            fn.proc_Pricing(ql_settle_dt, ibor_tiie, tiie_swp_engine, 
                         frCrvs, brCrvs, banxico_TIIE28, 
                         dic_data, str_cwd, str_file,
                         wb_pricing, rng_pricing)
            x = str(input('\nDisplay Cashflows? (Yes/No): ')[0]).lower()
            if x == 'y':
                dic_cf = fn.proc_displayCF(wb_pricing, ql_settle_dt, 
                                  ibor_tiie, tiie_swp_engine)
                df_cfOTR = pd.DataFrame()
                #col_tradeID = wb_pricing.range(
                #    'A3:A'+str(wb_pricing.range('A3').end('down').row)
                #    ).options(pd.DataFrame, 
                #              header=1, 
                #              index=False).value
                for k,v in dic_cf.items():
                    v['TradeID'] = k
                    v['Notional'] = wb_pricing.range('H'+str(int(k+3))).value
                    df_cfOTR = pd.concat([df_cfOTR, v]) # df_cfOTR.append(v)
                wb.sheets('CF1').range('J:S').clear_contents()
                df_cfOTR = df_cfOTR.sort_values(by='date')
                wb.sheets('CF1').range('J2').value = df_cfOTR.columns.to_list()
                wb.sheets('CF1').range('J3').value = df_cfOTR.values
                # pull cpn dates
                wb.sheets('CF1').range('W:X').clear_contents()
                wb.sheets('CF1').range('W2:X2').value = ['FltRate','DF']
                end_row = wb.sheets('CF1').range('J2').end('down').row
                df_StEndDt = pd.DataFrame(wb.sheets('CF1').range('K3:M'+str(end_row)).value,
                                          columns = wb.sheets('CF1').range('K2:M2').value)
                # Fwd & Disc Rates
                fwdRates = []
                discF = []
                for i,r in df_StEndDt.iterrows():
                    fixdt, start, end = r
                    fwdRates.append(ibor_tiie.fixing(ql.Date().from_date(fixdt)))
                    #fwdRates.append(crvTIIE.forwardRate(ql.Date().from_date(start), 
                    #                                    ql.Date().from_date(end),
                    #                                    ql.Actual360(),
                    #                                    ql.Simple).rate())
                    discF.append(crvMXNOIS.discount(ql.Date().from_date(end)))
                df_StEndDt['FltRate'] = fwdRates
                df_StEndDt['DF'] = discF
                # print fwd & disc rates
                wb.sheets('CF1').range('W3').value = df_StEndDt[['FltRate']].values
                wb.sheets('CF1').range('X3').value = df_StEndDt[['DF']].values
                
        elif user_option==2:
            # Implied Pricing
            print('\nShortEnd Pricing...')
            fn.proc_ShortEndPricing(crvMXNOIS, crvTIIE, wb)
            print('\tTenor Fwd TIIE28 Done!')
            fn.proc_ShortEndPricing_byMPC(crvTIIE, wb)
            print('\tMPC Date Fwd TIIE28 Done!')
        elif user_option==3:
            # Update Input
            print('\nUpdating feedstock...')
            dic_data = fn.proc_UpdateCurves_Inputs(input_sheets, 
                                                str_cwd, str_file, wb)
            # Update Curves
            print('Updating Curves...')
            #crvUSDOIS, crvUSDSOFR, crvMXNOIS, crvTIIE, \
            #    frCrvs, brCrvs, ibor_tiie, tiie_swp_engine = \
            #        fn.proc_UpdateCurves(dic_data, banxico_TIIE28)
            # Fast Track 2 TIIE Curve (35s)
            crvMXNOIS, crvTIIE, frCrvs, brCrvs, ibor_tiie, tiie_swp_engine = \
                    fn.proc_fastBuildCurves(dic_data, banxico_TIIE28, 
                                             crvUSDSOFR, crvUSDOIS)
            # crvTIIE, frCrvs, brCrvs = fn.proc_fastBuildCurves0(dic_data, ibor_tiie, crvUSDSOFR, crvUSDOIS, crvMXNOIS) (35s)

            print('Done!')
            z = str(input('\nUpdate Carry? (Yes/No): ')[0]).lower()
            if z == 'y':
                fn.proc_CarryCalc(ibor_tiie, tiie_swp_engine, 
                                  str_cwd, str_file, wb)                
        elif user_option==4:
            # Pxng Fwds
            print('\nPricing Fwds... ')
            fn.proc_Pricing(ql_settle_dt, ibor_tiie, tiie_swp_engine, 
                         frCrvs, brCrvs, banxico_TIIE28, 
                         dic_data, str_cwd, str_file,
                         wb_pxngfwds, rng_pricing)
            x = str(input('\nDisplay Cashflows? (Yes/No): ')[0]).lower()
            if x == 'y':
                dic_cf = fn.proc_displayCF(wb_pxngfwds, ql_settle_dt, 
                                  ibor_tiie, tiie_swp_engine)
                df_cfOTR = pd.DataFrame()
                for k,v in dic_cf.items():
                    v['TradeID'] = k
                    v['Notional'] = wb_pxngfwds.range('H'+str(int(k+3))).value
                    df_cfOTR = df_cfOTR.append(v)
                wb.sheets('CF1').range('J:S').clear_contents()
                df_cfOTR = df_cfOTR.sort_values(by='date')
                wb.sheets('CF1').range('J2').value = df_cfOTR.columns.to_list()
                wb.sheets('CF1').range('J3').value = df_cfOTR.values
                # pull cpn dates
                wb.sheets('CF1').range('W:X').clear_contents()
                wb.sheets('CF1').range('W2:X2').value = ['FltRate','DF']
                end_row = wb.sheets('CF1').range('J2').end('down').row
                df_StEndDt = pd.DataFrame(wb.sheets('CF1').range('K3:M'+str(end_row)).value,
                                          columns = wb.sheets('CF1').range('K2:M2').value)
                # Fwd & Disc Rates
                fwdRates = []
                discF = []
                for i,r in df_StEndDt.iterrows():
                    fixdt, start, end = r
                    fwdRates.append(ibor_tiie.fixing(ql.Date().from_date(fixdt)))
                    discF.append(crvMXNOIS.discount(ql.Date().from_date(end)))
                df_StEndDt['FltRate'] = fwdRates
                df_StEndDt['DF'] = discF
                # print fwd & disc rates
                wb.sheets('CF1').range('W3').value = df_StEndDt[['FltRate']].values
                wb.sheets('CF1').range('X3').value = df_StEndDt[['DF']].values
        elif user_option==5:
            # Trading Blotter Pull
            pull_blotters(path_trading_blotter)
            # Trading Blotter Valuation
            fn.proc_Pricing_Blotter(ql_settle_dt, ibor_tiie, tiie_swp_engine, 
                                    wb.sheets('Blotter'),brCrvs, dic_data)
        elif user_option==6:
            # Corros file will be updated with best bids and offers
            corros_book = xw.Book(corros_file, update_links=False)
            
            sheets = ['ENMX', 'RTMX', 'SIFM', 'TMEX6', 'GFIM', 'MEI', 'VAR', 
                      'SIPO', 'Spreads']
            for s in sheets:
                corros_book.sheets(s).api.Calculate()
                
            best_spreads, paths_data, closes_df = pr.corros_fn(corros_book)
            pr.fill_rates(wb, best_spreads, closes_df)
           
        elif user_option==7:
            # TW Collapse
            fn.proc_PriceCollpse_tw(ql_settle_dt, ibor_tiie, tiie_swp_engine, 
                                    frCrvs, brCrvs, banxico_TIIE28, dic_data, 
                                    str_cwd, str_file, wb, rng_pricing)
        elif user_option==8:
            # End Session
            proc_EndSession()
        elif user_option == 9:
            # Print bootstrapped curves
            tmpdF_OIS, tmpdF_TIIE = [], []
            for n in pd.Series(wb.sheets('DF').range('B4:B10924').value)/360:
                tmpdF_OIS.append(crvMXNOIS.discount(n))
                tmpdF_TIIE.append(crvTIIE.discount(n))
            wb.sheets('DF').range('D4').value = pd.DataFrame(tmpdF_OIS).values
            wb.sheets('DF').range('I4').value = pd.DataFrame(tmpdF_TIIE).values
        elif user_option == 10:
            # Compute cost rates
            fn.proc_CostRates(ql_settle_dt, ibor_tiie, tiie_swp_engine, frCrvs, brCrvs,
                             banxico_TIIE28, dic_data, str_cwd, str_file, 
                             wb_pricing, wb_tiie)
        elif user_option == 11:
            # Compute bid-offer rates
            fn.proc_BidOffer(ql_settle_dt, ibor_tiie, tiie_swp_engine, frCrvs, 
                             brCrvs, banxico_TIIE28, dic_data, str_cwd, 
                             str_file, wb_pricing, wb_tiie)
        else:
            # Continue
            continue

#%%
