# -*- coding: utf-8 -*-
"""
Blotter CME

This code helps to make the cme-blotter booking easier.

@author: jquintero
"""
#%%############################################################################
# MODULES
###############################################################################
import pandas as pd
import numpy as np
from datetime import date as dt
import xlwings as xw
import warnings
warnings.simplefilter("ignore")
#%%############################################################################
# GLOBALS & UDMODULES
###############################################################################
# Paths
path_blttrs = r'\\tlaloc\cuantitativa\Fixed Income\File Dump\Blotters\TIIE'
path_pswps = r'\\tlaloc\cuantitativa\Fixed Income\File Dump\PosSwaps'

# Files
dateIsNotOk = True
while dateIsNotOk:
    print('\n\nToday Date')
    input_year = int(input('\tYear: '))
    input_month = int(input('\tMonth: '))
    input_day = int(input('\tDay: '))
    try:
        dt_today = dt(input_year, input_month, input_day)
        dateIsNotOk = False
    except:
        print('Wrong date! Try again pls.')
        dateIsNotOk = True

# Last PosSwaps File
dt_yestb = dt_today - pd.tseries.offsets.BDay(1) #.strftime("%Y%m%d")
filepath_pswps = path_pswps+'\\'+'PosSwaps'+dt_yestb.strftime("%Y%m%d")+'.xlsx'
df_pswps = pd.read_excel(filepath_pswps)
df_pswps['swp_cpa_vta'] = df_pswps['swp_cpa_vta'].map({'CPA':'PAGO', 'VTA':'RECI'})

# Blotter CME File       
blttrName = 'Blotter CME ' + dt_today.strftime("%Y%m%d") 
str_inputsFileExt = '.xlsx'
str_filepath = path_blttrs + '\\' + blttrName + str_inputsFileExt

# CME-Blotter
xlsheet = 'CME'
df_blttrCME = pd.read_excel(str_filepath, xlsheet, usecols='A:M')

# DataFrame Mgmt
idx_r,idx_c = np.where(df_blttrCME.apply(
    lambda r: r.astype(str).str.contains('USUARIO')))
tmp_newcols = df_blttrCME.iloc[idx_r[0], :].tolist()
tmp_newcols = [str(s) for s in tmp_newcols]
df = df_blttrCME.drop(range(0,1+idx_r[0]))
df.columns = [s.replace(' ','') for s in tmp_newcols]
df = df.reset_index(drop=True)

#%% Check inconsistencies
uw_cond = df['DERIVADO'].fillna('').astype(str).apply(lambda x: x[:3]) == 'UNW'
bks_cond = df['USUARIO'].isin([1814, 8085])
df_orig_swps = df_pswps.iloc[
np.where(df_pswps['swp_ctrol'].isin(df[uw_cond*bks_cond]['FOLIOORIG']))[0]]
## Original trades from posswaps
orig_id = pd.concat([df_orig_swps['swp_ctrol'],
    df_orig_swps.apply(lambda x: x['swp_cpa_vta']+'_'+\
                       str(int(x['swp_monto']))+'_'+\
                       str(x['swp_val_i']+x['swp_val_i_pa']), 
                axis=1).rename('id')], axis=1).set_index('swp_ctrol')

## Suggested trades from blttr cme
found_id = pd.concat([df[uw_cond*bks_cond]['FOLIOORIG'],
                      df[uw_cond*bks_cond].apply(lambda x: 
                           x['DERIVADO'].replace('UNWIND/','')+'_'+\
                               str(int(x['MONTO']))+'_'+str(x['TASA']+0.0), 
                           axis=1).rename('id')], axis=1).set_index('FOLIOORIG')
found_id.index.name = 'swp_ctrol'
## UWD side, notional and rate coherent with pswaps
isBlttrCMEOk = orig_id.id.isin(found_id.to_numpy().reshape(-1,).tolist())

if not isBlttrCMEOk.all():
    import sys
    print('Blotter CME has some trades off than in original PosSwaps file!')
    print('Trades possibly wrong are showed below:\n')
    wt = df[df['FOLIOORIG'].isin(isBlttrCMEOk.index[~isBlttrCMEOk.id].tolist())]
    print(wt)
    sys.exit()

#%% Filter out new trades to book
tmp_cond = ~df['USUARIO'].isna() & df['FOLIOORIG'].isna()
df_trades = df[tmp_cond]

# JAM-Blotter template
try:
    blotter_template_file = xw.Book(
        r'\\tlaloc\cuantitativa\Fixed Income\File Dump\Catalogues'\
            + r'\Upload_blotter.xlsx', 
        update_links = False)
except:
    print("Many heavy files calc'ing by the moment. Setting manual mode on!")
    xw.App.calculation = 'manual'
    blotter_template_file = xw.Book(
        r'\\tlaloc\cuantitativa\Fixed Income\File Dump\Catalogues'\
            + r'\Upload_blotter.xlsx', 
        update_links = False)

blotter_template_sheet = \
    blotter_template_file.sheets('BlotterTIIE_auto')

blotter_template_sheet.range('C4:S10000').clear_contents()

blotter_folios = \
    pd.DataFrame(columns = blotter_template_sheet.range('C3:R3').value)

# New trades in blotter format
df_trades['Size'] = df_trades['MONTO']*df_trades['OPERACIÓN'].map({'RECI':1,'PAGO':-1})/1e6
dic_colrename = {'USUARIO':'Book', 'BROKER':'Socio Liquidador', 
 'FECHADEINICIO': 'Fecha Inicio', 'FECHADEVTO':'Fecha vencimiento',
 'TASA':'Yield(Spot)'}
df_trades = df_trades.rename(columns=dic_colrename)
df_trades['User'] = 2858
df_trades['Ctpty'] = 'i1056'
df_trades['Tenor'] = (df_trades['Fecha vencimiento']-df_trades['Fecha Inicio']).apply(lambda x: str(int(x.days/28))+'m')
df_trades['Broker'] = 'na'
df_trades['Socio Liquidador'][
    df_trades['Socio Liquidador'].apply(lambda x: x.lower()) != 'citi'
    ] = 'gs'
df_trades['Socio Liquidador'][
    df_trades['Socio Liquidador'].apply(lambda x: x.lower()) == 'citi'
    ] = ''
# drop duplicates
df_trades = df_trades.loc[:,~df_trades.columns.duplicated()]
# filter out non-trading desk books
df_trades = df_trades.loc[df_trades['Book'].isin([1814,8085]),:]
#%% NEW TRADES
#paste1
blotter_template_sheet.range('C4').value = \
    df_trades[['User','Book','Tenor','Yield(Spot)']].values
#paste2
blotter_template_sheet.range('I4').value = \
    df_trades[['Size','Ctpty','Fecha Inicio','Fecha vencimiento']].values
#paste3
blotter_template_sheet.range('N4').value = \
    df_trades[['Socio Liquidador','Broker']].values
    
#paste4
blotter_template_sheet.range('AN:AN').clear_contents()
blotter_template_sheet.range('AN3').value = 'P&L'
blotter_template_sheet.range('AR4').value = df_trades[['UTI']].values

#%% INTERN SWAPS
df_int = df[uw_cond].groupby(['FECHADEINICIO','FECHADEVTO','TASA'])
df_int = df_int.filter(lambda x: x['USUARIO'].nunique() > 1)
df_int = df_int[df_int['USUARIO'] != 1814]
df_int['MONTO'] = np.where(df_int['DERIVADO'].apply(lambda x: x[-1]) == 'O',
                           df_int['MONTO']*-1,df_int['MONTO'])
df_int_8085 = df_int.groupby(['FECHADEINICIO','FECHADEVTO','TASA','USUARIO'])
df_monto_int = df_int_8085['MONTO'].sum()

rows = []
for (start, end, yieldd, book), size in df_monto_int.to_dict().items():    
    tenor = f'{(end-start).days//28}m'
    size = size//1000000
    start = start.strftime("%d/%m/%Y")
    end = end.strftime("%d/%m/%Y")
    second_row = ['2058', '1814', tenor, yieldd, "", "", -size, f'U{book}', start, end]
    rows.append(second_row)
    
    if book != 2342:
        first_row = ['2058', book, tenor, yieldd, "", "", size, 'U1814', start, end]
        rows.append(first_row)
    
excel_row = blotter_template_sheet.range('C3').end('down').row
blotter_template_sheet.range(f'C{excel_row+1}').value = rows
#%% UNWIND TRADES
#uw_cond = df['DERIVADO'].fillna('').astype(str).apply(lambda x: x[:3]) == 'UNW'
#bks_cond = df['USUARIO'].isin([1814, 8085])
excel_row2 = blotter_template_sheet.range('C3').end('down').row+3

df_uw = df[uw_cond*bks_cond].rename(columns=dic_colrename).dropna(axis=1)
df_uw['Tenor'] = (df_uw['Fecha vencimiento']-df_uw['Fecha Inicio']).apply(lambda x: str(int(x.days/28))+'m')
df_uw['Size'] = df_uw['MONTO']*df_uw['DERIVADO'].apply(lambda x: x[-4:]).map({'RECI':-1,'PAGO':1})/1e6
#paste1
blotter_template_sheet.range('D'+str(excel_row2)).value = \
    df_uw[['Book','Tenor','Yield(Spot)']].values
#paste2
blotter_template_sheet.range('I'+str(excel_row2)).value = \
    df_uw[['Size']].values
#paste3
blotter_template_sheet.range('K'+str(excel_row2)).value = \
    df_uw[['Fecha Inicio','Fecha vencimiento']].values
#paste3
blotter_template_sheet.range('P'+str(excel_row2)).value = \
    df_uw[['FOLIOORIG']].values
blotter_template_sheet.range('Q'+str(excel_row2)).value = \
    df_uw[['FOLIOORIG']].values
#paste4
blotter_template_sheet.range('AR'+str(excel_row2)).value = \
    df_uw[['UTI']].values

#%% SESSION END

# save
save_name = path_blttrs + rf'\blotter_tiie_cme_{dt_today.strftime("%Y%m%d") }.xlsx'
blotter_template_file.save(save_name)

# close
input_close = str(input('Close file or session? (F/S)')).lower()[0]
if input_close == 's':
    xlsessid = blotter_template_file.app.pid
    xw.apps[xlsessid].api.Quit()
else:
    blotter_template_file.close()

