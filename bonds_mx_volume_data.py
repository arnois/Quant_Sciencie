# -*- coding: utf-8 -*-
"""
Created on Fri May  7  2021
Script desgined for pulling and aggregating MBONOS trading volume data.

@author: jquintero; aka Arnulf, Arnois or Arnuá.
    arnulf.q@gmail.com
"""

# In[]
# # Modules
import pandas as pd
import numpy as np
import os

# In[]
# # UDFs    
# ## Data Processing
def xfiProc(xfi):
    tmp_xfi = xfi.copy()
    ## drop unnamed columns
    tmp_dropcols = list(tmp_xfi.filter(regex='Unnamed'))
    tmp_xfi.drop(tmp_dropcols, axis=1, inplace=True)
    ## drop unnecessary rows
    tmp_xfi.drop(labels=[0,1],axis=0, inplace=True)
    ## set dates as indices
    tmp_xfi.set_index(tmp_xfi.iloc[:,0], inplace=True)
    tmp_xfi.drop(tmp_xfi.iloc[:,0].name, axis=1, inplace=True)
    tmp_xfi.index.name = 'Date'
    ## replace column names
    tmp_xfi_newcols = [s.replace('\n','').replace(' ','') for s in tmp_xfi.columns]
    tmp_xfi.columns = tmp_xfi_newcols
    ## fill NaN with 0
    tmp_xfi.replace(np.nan,0, inplace=True)
    return tmp_xfi

# ## Data files importing
def xcelImport(dir_path,pattern,sheetname):
    tmp_mtch_files = [f for f in os.listdir(dir_path) if pattern in f]
    xfi = None
    if len(tmp_mtch_files)!=0:
        ## data importing and processing
        for file in tmp_mtch_files:
            tmp_fpath = dir_path+file
            tmpxfi = pd.read_excel(tmp_fpath, sheet_name=sheetname)
            tmpxfi_proc = xfiProc(tmpxfi)
            tmplst = [xfi, tmpxfi_proc]
            xfi = pd.concat(tmplst).sort_values(by='Date')
        ## shave off duplicates
        xfi = xfi[~xfi.index.duplicated()]
    return xfi

# ## Database save/export
def saveData(df,fname,dir_path,fext='.csv',index=True):
    tmp_fpath = dir_path+fname+fext
    df.to_csv(tmp_fpath, index=index)
    print(f'Data saved successfully in: {tmp_fpath}')

# ## Database import    
def importSavedData(dir_path, fname, fext='.csv', 
                    index_col=0, indexAsDates=False):
    tmp_fname = dir_path+fname+fext
    if indexAsDates:
        return pd.read_csv(tmp_fname, 
                           index_col=index_col, parse_dates=[index_col])
    else:
        return pd.read_csv(tmp_fname, index_col=index_col)

# ## Database update    
def updateDataFrame(df,dir_path,pattern,sheetname):
    xfi = xcelImport(dir_path,pattern,sheetname)
    tmplst = [df,xfi]
    df_updt = pd.concat(tmplst).sort_values(by='Date')
    df_updt = df_updt[~df_updt.index.duplicated()]
    return df_updt

# In[]
# # Database import
tmp_dpath = 'H:\Trading\SystemsMercenaries\RV\db\\'
## INSTITUCIONALES
tmp_fnam = 'db_INTT'
df_intt = importSavedData(tmp_dpath,tmp_fnam,indexAsDates=True)
## INTERBANCARIOS
tmp_fnam = 'db_INTB'
df_intb = importSavedData(tmp_dpath,tmp_fnam,indexAsDates=True)

# In[]
# # Database update
updtpath = 'H:\Trading\SystemsMercenaries\RV\db\\volume\\'
updtpttn = 'INSTITUCIONALES'
updtsht = 'INSTITUCIONALES Ms'
df_intt_updt = updateDataFrame(df_intt,updtpath,updtpttn,updtsht)

# # Databse init
# ## Excel Files Import
tmp_dpath = 'H:\Trading\SystemsMercenaries\RV\db\\volume\\'
# importing all INSTITUCIONALES files
tmp_fpttn = 'INSTITUCIONALES'
tmp_sht ='INSTITUCIONALES Ms'    
df_INTT = xcelImport(tmp_dpath,tmp_fpttn,tmp_sht)
# importing all INTERBANCARIO files
tmp_fpttn = 'INTERBANCARIO'
tmp_sht = 'INTERBANCARIO Ms'
df_INTB = xcelImport(tmp_dpath,tmp_fpttn,tmp_sht)


# In[]
# # Database export
tmp_spath = 'H:\Trading\SystemsMercenaries\RV\db\\'
saveData(df_INTT,'db_INTT',tmp_spath)
saveData(df_INTB,'db_INTB',tmp_spath)





