"""
MT5 Trading Strategy

Trading signals based on associative model fit by a copula between two assets.

@author: JArnulf QC (arnulf.q@gmail.com)
"""
###############################################################################
# MODULES
###############################################################################
import sys, os, pickle    
import json
sys.path.append(r'H:\Python\mt5_ea_copmod\\')
import mt5_interface
import numpy as np
import pandas as pd
import pandas_ta as ta
import warnings
warnings.filterwarnings("ignore")
import copulas.bivariate as biCop
import datetime
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
import time
import tkinter
###############################################################################
# GLOBALS
###############################################################################
# List of tradeable symbols
tradeables = np.array(['ND100m_O', 'ND100m_H', 'ND100m_L', 'ND100m_C', 
                       'SP500m_O', 'SP500m_H', 'SP500m_L', 'SP500m_C', 
                       'SPN35_O', 'SPN35_H', 'SPN35_L', 'SPN35_C', 
                       'STOX50_O', 'STOX50_H', 'STOX50_L', 'STOX50_C', 
                       'NI225_O', 'NI225_H', 'NI225_L', 'NI225_C', 
                       'UK100_O', 'UK100_H', 'UK100_L', 'UK100_C', 
                       'HSI50_O', 'HSI50_H', 'HSI50_L', 'HSI50_C', 
                       'AUDNZD_O', 'AUDNZD_H', 'AUDNZD_L', 'AUDNZD_C', 
                       'XAUUSD_O', 'XAUUSD_H', 'XAUUSD_L', 'XAUUSD_C', 
                       'EURCHF_O', 'EURCHF_H', 'EURCHF_L', 'EURCHF_C', 
                       'EURNOK_O', 'EURNOK_H', 'EURNOK_L', 'EURNOK_C', 
                       'XAGUSD_O', 'XAGUSD_H', 'XAGUSD_L', 'XAGUSD_C',
                       'XPDUSD_O', 'XPDUSD_H', 'XPDUSD_L', 'XPDUSD_C', 
                       'XPTUSD_O', 'XPTUSD_H', 'XPTUSD_L', 'XPTUSD_C', 
                       'EURSEK_O', 'EURSEK_H', 'EURSEK_L', 'EURSEK_C',
                       'USDMXN_O', 'USDMXN_H', 'USDMXN_L', 'USDMXN_C', 
                       'EURUSD_O', 'EURUSD_H', 'EURUSD_L', 'EURUSD_C', 
                       'GBPUSD_O', 'GBPUSD_H', 'GBPUSD_L', 'GBPUSD_C', 
                       'EURGBP_O', 'EURGBP_H', 'EURGBP_L',  'EURGBP_C', 
                       'USDJPY_O', 'USDJPY_H', 'USDJPY_L', 'USDJPY_C',
                       'EURJPY_O', 'EURJPY_H', 'EURJPY_L', 'EURJPY_C', 
                       'GBPJPY_O', 'GBPJPY_H', 'GBPJPY_L', 'GBPJPY_C', 
                       'AUDUSD_O', 'AUDUSD_H', 'AUDUSD_L', 'AUDUSD_C', 
                       'USDCAD_O', 'USDCAD_H', 'USDCAD_L', 'USDCAD_C', 
                       'NZDUSD_O', 'NZDUSD_H', 'NZDUSD_L', 'NZDUSD_C',
                       'USDTRY_O', 'USDTRY_H', 'USDTRY_L', 'USDTRY_C', 
                       'USDZAR_O', 'USDZAR_H', 'USDZAR_L', 'USDZAR_C', 
                       'USDNOK_O', 'USDNOK_H', 'USDNOK_L', 'USDNOK_C', 
                       'USDCNH_O', 'USDCNH_H', 'USDCNH_L', 'USDCNH_C', 
                       'USDCHF_O', 'USDCHF_H', 'USDCHF_L', 'USDCHF_C',
                       'USDSEK_O', 'USDSEK_H', 'USDSEK_L', 'USDSEK_C', 
                       'USDCZK_O', 'USDCZK_H', 'USDCZK_L', 'USDCZK_C', 
                       'USDHUF_O', 'USDHUF_H', 'USDHUF_L', 'USDHUF_C', 
                       'USDBRL_O', 'USDBRL_H', 'USDBRL_L', 'USDBRL_C', 
                       'USDCOP_O', 'USDCOP_H', 'USDCOP_L', 'USDCOP_C',
                       'USDCLP_O', 'USDCLP_H', 'USDCLP_L', 'USDCLP_C', 
                       'AUDCAD_O', 'AUDCAD_H', 'AUDCAD_L', 'AUDCAD_C', 
                       'AUDJPY_O', 'AUDJPY_H', 'AUDJPY_L', 'AUDJPY_C', 
                       'NZDJPY_O', 'NZDJPY_H', 'NZDJPY_L', 'NZDJPY_C', 
                       'CADJPY_O', 'CADJPY_H', 'CADJPY_L', 'CADJPY_C',
                       'CHFJPY_O', 'CHFJPY_H', 'CHFJPY_L', 'CHFJPY_C', 
                       'USDHKD_O', 'USDHKD_H', 'USDHKD_L', 'USDHKD_C'])
# External database asset names
lst_assetnames = ['NQ1','ES1','TU1','FV1','TY1','RX1','CL1','GC1','USDMXN']
# 5M OHLC data file path
str_path = r'C:\Users\jquintero\db'
str_file = r'\data_5m_y2023.xlsx'
###############################################################################
# UDF
###############################################################################
# Function to slice high time-granular dataframe by hour number limits
def slice_HGDF(data: pd.DataFrame, hour_n1: int = 6, hour_n2: int = 15):
    """
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with dattime64[ns] index data type.
    hour_n1 : int, optional
        Lower bound hour digit. The default is 6.
    hour_n2 : int, optional
        Upper bound hour digit. The default is 15.

    Returns
    -------
    Filtered data by hour bounds.
    """
    # Sections in index between hour_n1 and hour_n2
    idx_fltrd = (data.index.hour<=hour_n2)*(data.index.hour>=hour_n1)
    
    return data[idx_fltrd]

# Function to buildup database: import xl, export parquet
def set_data_5m(str_path, str_file, str_shtname):
    """
    Database buildup from excel file when non-existent.
    """
    # columns
    tmpdf_head =  pd.read_excel(str_path+str_file,
                                sheet_name = str_shtname,
                          header=None, nrows = 2, skiprows=2)
    assetnames = tmpdf_head.iloc[0,:].dropna()
    repls = {' Index' : '', ' Curncy' : '', ' Index':'', ' Comdty':''}
    names = np.array(
        [s.replace(s[s.find(' '):], 
               repls.get(s[s.find(' '):])) 
             for s in assetnames]
        )
    # data
    tmpdf = pd.read_excel(str_path+str_file, sheet_name = str_shtname, 
                          header=None, index_col=0, engine='openpyxl')
    tmpdf = tmpdf.iloc[4:,:].dropna(axis=1)
    tmpdf.columns = tmpdf_head.iloc[1,:].dropna().drop(0)
    tmpdf = tmpdf.drop('Dates',axis=1)
    df_cols = []
    for name in names:
        df_cols = df_cols + [name+'_'+t for t in ['O', 'H', 'L', 'C']]
    tmpdf.columns = df_cols
    tmpdf.index.name = 'date'
    tmpdf.to_parquet(r'H:\db\data_5m.parquet', compression=None)
    return None

# Function to import and parse data from xlfile
def readParseData(str_path, str_file, str_shtname, n_skipRows):
    """
    Import, parse and clean data from excel database file.
    """
    xlwb = pd.ExcelFile(str_path+str_file)
    xlwb_data = xlwb.book.get_sheet_by_name(str_shtname)
    xlwb_rows = xlwb_data.max_row
    tmpdf_head =  pd.read_excel(str_path+str_file,
                                sheet_name = str_shtname,
                          header=None, nrows = 2, skiprows=2)
    assetnames = tmpdf_head.iloc[0,:].dropna()
    repls = {' Index' : '', ' Curncy' : '', ' Index':'', ' Comdty':''}
    names = np.array(
        [s.replace(s[s.find(' '):], 
               repls.get(s[s.find(' '):])) 
             for s in assetnames]
        )
    tmp_rows2skip = max(int(xlwb_rows-n_skipRows), 4)
    tmpdf = pd.read_excel(str_path+str_file,
                          sheet_name = str_shtname,
                          header=None, index_col=0,
                          skiprows=tmp_rows2skip)
    tmpdf = tmpdf.fillna(method='ffill').dropna(axis=1)
    tmpdf.columns = tmpdf_head.iloc[1,:].dropna().drop(0)
    tmpdf = tmpdf.drop('Dates',axis=1)
    
    df_cols = []
    for name in names:
        df_cols = df_cols + [name+'_'+t for t in ['O', 'H', 'L', 'C']]
    tmpdf.columns = df_cols
    tmpdf.index.name = 'date'
    xlwb.close()
    return(tmpdf)

# Function to update database
def update_data_5m(str_path, str_file, str_shtname, n_skipRows, str_dbpath):
    """
    Update existing parquet database with new observations found in str_path+
    str_file data file.
    """
        #str_shtname = "data"
        #n_skipRows = 2e4
        #str_dbpath = r"H:\db\data_5m.parquet"
    
    data = pd.read_parquet(str_dbpath)
    tmpdata = readParseData(str_path, str_file, str_shtname, n_skipRows)
    # tmpdata.iloc[~tmpdata.index.isin(data.index),:]
    updatedata = pd.concat([data, tmpdata.iloc[~tmpdata.index.isin(data.index),:]]) # data.append(tmpdata.iloc[~tmpdata.index.isin(data.index),:])
    updatedata.to_parquet(str_dbpath)
    return updatedata

# Function to get datetime delimiters for train-validation split
def dt_delims_train_valid():
    # Today's date
    today = datetime.date.today()
    # Last friday date
    lastFriday = today - datetime.timedelta(days=today.weekday()) +\
                        datetime.timedelta(days=4, weeks=-1)
    # Second last friday
    last2Friday = lastFriday - datetime.timedelta(days=7)
    # Validation set datetime ending point
    tEnd_valid = datetime.datetime(lastFriday.year, lastFriday.month, 
                                   lastFriday.day, 16, 0, 0)
    # Validation set datetime starting point
    tStart_valid = datetime.datetime(last2Friday.year, last2Friday.month, 
                                   last2Friday.day, 16, 0, 0)
    # Training set datetime starting point
    tStart_train = tStart_valid - datetime.timedelta(days=27, hours=16)
    
    return tStart_train, tStart_valid, tEnd_valid

# Function to split data into train-validation sets
def datasplit_train_valid(data):
    # Datetime delimiters
    tStart_train, tStart_valid, tEnd_valid = dt_delims_train_valid()
    # Data split: train(28d)-valid(7d)-test(1d)
    df_train = data.loc[tStart_train:tStart_valid,]
    df_valid = data.loc[tStart_valid:tEnd_valid,]
    
    return df_train, df_valid

# Function to fit best copula model
def get_bestBivarCop(tgt_y, df_train_r_u, df_valid_r_u):
    names_posspairs = df_train_r_u.columns.drop(tgt_y)
    n_posspairs = len(names_posspairs)
    df_tgtPairs = pd.DataFrame(names_posspairs, 
                               index=[tgt_y]*n_posspairs,
                               columns=['y2'])
    lst_tgtPairs = [[tgt_y,c] for c in df_tgtPairs['y2']]
    # possible best bivariate copulas
    tmpdic_copPairs = dict(
        [(p[1],biCop.select_copula(df_train_r_u[p].to_numpy())) 
         for p in lst_tgtPairs]
        )
    # possible mispricing indexes
    tmpdf_pairsM = pd.DataFrame(
            [
                v.partial_derivative(
                        df_valid_r_u[[tgt_y,k]].to_numpy()
                        ) - 0.5
                for k,v in tmpdic_copPairs.items()
             ]
        ).T.fillna(method = 'ffill').cumsum().\
        rename(columns=dict(
            [(n,name) 
             for n,name in zip(range(n_posspairs), 
                               names_posspairs)
            ]
        ))
    # best mispricing index
    tmp_adfstatsPairs = tmpdf_pairsM.apply(lambda y: adfuller(y)[0])
    # best paired mean-reverting-M
    bestpair = names_posspairs[tmp_adfstatsPairs.argmin()]
    # best bivariate copula update
    bestcopair = [tgt_y, bestpair]
    try:
        # update params
        tmpdic_copPairs[bestpair].\
            fit(pd.concat([df_train_r_u, df_valid_r_u])[bestcopair].to_numpy()) # df_train_r_u.append(df_valid_r_u)[bestcopair].to_numpy()
    except ValueError:
        # update copula model
        tmpdic_copPairs[bestpair] = \
            biCop.\
                select_copula(
                    pd.concat([df_train_r_u,df_train_r_u])[bestcopair].to_numpy()
                    )
    # save updated model
    selCop_dic = tmpdic_copPairs[bestpair].to_dict()
    # save updated mispricing index
    selCop_valid_M = pd.DataFrame(
        tmpdic_copPairs[bestpair].partial_derivative(
            df_valid_r_u[bestcopair].to_numpy())-0.5,
        index = df_valid_r_u.index,
            ).cumsum().rename(columns={0:'M'})
    res = {'covar': bestpair, 'cop': selCop_dic, 'M': selCop_valid_M}
    return(res)

# Function to fit copula model from 5M-OHLC external database
def copmodel_5M_fromDotParquet(path, filename, symbol):
    """
    Fits Best Bivariate Copula Model from a 5M-OHLC parquet database.
        path = 'C:\\Users\\jquintero\\db'
        filename = 'data_5m'
        symbol = 'ES1'
    """
    # File path
    data_path = path+r'\\'+filename+'.parquet'
    # Data
    data = pd.read_parquet(data_path)
    # Train-Valid data split
    df_train, df_valid = datasplit_train_valid(data)
    # Data log-returns
    df_train_r = get_df_ret(df_train)
    df_valid_r = get_df_ret(df_valid)
    
    # ECDF of train-set returns
    df_train_r_ecdf = get_df_ret_ecdf(df_train_r)

    # Uniform-plane data for train-valid-sets
    df_train_r_u = df_train_r.apply(lambda y: df_train_r_ecdf[y.name](y))
    df_valid_r_u = df_valid_r.apply(lambda y: df_train_r_ecdf[y.name](y))
    
    # ECDF update
    df_trva_r_ecdf = get_df_ret_ecdf(pd.concat([df_train_r, df_valid_r])) # df_train_r.append(df_valid_r)
    
    # Model fit
    model = get_bestBivarCop(symbol, df_train_r_u, df_valid_r_u)
    model = get_bestBivarCop(symbol, df_train_r_u, df_valid_r_u)
    model['df_ecdfs'] = df_trva_r_ecdf
    
    return model

# Function to export/save fitted copula model
def save_copmodel(savepath, symbol, model):
    # Week number
    week = datetime.datetime.today().isocalendar().week
    # Path to save model
    str_modelname = savepath+r'\\'+'copmodels_'+r'w'+str(week)+r'.pickle'
    # Models file
    if os.path.isfile(str_modelname):
        models = pd.read_pickle(str_modelname)
    else:
        models = {}
        with open(str_modelname, 'wb') as f:
            pickle.dump(models, f)
    # Export model
    models[symbol] = model
    with open(str_modelname, 'wb') as f:
        pickle.dump(models, f)
    return None

# Function to fit and save copula model from 5M-OHLC external database
def set_copmodel_5M_fromDotParquet(datapath, dataname, symbol, savepath):
    # Model
    model = copmodel_5M_fromDotParquet(datapath, dataname, symbol)
    # Export model
    save_copmodel(savepath, symbol, model)
    
    return None
###############################################################################
# Function to fit a set of copula models from external database
def bulkset_copmodel_5M_fromDotParquet(datapath, dataname, 
                                       symbols, savepath):
    """
    Fits Best Bivariate Copula Models for a list of target variables. This
    function-method usually run at first day of starting week. 
    
    Default: 
        datapath, dataname, savepath = r'H:\db', 'data_5m', r'H:\Python\models'
        symbols = lst_assetnames
    """
    # Models dictionary
    models = {}
    # Models fitting process
    itime = datetime.datetime.now()
    for symbol in symbols:
        models[symbol] = copmodel_5M_fromDotParquet(datapath, dataname, symbol)
    ftime = datetime.datetime.now()
    elapsedtime = ftime-itime
    # Path to save model
    week = datetime.datetime.today().isocalendar().week
    str_modelname = savepath+r'\\'+'copmodels_'+r'w'+str(week)+r'.pickle'
    # Export model
    with open(str_modelname, 'wb') as f:
        pickle.dump(models, f)
    print(f"\nElapsed time fitting models: {elapsedtime.seconds/60:.2f} minutes\n")
    return None
###############################################################################
# Function to import settings from settings.json
def get_project_settings(importFilepath):
    # Test the filepath to sure it exists
    if os.path.exists(importFilepath):
        # Open the file
        f = open(importFilepath, "r")
        # Get the information from file
        project_settings = json.load(f)
        # Close the file
        f.close()
        # Return project settings to program
        return project_settings
    else:
        return ImportError

# Funciton to init MT5 program and account
def main_initMT5(settings_json_path):
    # Trading environment settings
    project_settings = get_project_settings(settings_json_path)
    
    # Start MT5
    mt5_interface.start_mt5(project_settings["username"], 
                            project_settings["password"], 
                            project_settings["server"],
                           project_settings["mt5Pathway"])
    # Init trading symbol
    mt5_interface.initialize_symbols(project_settings["symbols"])
    
# Function to manage OnCandleEvent
def time_OnChartEvent_5M_XL(wb, wb_test, nprevsecs = 11):
    n_secs2wait1 = min(abs(time2wait_M5().seconds-nprevsecs),300)
    print(f"Sleeping {int(n_secs2wait1)} secs . . .")
    slpuntl = datetime.datetime.today() +\
                  datetime.timedelta(seconds=n_secs2wait1)
    print("until {:02d}:{:02d}:{:02d}".\
          format(slpuntl.hour, slpuntl.minute, slpuntl.second))
    time.sleep(int(n_secs2wait1))
    wb_test.range('C2').value = 1
    time.sleep(5)
    wb.save()
    wb_test.range('C2').value = 0
    time.sleep(5)
    
# Function to manafe OnCandleEvent from XL database
def time_OnChartEvent_XL_update(wb):
    time.sleep(6)
    wb.save()

# Function to run copmodel for assets in external XL file
def main_copmodel_5M_feedFromXL(feedpath = r'C:\Users\jquintero\db', 
                        feedname = r'fut_copmod',
                        symbol = 'ES1',
                        stop_hour = 15,
                        isMain = True,
                        wb = None): 
    # Models
    week = datetime.datetime.today().isocalendar().week
    str_modelname = r'H:\Python\models\copmodels_'+r'w'+str(week)+r'.pickle'
    models = pd.read_pickle(str_modelname)
    
    # Current time
    df_run = get_copmodel_run_fromXL(feedpath, feedname, symbol, models)
    strategy_one(symbol, models, feedpath, feedname)
    
    # Session
    today_hour = get_session_hour()
    while today_hour < stop_hour:
        # Update session hour
        today_hour = get_session_hour()
        
        # Compare against previous time
        if isNewCandle('M5', df_run):
            # Notify user
            print("\nNew Candle")
            # Model run
            df_run = get_copmodel_run_fromXL(feedpath, feedname, 
                                             symbol, models)
            # CopMod strategy over selected asset
            strategy_one(symbol, models, feedpath, feedname)
        
        if isMain:
            wb_test = wb.sheets['test']
            time_OnChartEvent_5M_XL(wb, wb_test, nprevsecs = 11)
        else:
            # Waiting till next candle arises
            n_secs2wait1 = min(abs(time2wait_M5().seconds+5),300)
            print(f"Sleeping {int(n_secs2wait1)} secs . . .")
            slpuntl = datetime.datetime.today() +\
                          datetime.timedelta(seconds=n_secs2wait1)
            print("until {:02d}:{:02d}:{:02d}".\
                  format(slpuntl.hour, slpuntl.minute, slpuntl.second))
            time.sleep(int(n_secs2wait1))

    # Close XL
    return None

# Function to algo-trade futures feed from external xl database
def main_algotrade(feedpath = r'C:\Users\jquintero\db', 
                   feedname = r'fut_copmod',
                   futures = ['ES1', 'NQ1'],
                   models = None,
                   wb = None):
    # Current time
    str_xldatafeed = feedpath + '\\' + feedname + '.xlsx'
    df_candlest = pd.read_excel(str_xldatafeed, index_col=0, 
                                usecols=list(range(2)), 
                                sheet_name='test', skiprows=3).dropna()
    # Compare against previous time
    if isNewCandle('M5', df_candlest):
        # Notify user
        print("\nNew Candle")
        # Update current time
        time_OnChartEvent_XL_update(wb)
        for symbol in futures:
            # Model run
            df_run = get_copmodel_run_fromXL(feedpath, feedname, 
                                             symbol, models)
            # CopMod strategy over selected asset
            res = algotrade_symbol(symbol, df_run)
            print(res)
    # Continue      
    return None

# Function to get ECDF
def get_df_ret_ecdf(df_ret):
    from statsmodels.distributions.empirical_distribution import ECDF
    df_ret_ecdf = df_ret.apply(ECDF)
    df_ret_ecdf.index = [s.replace('_C','') for s in df_ret.columns]
    return(df_ret_ecdf)

# Function to filter out symbols disabled for trade
def get_disabled_symbols(df):
    symbol_names = np.unique(
        [s.replace('_C','') for s in [s for s in df.columns if '_C' in s]]
        )
    s_trade_mode = {}
    for s in symbol_names:
        s_trade_mode[s] = mt5_interface.\
            MetaTrader5.symbol_info(s)._asdict()['trade_mode']
    return [k for k, v in s_trade_mode.items() if v == 0]

# Function to fit copula model throughout a whole year
def model_train(tgt_y: str = 'USDMXN', n_year: int = 2022) -> dict:
    # Data
    str_dbpath = r"H:\db\data_5m.parquet"; data = pd.read_parquet(str_dbpath)
    
    # Filtered Data from 6 to 15 hour handles
    data = slice_HGDF(data,6,15)
    
    # OC returns
    df_log_ret = get_df_ret(data.loc[str(n_year)])
    
    # ECDF from OC returns
    df_ecdf = get_df_ret_ecdf(df_log_ret)
    
    # Uniform
    df_U = df_log_ret.apply(lambda y: df_ecdf[y.name](y))
    
    # Target-Covariate Pairs
    # # tgt_y = 'USDMXN'
    names_posspairs = df_U.columns.drop(tgt_y)
    n_posspairs = len(names_posspairs)
    df_tgtPairs = pd.DataFrame(names_posspairs, 
                               index=[tgt_y]*n_posspairs,
                               columns=['y2'])
    lst_tgtPairs = [[tgt_y,c] for c in df_tgtPairs['y2']]
    
    # Bivariate Copulas
    tmpdic_copPairs = dict(
        [(p[1], biCop.select_copula(df_U[p].to_numpy())) 
         for p in lst_tgtPairs]
        )
    
    # Association stats
    pairsTau = pd.DataFrame([v.tau for k,v in tmpdic_copPairs.items()], 
                 columns=['tau'], 
                 index=tmpdic_copPairs.keys()).abs()
    
    # Mispricing indexes
    tmpdf_pairsM = pd.DataFrame([
        v.partial_derivative(df_U[[tgt_y,k]].to_numpy())-0.5 
        for k,v in tmpdic_copPairs.items()
        ]).T.fillna(method = 'ffill').cumsum().\
        rename(columns=dict(
            [(n,name) 
             for n,name in zip(range(n_posspairs), 
                               names_posspairs)
            ]
        ))
    
    # Stationarity stats
    pairsM_adf = tmpdf_pairsM.apply(lambda y: adfuller(y)[0])
    pairsM_adf_top5 = pairsM_adf.drop(
        [s for s in pairsM_adf.index.tolist() if len(s) != 6]
        ).sort_values()[:5]
    
    # Top 5 weights
    tau_top5 = pairsTau.loc[pairsM_adf_top5.index.tolist()]
    w = tau_top5/tau_top5.sum()
    
    # Weighted M
    top_M = tmpdf_pairsM[w.index.tolist()].set_index(df_U.index)
    
    # Covariate selection
    bestCovar = pairsM_adf_top5.index[0] # pairsTau[0]
    bestPair = [tgt_y, bestCovar]
    
    # Model
    selCop_dict = tmpdic_copPairs[bestCovar].to_dict()
    selCop_M = tmpdf_pairsM[bestCovar].rename('M').to_frame().set_index(df_U.index)
    selCop_ECDF = df_ecdf[[tgt_y]+w.index.tolist()]
    cop_top5 = dict(zip(w.index.tolist(),
                    [tmpdic_copPairs[s].to_dict() for s in w.index.tolist()]))
    model = {'covar': bestCovar, 'cop': selCop_dict, 'M': selCop_M, 
             'df_ecdfs': selCop_ECDF, 'w': tau_top5, 'top_M': top_M, 'topCovar': cop_top5}
    
    # Save
    savepath = r'H:\Python\models'
    str_modelname = savepath+f'\copmodel_{tgt_y}'+r'.pickle'
    with open(str_modelname, 'wb') as f:
        pickle.dump(model, f)
    
    return model

# Function to fit copmodel given couple
def copmodel_byCouple(symbol='USDMXN', timeframe='M5', couple='USDZAR'):
    # Selected pairs
    lst_pairs = [s+'_'+t for s in [symbol, couple] 
                 for t in ['O', 'H', 'L', 'C']]
    # Timezone shift
    hourshift = mt5_interface.get_timezone_shift_hour(symbol)
    # Pull data
    tmpdf = mt5_interface.query_bulkdata(timeframe, hourshift)
    tmpdf2 = tmpdf.loc[:,lst_pairs].fillna(method='ffill').fillna(method='bfill').dropna(axis=1)
    
    # Train-Valid data split
    df_train, df_valid = datasplit_train_valid(tmpdf2)
    # Data log-returns
    df_train_r = get_df_ret(df_train)
    df_valid_r = get_df_ret(df_valid)
    
    # ECDF of train-set returns
    df_train_r_ecdf = get_df_ret_ecdf(df_train_r)

    # Uniform-plane data for train-valid-sets
    df_train_r_u = df_train_r.apply(lambda y: df_train_r_ecdf[y.name](y))
    df_valid_r_u = df_valid_r.apply(lambda y: df_train_r_ecdf[y.name](y))
    
    # ECDF update
    df_trva_r_ecdf = get_df_ret_ecdf(pd.concat([df_train_r, df_valid_r])) # get_df_ret_ecdf(df_train_r.append(df_valid_r))
    
    #Model specs
    model = get_bestBivarCop(symbol, df_train_r_u, df_valid_r_u)
    model = get_bestBivarCop(symbol, df_train_r_u, df_valid_r_u)
    model['df_ecdfs'] = df_trva_r_ecdf
    
    return model

# Function to fit copmodel
def copmodel(symbol='USDMXN', timeframe='M5'):
    # Timezone shift
    hourshift = mt5_interface.get_timezone_shift_hour(symbol)
    # Pull data
    tmpdf = mt5_interface.query_bulkdata(timeframe, hourshift)
    tmpdf2 = tmpdf.iloc[:,np.isin(tmpdf.columns.to_numpy(), tradeables)].\
        fillna(method='ffill').dropna(axis=1)
    untradeable = np.unique(
        [s.replace('_C','') 
         for s in 
         [s for s in tmpdf.columns[~np.isin(tmpdf.columns, tmpdf2.columns)] 
          if '_C' in s]
         ]
        )
    # Filter out disabled symbols
    disabled_s = get_disabled_symbols(tmpdf2)
    if symbol in disabled_s+untradeable.tolist():
        print(f'\n{symbol} disabled for trading!\n')
        return
    else:
        tmpdf2 = tmpdf2.\
            drop([s+'_'+t for s in disabled_s for t in ['O', 'H', 'L', 'C']],
                 axis=1)
    # Train-Valid data split
    df_train, df_valid = datasplit_train_valid(tmpdf2)
    # Data log-returns
    df_train_r = get_df_ret(df_train)
    df_valid_r = get_df_ret(df_valid)
    
    # Kendall's tau
    ktau_train = df_train_r.corr(method='kendall')
    ktau_train_top5 = ktau_train['USDMXN'].apply(abs).sort_values(ascending=False).drop('USDMXN')[:5]
    
    # ECDF of train-set returns
    df_train_r_ecdf = get_df_ret_ecdf(df_train_r)

    # Uniform-plane data for train-valid-sets
    df_train_r_u = df_train_r.apply(lambda y: df_train_r_ecdf[y.name](y))
    df_valid_r_u = df_valid_r.apply(lambda y: df_train_r_ecdf[y.name](y))
    
    # ECDF update
    df_trva_r_ecdf = get_df_ret_ecdf(df_train_r.append(df_valid_r))
    
    # Model specs
    model = get_bestBivarCop(symbol, df_train_r_u, df_valid_r_u)
    model = get_bestBivarCop(symbol, df_train_r_u, df_valid_r_u)
    model['df_ecdfs'] = df_trva_r_ecdf
    
    return model

# Function to get multiple real-time data for copmodel
def get_copmodel_hfdata_multiple(lst_y, timeframe):
    # Timezone shift
    hourshift = mt5_interface.get_timezone_shift_hour(lst_y[0])
    
    # Covariables data query from MT5
    hfdata = mt5_interface.query_today_data(lst_y[0], timeframe, hourshift)
    for y in lst_y[1:]:
        tmp = mt5_interface.query_today_data(y, timeframe, hourshift)
        hfdata = hfdata.merge(tmp, how='left', left_index =True, right_index=True)
        
    return hfdata

# Function to get pairs of real-time data for copmodel
def get_copmodel_hfdata(y1, y2, timeframe):
    # Timezone shift
    hourshift = mt5_interface.get_timezone_shift_hour(y1)
    # Covariables data query from MT5
    hfdata_y1 = mt5_interface.query_today_data(y1, timeframe, hourshift)
    hfdata_y2 = mt5_interface.query_today_data(y2, timeframe, hourshift)
    # Test data to use in copula model
    hfdata = hfdata_y1.merge(hfdata_y2, how='left', 
                             left_index =True, right_index=True)
    return hfdata

# Function to run weighted model on given variables and its respective data
def get_copmodel_run_w(symbol, timeframe, model):
    # Covariates
    covars = model['w'].index.tolist()
    
    # Copulas load
    dict_cops = {}
    for name in covars:
        dict_cops[name] = biCop.Bivariate.from_dict(model['topCovar'][name])
    
    # ECDFs
    df_trva_r_ecdf = model['df_ecdfs']

    # Test data
    lst_y = df_trva_r_ecdf.index.tolist()
    hfdata = get_copmodel_hfdata_multiple(lst_y, timeframe)
    
    # Test-set uniform data
    df_test_r = get_df_ret(hfdata)
    df_test_r_u = df_test_r.apply(lambda y: df_trva_r_ecdf[y.name](y))
    covars = df_test_r_u.columns.tolist(); covars.remove(symbol)
    
    # Conditional cdfs
    mpxidx_test = pd.DataFrame(columns=covars)
    for cov in covars:
        cop = dict_cops[cov]
        cop_cprob = cop.partial_derivative(df_test_r_u[[symbol, cov]].to_numpy())
        mpxidx_test[cov] = cop_cprob
    
    # Mispricing indexes
    mpxidx_test = mpxidx_test - 0.5
    mpxidx_test = mpxidx_test.cumsum(axis=1)
    
    # Weighted Mispricing Index (WMI)
    taus = model['w'].loc[covars]
    w = taus/taus.sum()
    mpxidx_test = mpxidx_test.dot(w).\
        set_index(df_test_r_u.index).rename(columns={'tau':'M'})
    
    # Trainset WMI
    train_M = model['top_M'][covars].dot(w).rename(columns={'tau':'M'})
    
    # TI applied to Mt
    df_M = train_M.append(mpxidx_test[['M']])
    df_M.ta.rsi(close='M', length=9, suffix='M', append=True)
    df_run = df_M.copy()
    df_run = df_run.merge(hfdata[[symbol+'_C']], 
                          left_index = True, right_index = True)
    
    return df_run

# Function to run model on given variables and its respective data
def get_copmodel_run(symbol, timeframe, model):
    # Model specs
    cop = biCop.Bivariate.from_dict(model['cop'])
    y2 = model['covar']
    df_trva_r_ecdf = model['df_ecdfs']

    # Test data
    hfdata = get_copmodel_hfdata(symbol, y2, timeframe)
    if y2+'_O' not in hfdata.columns:
        hfdata = get_copmodel_hfdata(symbol, y2, timeframe)
        hfdata = get_copmodel_hfdata(symbol, y2, timeframe)
    # Test-set uniform data
    df_test_r = get_df_ret(hfdata)
    df_test_r_u = df_test_r.apply(lambda y: df_trva_r_ecdf[y.name](y))
    # Conditional cum. dist. function
    mpxidx_test = pd.DataFrame(cop.partial_derivative(
        df_test_r_u[[symbol, y2]].to_numpy()),
        index = df_test_r_u.index, columns = ['h'])
    # Mispricing index (Mt)
    mpxidx_test['m'] = mpxidx_test.h-0.5
    mpxidx_test['M'] = mpxidx_test.m.cumsum()
    # TI applied to Mt
    df_M = pd.concat([model['M'], mpxidx_test[['M']]]) # model['M'].append(mpxidx_test[['M']])
    df_M.ta.rsi(close='M', length=9, suffix='M', append=True)
    df_run = df_M.copy()
    df_run = df_run.merge(hfdata[[symbol+'_C']], 
                          left_index = True, right_index = True)
    
    return df_run

# Function to run model from external data
def get_copmodel_run_fromXL(str_path, filename, symbol, models):
    # Model specs
    tgt_y = symbol
    cop = biCop.Bivariate.from_dict(models[tgt_y]['cop'])
    y2 = models[tgt_y]['covar'] 
    df_trva_r_ecdf = models[tgt_y]['df_ecdfs']
    
    str_file = '\\'+filename+'.xlsx'
    # Test data
    hfdata = readParseHFData(str_path ,str_file, 'test')
    # Target variable ATR
    atrcols = [tgt_y+s for s in ['_H','_L','_C']]
    dict_rename = dict(zip(atrcols, ['high','low','close']))
    df_atr = hfdata[atrcols].rename(columns = dict_rename).\
        ta.atr(64,ta.ema).fillna(method='bfill').\
            rename(tgt_y+'_ATR').to_frame()
    
    # Uniform data
    df_test_r = get_df_ret(hfdata)
    df_test_r_u = df_test_r[df_trva_r_ecdf.index.tolist()]\
        .apply(lambda y: df_trva_r_ecdf[y.name](y))

    # Conditional cum. distribution 
    mpxidx_test = pd.DataFrame(cop.partial_derivative(
        df_test_r_u[[tgt_y, y2]].to_numpy()),
        index = df_test_r_u.index, columns = ['h'])
    # Mispricing index (Mt)
    mpxidx_test['m'] = mpxidx_test.h-0.5
    mpxidx_test['M'] = mpxidx_test.m.cumsum()
    # TI applied to Mt
    df_M = pd.concat([models[tgt_y]['M'], mpxidx_test[['M']]]) # models[tgt_y]['M'].append(mpxidx_test[['M']])
    df_M.ta.rsi(close='M', length=9, suffix='M', append=True)
    df_run = df_M.copy()
    lst_priceVars = [s+'_C' for s in [tgt_y, y2]]
    df_run = df_run.merge(hfdata[lst_priceVars], 
                          left_index = True, right_index = True).\
        merge(df_atr, left_index = True, right_index = True)
    return df_run

# Function to print model run
def print_copmodel_run(df_run):
    # Timehandles
    today = datetime.datetime.today()
    dt_now = datetime.datetime(today.year, 
                               today.month,
                               today.day, 
                               today.hour, 
                               np.max(
                                   ((int(today.minute/5)-1)*5,
                                    0)
                                   ), 0)
    dt_now_0 = dt_now - datetime.timedelta(minutes=5*10)
    # Print dataframe
    print(r'---------------------------------------------------------------'+\
          '----------------------------------------------------------------')
    print(df_run.loc[str(dt_now_0):str(dt_now),])
    print(r'---------------------------------------------------------------'+\
          '----------------------------------------------------------------')
    return None
        
# Function to plot mispricing index model run
def plot_copmodel_Mt(df_run, y2):
    # Var
    y1 = [name.replace('_C','') for name in df_run.columns 
     if '_C' in name and name.replace('_C','') != y2][0]
    #y1 = df_run.columns[-1].replace('_C','')
    # Timehandle
    today = datetime.datetime.today()
    dt_now_st = datetime.datetime(today.year, 
                               today.month,
                               today.day,
                               6,0,0)
    # Plot mispricing index
    ax = df_run.loc[str(dt_now_st):,'RSI_9_M'].plot()
    ax.set_title(rf'$M_t\left( {y1} | {y2} \right)$', 
                 fontsize = 8, loc = 'left', color = 'darkblue')
    ax.axhline(y=30, color='g', linestyle='--', lw=1.5, alpha=0.5)
    ax.axhline(y=70, color='r', linestyle='--', lw=1.5, alpha=0.5)
    ax.axhline(y=50, color='gray', linestyle='-', lw=1, alpha=0.4)
    plt.suptitle(r'RSI($M_t$)', fontsize = 15)
    plt.tight_layout()
    plt.show()
    return None

# Function to make decision based on most recent model run
def assess_copmodel_run(df_run, symbol, timeframe, pip_size, lots):
    # Test for open positions
    positions = mt5_interface.get_open_positions_s(symbol)
    # Decision making
    if len(positions) <= 0:
        # Assess Mispricing index RSI's levels
        decision = make_decision(df_run)
        create_new_order(decision, pip_size, symbol, lots)
        print(str(decision)+'\n')
    else: # Assess adding to open position
        create_new_order(make_decision(df_run), pip_size, symbol, lots)
        # Manage open positions
        for position in positions:
            decision = make_decision_position(df_run, position)
            print('\n'+decision+'\n')
            # SL based on ATR measure
            s_dig = mt5_interface.\
                MetaTrader5.symbol_info(position.symbol)._asdict()['digits']
            lastATR = mt5_interface.get_atr(position.symbol, 
                                            timeframe, 
                                            64).iloc[-1]
            new_sl_pts = round(lastATR*3, s_dig)
            eval_order(decision, position, new_sl_pts)
    return None

# Function to dosplay message box
def msgbx_showinfo(str_ttl, str_msg):
    # Info.message box
    tkinter.messagebox.showinfo(str_ttl, str_msg)

# Function to post decision based on model run from external feed
def assess_copmodel_run_5M_fromXL(df_run, symbol):
    # Assess Mispricing index RSI's levels
    decision = make_decision(df_run)
    
    # WAIT
    if decision == "DoNothing":
        return decision
    elif decision == "XA30":
    # BUY
        price = df_run[symbol+'_C'].iloc[-1]
        str_buymsg = "BUY "+symbol+f' at {price:,.2f}'
        root=tkinter.Tk()
        root.title('CopModel: '+symbol)  
        root.geometry('200x200')  
        root.resizable(True, True)  
        msgbx_showinfo(decision, str_buymsg)
        root.mainloop()
        return decision
    elif decision == "XB70":
    # SELL
        price = df_run[symbol+'_C'].iloc[-1]
        str_sellmsg = "SELL "+symbol+f' at {price:,.2f}'
        root=tkinter.Tk()
        root.title('CopModel: '+symbol)  
        root.geometry('200x200')  
        root.resizable(True, True)  
        msgbx_showinfo(decision, str_sellmsg)
        root.mainloop()
        return decision
    
    return decision

# Function to get session hour
def get_session_hour():
    return datetime.datetime.today().hour

# Function to pull realtime data
def readParseHFData(str_path, str_file, str_shtname):
    """
    Import, parse and clean real-time data from excel database file
    """
    tmpdf_head =  pd.read_excel(str_path+str_file,
                                sheet_name = str_shtname,
                          header=None, nrows = 2, skiprows=2)
    assetnames = tmpdf_head.iloc[0,:].dropna()
    repls = {' Index' : '', ' Curncy' : '', ' Index':'', ' Comdty':''}
    names = np.array(
        [s.replace(s[s.find(' '):], 
               repls.get(s[s.find(' '):])) 
             for s in assetnames]
        )
    names = np.array([s.replace('2','1') for s in names])
    # Import data from external excel spreadsheet
    tmpdf = pd.read_excel(str_path+str_file,
                          sheet_name = str_shtname,
                          header=None, index_col=0,
                          engine='openpyxl',
                          skiprows=4).dropna(axis=1)
    try:
        tmpdf.columns = tmpdf_head.iloc[1,:].dropna().drop(0)
    except ValueError as import_err:
        print('ValueError Found\n')
        print(import_err)
        print("\nExplanation: "+\
              f"More columns than expected imported from {str_file}\n")
        print('\nRe-trying...\n')
        tmpdf = pd.read_excel(str_path+str_file,
                              sheet_name = str_shtname,
                              header=None, index_col=0,
                              engine='openpyxl',
                              skiprows=4)
        tmpdf = tmpdf.drop(tmpdf.index[tmpdf.index.map(np.isnan)])
        tmpdf.dropna(axis=1, inplace=True)
        tmpdf.columns = tmpdf_head.iloc[1,:].dropna().drop(0)
    
    tmpdf = tmpdf.drop('Dates',axis=1)
    
    df_cols = []
    for name in names:
        df_cols = df_cols + [name+'_'+t for t in ['O', 'H', 'L', 'C']]
    
    tmpdf.index.name = 'date'
    tmpdf.columns = df_cols
    
    return(tmpdf)

# Function to compute OC returns
def get_df_ret(df_px):
    """
    Open2Close Log-Returns
    """
    idx_cols = np.concatenate(
        (np.where(np.array([s.find('_O') for s in df_px.columns])
                            >=0 )[0], 
                   np.where(np.array([s.find('_C') for s in df_px.columns])
                            >=0 )[0])
                   )
    idx_cols.sort()
    df_logRet = df_px.iloc[:,idx_cols].\
        apply(np.log).diff(axis=1).dropna(axis=1)
    idx_cols_logret = np.where(
        np.array([s.find('_C') for s in df_logRet.columns])>=0 )[0]
    df_logRet.columns = [s.replace('_C','') for s in df_logRet.columns]
    return(df_logRet.iloc[:,idx_cols_logret])

# Function to set position size based on balance pct risk per trade
def get_lotsize(symbol, risk, stop_ticks):
    tick_size = mt5_interface.MetaTrader5.symbol_info(symbol).\
        _asdict()['trade_tick_size']
    pipval = mt5_interface.get_pip_value(symbol)
    balance = mt5_interface.get_account_info()['balance']
    money_risk = balance*risk
    norm_lot = tick_size*money_risk/(pipval*stop_ticks)
    return int(100*norm_lot)/100

# Function to articulate copmodel strategy on given symbol
def algotrade_symbol(symbol, df_run):
    #df_run = get_copmodel_run_fromXL(str_path, filename, symbol, models)
    # Covar name
    y2 = [name.replace('_C','') for name in df_run.columns 
     if '_C' in name and name.replace('_C','') != symbol][0]
    
    # Display run
    print_copmodel_run(df_run)
    
    # Decision making
    decision = assess_copmodel_run_5M_fromXL(df_run, symbol)
    
    # Plot mispricing index
    plot_copmodel_Mt(df_run, y2)
    
    return str(decision)+f' in {symbol}'

# Function to articulate strategy_one
def strategy_one(symbol, models, str_path, filename):
    # Model covar
    y2 = models[symbol]['covar']
    # Model run
    df_run = get_copmodel_run_fromXL(str_path, filename, symbol, models)
    
    # Display run
    print_copmodel_run(df_run)
    
    # Decision making
    assess_copmodel_run_5M_fromXL(df_run, symbol)
    
    # Plot mispricing index
    plot_copmodel_Mt(df_run, y2)
    
    return "Completed"

# Function to articualate strategy via MT5 data feed
def strategy_one_mt5(symbol, timeframe, model, pip_size, lots):
    # Model run
    df_run = get_copmodel_run(symbol, timeframe, model)
    # Display run
    print_copmodel_run(df_run)
    
    # Decision making
    assess_copmodel_run(df_run, symbol, timeframe, pip_size, lots)
    
    # Plot mispricing index
    plot_copmodel_Mt(df_run, model['covar'])
    
    return "Completed"

# Function to articualate strategy via MT5 data feed
def strategy_wM_mt5(symbol, timeframe, model, pip_size, lots):
    # Model run
    df_run = get_copmodel_run_w(symbol, timeframe, model)
    # Display run
    print_copmodel_run(df_run)
    
    # Decision making
    assess_copmodel_run(df_run, symbol, timeframe, pip_size, lots)
    
    # Plot mispricing index
    plot_copmodel_Mt(df_run, model['covar'])
    
    return "Completed"

# Function to test if next timeframe-candle is available    
def isNewCandle(timeframe, df_run):
    # Now
    now_dt = datetime.datetime.today()
    tf_val = mt5_interface.set_query_timeframe(timeframe)
    now_dt_tf_min = int(now_dt.minute/tf_val)*tf_val
    now_dt_tf = datetime.datetime(now_dt.year, now_dt.month, now_dt.day,
                                  now_dt.hour, now_dt_tf_min, 0) - \
        datetime.timedelta(minutes=tf_val)
    # Last candle datetime
    last_dt = df_run.index[-1]
    
    return str(last_dt) < str(now_dt_tf)

# Function to set waiting time until next timeframe-candle
def time2wait_M5():
    dt_now = datetime.datetime.today()
    dt_now_5m = datetime.\
        datetime(dt_now.year, dt_now.month, dt_now.day, dt_now.hour, 
                 np.max(((int(dt_now.minute/5)-1)*5,0)), 0
                 )
    dt_nextT = dt_now_5m + datetime.timedelta(minutes=5*2)
    return dt_nextT - dt_now


# Function to query data for copula
def get_copdata_mt5(symbol, timeframe):
    n_candles_5m = int((datetime.datetime(datetime.datetime.today().year, 
                               datetime.datetime.today().month,
                               datetime.datetime.today().day, 
                               datetime.datetime.today().hour, 
                               np.max(
                            ((int(datetime.datetime.today().minute/5)-1)*5,
                                    0)
                                   ), 0) - \
                        datetime.datetime(datetime.datetime.today().year, 
                               datetime.datetime.today().month,
                               datetime.datetime.today().day,
                               0,0,0) + \
                            datetime.timedelta(minutes = 5)).seconds/(60*5))
    tmpdf = get_and_transform_mt5_data(symbol, timeframe, n_candles_5m)
    # Timezone shift from MT5 to local
    tz_hr_shft = mt5_interface.get_timezone_shift_hour(symbol)
    tmpdf['time'] = tmpdf['time'] - datetime.timedelta(hours=tz_hr_shft)
    return tmpdf

# Function to query last two candles in MetaTrader 5 based upon timeframe
def get_and_transform_mt5_data(symbol, timeframe, number_of_candles):
    # Retrieve the raw data from MT5 platform
    raw_data = mt5_interface.\
        query_historic_data(symbol, timeframe, number_of_candles)
    # Transform raw data into Pandas DataFrame
    df_data = pd.DataFrame(raw_data)
    # Convert the time in seconds into a human readable datetime format
    df_data['time'] = pd.to_datetime(df_data['time'], unit='s')
    cols2keep = ['time', 'open', 'high', 'low', 'close']
    # Return the data frame to the user
    return df_data[cols2keep]

# Function to make decisions based on presented dataframe
def make_decision(df_run):
    prev_rsi, curr_rsi = df_run['RSI_9_M'][-2:].values
    # Look for RSI cross 
    if(prev_rsi < 30 or prev_rsi > 70):
        # Test if cross above lower limit
        if(prev_rsi < 30 and curr_rsi > 30):
            return "XA30"
        # Test if cross below upper limit
        elif(prev_rsi > 70 and curr_rsi < 70):
            return "XB70"
        else:
            return "Still stressed"
    # Stressed but no signal yet
    elif(curr_rsi < 30 or curr_rsi > 70):
        return "Stressed"
    # Default to waiting if no stressed levels foudn in RSI
    else:
        return "DoNothing"

# Function to make decisions based on presented dataframe and open position
def make_decision_position(df_run, position):
    # Open position info
    posdict = position._asdict()
    postype = posdict['type']
    # Model info
    prev_rsi, rsi, curr_rsi = df_run['RSI_9_M'][-3:].values
    # Open position is long
    if postype == 0:
        # Look for RSI XA50
        cond_is_xa50 = prev_rsi < 50 and rsi > 50
        if(cond_is_xa50 or curr_rsi >= 50):
            return "XA50"
        else:
            return "Waiting to rise more"
    # Open position is short
    else:
        # Look for RSI XB50
        cond_is_xb50 = prev_rsi > 50 and rsi < 50
        if(cond_is_xb50 or curr_rsi <= 50):
            return "XB50"
        else:
            return "Waiting to drop more"
    # Default to none
    return None

# Function to create a new order based upon previous analysis
def create_new_order(decision_outcome, pip_size, symbol, lots):
    # Do nothing if outcome is "DoNothing
    if decision_outcome == "DoNothing":
        return
    elif decision_outcome == "XA30":
        # Calculate the order stop_loss
        stop_loss = 0.0
        # Calculate the order take_profit
        take_profit = 0.0
        # Add in an order comment
        comment = "copmod"
        # Send order to place_order function in mt5_interface.py
        mt5_interface.place_mktorder("BUY", symbol, lots, 
                                  stop_loss, take_profit, comment)
        return
    elif decision_outcome == "XB70":
        # Calculate the order stop_loss
        stop_loss = 0.0
        # Calculate the order take_profit
        take_profit = 0.0
        # Add in an order comment
        comment = "copmod"
        # Send order to place_order function in mt5_interface.py
        mt5_interface.place_mktorder("SELL", symbol, lots,
                                  stop_loss, take_profit, comment)
        return
    
# Function to evaluate open order based upon previous analysis
def eval_order(decision_outcome, position, new_sl_pts):
    # Condition from decision outcome
    cond_is_close = decision_outcome=="XA50" or decision_outcome=="XB50"
    if cond_is_close:
        # Close open position if RSI's 50 level is crossed
        posdict = position._asdict()
        mt5_interface.close_order(posdict['symbol'], 
                                         decision_outcome, 
                                         posdict['ticket'])
        return 
    else:
        # Update SL if outcome is "Waiting..."
        update_trailing_stop(order=position, 
                             trailing_stop_pips=new_sl_pts, pip_size=1)
        return

# Function to update trailing stop if needed
def update_trailing_stop(order, trailing_stop_pips, pip_size):
    # Convert trailing_stop_pips into pips
    trailing_stop = trailing_stop_pips * pip_size
    if order[-2] != '' and order[-2] != 'copmod':
        # Determine if Red or Green
        # A Green Position will have a take_profit > stop_loss
        if order[12] > order[11]:
            # If Green, new_stop_loss = current_price - trailing_stop_pips
            new_stop_loss = order[13] - trailing_stop
            # Test to see if new_stop_loss > current_stop_loss
            if new_stop_loss > order[11]:
                print("\nUpdate Stop Loss:")
                # Create updated values for order
                order_number = order[0]
                symbol = order[16]
                # New take_profit will be the difference between new_stop_loss and old_stop_loss added to take profit
                new_take_profit = order[12] + new_stop_loss - order[11]
                print("\t"+new_take_profit)
                # Send order to modify_position
                mt5_interface.\
                    modify_position(order_number=order_number, 
                                    symbol=symbol, 
                                    new_stop_loss=new_stop_loss,
                                    new_take_profit=new_take_profit)
        elif order[12] < order[11]:
            # If Red, new_stop_loss = current_price + trailing_stop_pips
            new_stop_loss = order[13] + trailing_stop
            # Test to see if new_stop_loss < current_stop_loss
            if new_stop_loss < order[11]:
                print("\nUpdate Stop Loss:")
                # Create updated values for order
                order_number = order[0]
                symbol = order[16]
                # New take_profit will be the difference between new_stop_loss and old_stop_loss subtracted from old take_profit
                new_take_profit = order[12] - new_stop_loss + order[11]
                print("\t"+new_take_profit)
                # Send order to modify_position
                mt5_interface.\
                    modify_position(order_number=order_number, 
                                    symbol=symbol, 
                                    new_stop_loss=new_stop_loss,
                                    new_take_profit=new_take_profit)
    elif (order[13] - order[10])*(1,-1)[order[5]] >= trailing_stop:
        if order[5] == 0: # long position
            new_stop_loss = order[13] - trailing_stop
            isUpdate = new_stop_loss > order[11]
        else: # short position
            new_stop_loss = order[13] + trailing_stop
            isUpdate = new_stop_loss < order[11]
        if isUpdate:
            # Update SL
            print("\nUpdate Stop Loss:")
            order_number = order[0]
            symbol = order[16]
            print("\t"+str(new_stop_loss)+"\n")
            # Send order to modify_position
            mt5_interface.\
                modify_position(
                    order_number=order_number, 
                    symbol=symbol, 
                    new_stop_loss=new_stop_loss,
                    new_take_profit=order[10] + 3*trailing_stop)
    else:
        if order[5] == 0: # long position
            new_stop_loss = order[10] - trailing_stop
            isUpdate = new_stop_loss > order[11] or order[11]==0.0
        else: # short position
            new_stop_loss = order[10] + trailing_stop
            isUpdate = new_stop_loss < order[11] or order[11]==0.0 
        if isUpdate:
            # Update SL
            print("\nUpdate Stop Loss:")
            order_number = order[0]
            symbol = order[16]
            print("\t"+str(new_stop_loss)+"\n")
            # Send order to modify_position
            mt5_interface.\
                modify_position(
                    order_number=order_number, 
                    symbol=symbol, 
                    new_stop_loss=new_stop_loss,
                    new_take_profit=order[10]+3*trailing_stop*(1,-1)[order[5]])
                
def main(symbol, timeframe='M5', risk=0.001):
    mt5_interface.MetaTrader5.initialize()
    model = copmodel(symbol, timeframe)
    try:
        df_run = get_copmodel_run(symbol, timeframe, model)
    except KeyError as ke:
        print(ke)
        print('Retrying...')
        time.sleep(20)
        df_run = get_copmodel_run(symbol, timeframe, model)
    today_hour = get_session_hour()
    stop_ticks = 3*mt5_interface.get_atr(symbol,timeframe,64).iloc[-1]
    lots = get_lotsize(symbol, risk, stop_ticks)
    strategy_one_mt5(symbol, timeframe, model, 0, lots)
    while today_hour < 15:
        today_hour = get_session_hour()
        if isNewCandle(timeframe, df_run):
            time.sleep(24)
            df_run = get_copmodel_run(symbol, timeframe, model)
            today_hour = get_session_hour()
            stop_ticks = 3*mt5_interface.get_atr(symbol,timeframe,64).iloc[-1]
            lots = get_lotsize(symbol, risk, stop_ticks)
            strategy_one_mt5(symbol, timeframe, model, 0, lots)
            
def main2(symbol='GBPJPY', timeframe='M5', couple='NZDJPY', risk=0.001):
    mt5_interface.MetaTrader5.initialize()
    model = copmodel_byCouple(symbol, timeframe, couple)
    try:
        df_run = get_copmodel_run(symbol, timeframe, model)
    except KeyError as ke:
        print(ke)
        print('Retrying...')
        time.sleep(20)
        df_run = get_copmodel_run(symbol, timeframe, model)
    today_hour = get_session_hour()
    stop_ticks = 3*mt5_interface.get_atr(symbol,timeframe,64).iloc[-1]
    lots = get_lotsize(symbol, risk, stop_ticks)
    strategy_one_mt5(symbol, timeframe, model, 0, lots)
    while today_hour < 15:
        today_hour = get_session_hour()
        if isNewCandle(timeframe, df_run):
            time.sleep(24)
            df_run = get_copmodel_run(symbol, timeframe, model)
            today_hour = get_session_hour()
            stop_ticks = 3*mt5_interface.get_atr(symbol,timeframe,64).iloc[-1]
            lots = get_lotsize(symbol, risk, stop_ticks)
            strategy_one_mt5(symbol, timeframe, model, 0, lots)