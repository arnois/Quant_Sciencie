"""
###############################################################################
MT5 Trading Bot
Copulae models for trading USDMXN
###############################################################################
:: Copula Models ::
    copmodels = {
        "week": "8"
        "copmodels": ""
    }

:: settings ::
    
   settings = {
    "username": "5010559598",
    "password": "zx0osroa",
    "server": "MetaQuotes-Demo",
    "mt5Pathway": "C:/Program Files/MetaTrader 5/terminal64.exe",
    "symbols": ["USDMXN"],
    "timeframe": "M5",
    "pip_size": 0.0001,
    "risk_pctbalance": 0.005
    }
   
   with open(r'H:\Python\mt5_ea_copmod\settings.json', 'w') as fp:
    json.dump(settings, fp)
    
:: settings ::
###############################################################################
@author: Arnulf QC
@email: arnulf.q@gmail.com
"""
###############################################################################
# MODULES
#import pandas as pd
import json
import os, sys
sys.path.append(r'H:\Python\mt5_ea_copmod\\')
import mt5_interface # if issue >> :: >> pip install numpy --upgrade
import copstrat
import time
#import xlwings as xw
import datetime
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
# Function to set position size based on balance pct risk per trade
def get_lotsize(symbol, risk, stop_ticks):
    tick_size = mt5_interface.MetaTrader5.symbol_info(symbol).\
        _asdict()['trade_tick_size']
    pipval = mt5_interface.get_pip_value(symbol)
    balance = mt5_interface.get_account_info()['balance']
    money_risk = balance*risk
    norm_lot = tick_size*money_risk/(pipval*stop_ticks)
    return int(100*norm_lot)/100
###############################################################################
# GLOBVARS
###############################################################################
# Paths to models and resources
filepath = r'C:\Users\jquintero\db\fut_copmod.xlsx'
str_path = r'C:\Users\jquintero\db'
week = datetime.datetime.today().isocalendar().week
str_modelname = r'H:\Python\models\copmodels_'+r'w'+str(week)+r'.pickle'
# Models for each asset
# models = pd.read_pickle(str_modelname)
# Excel session for database
#wb = xw.Book(filepath)
#wb_test = wb.sheets['test']
###############################################################################
# MAIN FUNCTION
###############################################################################
stop_hour = 14
if __name__ == '__main__':
    # Project settings
    import_filepath = r'H:\Python\mt5_ea_copmod\settings.json'
    project_settings = get_project_settings(import_filepath)
    # Start MT5
    mt5_interface.start_mt5(project_settings["username"], 
                            project_settings["password"], 
                            project_settings["server"],
                           project_settings["mt5Pathway"])
    # Init trading symbol
    mt5_interface.initialize_symbols(project_settings["symbols"])
    # Select symbol to run strategy on
    symbol_for_strategy = project_settings['symbols'][0]
    
    # Set Trading Model
    ### Copula model by association tests
    #model = copstrat.copmodel(symbol_for_strategy, 
    #                          project_settings['timeframe'])
    ### Copula model by manual selection
    model = copstrat.copmodel_byCouple(symbol_for_strategy, 
                              project_settings['timeframe'],'AUDUSD')
    ### Copula model by prev-trained model
    #path_model = r'H:\Python\models\copmodel_USDMXN.pickle'
    #model = copstrat.pd.read_pickle(path_model)
    
    # Set up a previous time variable
    previous_time = 0
    # Set up a current time variable
    current_time = 0
    # Session hour
    today_hour = copstrat.get_session_hour()
    # Start a while loop to poll MT5
    while today_hour < stop_hour:
        # Update session hour
        today_hour = copstrat.get_session_hour()
        # Retrieve the current candle data
        candle_data = mt5_interface.\
            query_historic_data(symbol=symbol_for_strategy,
                                timeframe=project_settings['timeframe'], 
                                number_of_candles=1)
        # Extract the timeframe data
        current_time = candle_data[0][0]
        # Compare against previous time
        if current_time != previous_time:
            # Notify user
            print("\nNew Candle")
            # Update vol measure
            lastATR = mt5_interface.get_atr(symbol_for_strategy, 
                                            project_settings['timeframe'], 
                                            64).iloc[-1]
            lastATR_pips = lastATR/project_settings['pip_size']
            # Stop measure
            stop_ticks = 3*lastATR
            # Pos sizing according to risk and vol measure
            lot_size = get_lotsize(symbol_for_strategy, 
                                   project_settings['risk_pctbalance'], 
                                   stop_ticks)
            # Update previous time
            previous_time = current_time
            # Retrieve previous orders
            orders = mt5_interface.get_open_orders()
            # Cancel orders
            for order in orders:
                mt5_interface.cancel_order(order)
            
            # CopMod strategy on selected symbol
            copstrat.strategy_one_mt5(symbol_for_strategy, 
                                      project_settings['timeframe'], 
                                      model, 
                                      project_settings['pip_size'], 
                                      lot_size)
            # GreenWolf Strategy - CopMod on wiegthed M
            #copstrat.strategy_wM_mt5(symbol_for_strategy, 
            #                         project_settings['timeframe'], 
            #                         model, 
            #                         project_settings['pip_size'], 
            #                         lot_size)
        else:
            # Get positions
            positions = mt5_interface.get_open_positions()
            # Pass positions to update_trailing_stop
            for position in positions:
                # Symbol name
                s_name = position._asdict()['symbol']
                # Vol measure
                lastATR = mt5_interface.get_atr(s_name, 
                                                project_settings['timeframe'], 
                                                64).iloc[-1]
                # Symbol digits
                s_dig = mt5_interface.\
                    MetaTrader5.symbol_info(s_name)._asdict()['digits']
                # Update SL
                new_sl_pts = round(lastATR*3, s_dig)
                copstrat.\
                    update_trailing_stop(
                        order=position, 
                        trailing_stop_pips=new_sl_pts,
                        pip_size=1)
        # Waiting till next candle arises
        n_secs2wait1 = min(copstrat.time2wait_M5().seconds+3,300)
        print(f"Sleeping {int(n_secs2wait1)} secs . . .")
        slpuntl = datetime.datetime.today() +\
                      datetime.timedelta(seconds=n_secs2wait1)
        print("until {:02d}:{:02d}:{:02d}".\
              format(slpuntl.hour, slpuntl.minute, slpuntl.second))
        time.sleep(int(n_secs2wait1))

    # End trading session
    # Get positions
    positions = mt5_interface.get_open_positions()
    if len(positions) > 0:
        mt5_interface.close_all_positions()
    # PnL
    eod_pnl = mt5_interface.get_eod_total_profit()
    print(f'\nPnL: {eod_pnl:.2f} USD\n')
    # Shutdown MT5 platform
    mt5_interface.end_mt5()
    os.system("taskkill /f /im terminal64.exe")

# PnL Assessment
if __name__ != '__main__':
    today = datetime.datetime.today()
    from_date = datetime.datetime(2023,1,1,1,0)
    to_date = datetime.datetime(today.year, 
                                today.month, 
                                today.day,16,0,0) +\
        datetime.timedelta(hours=9)
    deals = mt5_interface.MetaTrader5.history_deals_get(from_date, to_date)
    # USDMXN trading results
    tmplst = []
    for deal in deals:
        tmpdict = deal._asdict()
        tmpsymb = tmpdict['symbol']
        if tmpsymb == 'NZDUSD': # 'NZDJPY', 'NZDUSD', 'USDCHF', 'USDJPY', 'USDSEK'
            tmpentry = tmpdict['entry']
            tmpvol = tmpdict['volume']
            tmppnl = tmpdict['profit']
            tmplst.append([tmppnl, tmpvol, tmpentry])
    df_res = copstrat.pd.DataFrame(tmplst,columns=['profit','volume','entry'])
    df_res[df_res['entry']!=0].describe()
    df_res[df_res['entry']!=0]['profit'].cumsum().plot()
