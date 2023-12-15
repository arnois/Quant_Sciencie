# -*- coding: utf-8 -*-
"""
MT5 Interface
Initialization and login to MT5

@author: Arnulf QC
"""
###############################################################################
import MetaTrader5
import pandas as pd
import datetime
###############################################################################
# Function to start Meta Trader 5 (MT5)
def start_mt5(username, password, server, path):
    # Ensure that all variables are the correct type
    uname = int(username) # Username must be an int
    pword = str(password) # Password must be a string
    trading_server = str(server) # Server must be a string
    filepath = str(path) # Filepath must be a string

    # Attempt to start MT5
    if MetaTrader5.initialize(login=uname, password=pword, 
                              server=trading_server, path=filepath):
        print("\nTrading Bot Starting...\n")
        # Login to MT5
        if MetaTrader5.login(login=uname, password=pword, 
                             server=trading_server):
            print("Trading Bot Logged in and Ready to Go!\n")
            return True
        else:
            print("Login Fail\n")
            quit()
            return PermissionError
    else:
        print("\nMT5 Initialization Failed\n")
        print(MetaTrader5.last_error())
        quit()
        return ConnectionAbortedError
    
# Function to shut Meta Trader 5 (MT5)
def end_mt5():
    # Ensure that all variables are the correct type
    MetaTrader5.shutdown()
    
# Function to initialize a symbol on MT5
def initialize_symbols(symbol_array):
    # Get a list of all symbols supported in MT5
    all_symbols = MetaTrader5.symbols_get()
    # Create an array to store all the symbols
    symbol_names = []
    # Add the retrieved symbols to the array
    for symbol in all_symbols:
        symbol_names.append(symbol.name)

    # Check each symbol in symbol_array to ensure it exists
    for provided_symbol in symbol_array:
        if provided_symbol in symbol_names:
            # If it exists, enable
            if MetaTrader5.symbol_select(provided_symbol, True):
                print(f"Sybmol {provided_symbol} enabled")
            else:
                return ValueError
        else:
            return SyntaxError

    # Return true when all symbols enabled
    return True

# Function to get terminal info status
def get_terminal_info():
    return MetaTrader5.terminal_info()._asdict()

# Function to pull trading ccount info
def get_account_info():
    return MetaTrader5.account_info()._asdict()

# Function to compute pip value of given forex pair
def get_pip_value(symbol):
    return MetaTrader5.symbol_info(symbol)._asdict()['trade_tick_value']

# Function to place market order in MT5
def place_mktorder(order_type, symbol, volume, sl, tp, comment):
    # If order type SELL_STOP
    if order_type == "SELL":
        order_type = MetaTrader5.ORDER_TYPE_SELL
        price = MetaTrader5.symbol_info_tick(symbol).bid
    elif order_type == "BUY":
        order_type = MetaTrader5.ORDER_TYPE_BUY
        price = MetaTrader5.symbol_info_tick(symbol).ask
    # Create the request
    request = {
        "action": MetaTrader5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 1000,
        "type_filling": MetaTrader5.ORDER_FILLING_RETURN,
        "type_time": MetaTrader5.ORDER_TIME_GTC,
        "comment": comment
    }
    # Send the order to MT5
    order_result = MetaTrader5.order_send(request)
    # Notify based on return outcomes
    if order_result[0] == 10009:
        print(f"\nTrade for {symbol} successful!")
    else:
        print("\nError placing trade. \n" + \
              f"ErrorCode {order_result[0]}, \nError Details: {order_result}\n"
              )
    return order_result

# Function to place stop limit order on MT5
def place_order(order_type, symbol, volume, price, 
                stop_loss, take_profit, comment):
    # If order type SELL_STOP
    if order_type == "SELL_STOP":
        order_type = MetaTrader5.ORDER_TYPE_SELL_STOP
    elif order_type == "BUY_STOP":
        order_type = MetaTrader5.ORDER_TYPE_BUY_STOP
    # Create the request
    request = {
        "action": MetaTrader5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": round(price, 5),
        "sl": round(stop_loss, 5),
        "tp": round(take_profit, 5),
        "type_filling": MetaTrader5.ORDER_FILLING_RETURN,
        "type_time": MetaTrader5.ORDER_TIME_GTC,
        "comment": comment
    }
    # Send the order to MT5
    order_result = MetaTrader5.order_send(request)
    # Notify based on return outcomes
    if order_result[0] == 10009:
        print(f"\nOrder for {symbol} successful\n")
    else:
        print("\nError placing order. \n" + \
              f"ErrorCode {order_result[0]}, \nError Details: {order_result}\n"
              )
    return order_result

# Function to close open order
def close_order(symbol, comment, ticket):
    # Send order to MT5
    order_result = MetaTrader5.Close(symbol=symbol,
                                     comment=comment,
                                     ticket=ticket)
    # Notify based on return outcomes
    if order_result:
        print(f"\nClosed open trade for {symbol} successful")
    else:
        print("\nError closing trade. \n" + \
              f"ErrorCode {order_result[0]}, "+\
                  f"\nError Details:\n {order_result}"
              )
    return order_result

# Function to close all open positions
def close_all_positions():
    # Fetch all open positions
    positions = get_open_positions()
    # Loop through each position
    for position in positions:
        close_order(position.symbol,'EoD',position.ticket)

# Function to cancel an order
def cancel_order(order_number):
    # Create the request
    request = {
        "action": MetaTrader5.TRADE_ACTION_REMOVE,
        "order": order_number,
        "comment": "Order Removed"
    }
    # Send order to MT5
    order_result = MetaTrader5.order_send(request)
    return order_result

# Function to modify an open position
def modify_position(order_number, symbol, new_stop_loss, new_take_profit):
    # Create the request
    request = {
        "action": MetaTrader5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "sl": new_stop_loss,
        "tp": new_take_profit,
        "position": order_number
    }
    # Send order to MT5
    order_result = MetaTrader5.order_send(request)
    if order_result[0] == 10009:
        return True
    else:
        return False

# Function to convert a timeframe string in MetaTrader 5 friendly format
def set_query_timeframe(timeframe):
    """
    Implement a Pseudo Switch statement. 
    Note that Python 3.10 implements match / case 
    but have kept it this way for backwards integration
    """
    
    if timeframe == "M1":
        return MetaTrader5.TIMEFRAME_M1
    elif timeframe == "M2":
        return MetaTrader5.TIMEFRAME_M2
    elif timeframe == "M3":
        return MetaTrader5.TIMEFRAME_M3
    elif timeframe == "M4":
        return MetaTrader5.TIMEFRAME_M4
    elif timeframe == "M5":
        return MetaTrader5.TIMEFRAME_M5
    elif timeframe == "M6":
        return MetaTrader5.TIMEFRAME_M6
    elif timeframe == "M10":
        return MetaTrader5.TIMEFRAME_M10
    elif timeframe == "M12":
        return MetaTrader5.TIMEFRAME_M12
    elif timeframe == "M15":
        return MetaTrader5.TIMEFRAME_M15
    elif timeframe == "M20":
        return MetaTrader5.TIMEFRAME_M20
    elif timeframe == "M30":
        return MetaTrader5.TIMEFRAME_M30
    elif timeframe == "H1":
        return MetaTrader5.TIMEFRAME_H1
    elif timeframe == "H2":
        return MetaTrader5.TIMEFRAME_H2
    elif timeframe == "H3":
        return MetaTrader5.TIMEFRAME_H3
    elif timeframe == "H4":
        return MetaTrader5.TIMEFRAME_H4
    elif timeframe == "H6":
        return MetaTrader5.TIMEFRAME_H6
    elif timeframe == "H8":
        return MetaTrader5.TIMEFRAME_H8
    elif timeframe == "H12":
        return MetaTrader5.TIMEFRAME_H12
    elif timeframe == "D1":
        return MetaTrader5.TIMEFRAME_D1
    elif timeframe == "W1":
        return MetaTrader5.TIMEFRAME_W1
    elif timeframe == "MN1":
        return MetaTrader5.TIMEFRAME_MN1

# Function to query bulk data from MT5
def query_bulkdata(timeframe, shift_hour):
    # Convert the timeframe into an MT5 friendly format
    mt5_timeframe = set_query_timeframe(timeframe)
    # Get all symbols available
    tpl_symbols_get = MetaTrader5.symbols_get()
    # Pull all tradeables info
    lst_symbls = [[symbInfo.name, symbInfo.trade_tick_size, 
      symbInfo.trade_tick_value, symbInfo.trade_contract_size, 
      symbInfo.volume_min, symbInfo.currency_profit, 
      symbInfo.currency_margin] for symbInfo in tpl_symbols_get]
    lst_colnames = ['symbol', 'tick_size', 'tick_value', 'ct_size', 
             'volume_min', 'ccy_pnl', 'ccy_mrgn']
    total_symbols = pd.\
        DataFrame(lst_symbls, columns=lst_colnames).set_index('symbol')
    # Set maximum bars available to pull
    max_bars = MetaTrader5.terminal_info()._asdict()['maxbars']-1
    # Pull data
    ['O','H','L','C']
    prices = {}
    for symbol in total_symbols.index:
        rates_data = MetaTrader5.copy_rates_from_pos(symbol, 
                                            mt5_timeframe, 1, max_bars)
        if rates_data is None:
            continue
        prices[symbol] = pd.DataFrame(rates_data)[["time", "open", "high", 
                                                   "low", "close"]]
        prices[symbol]['time'] = \
            pd.\
            to_datetime(prices[symbol]['time'], unit='s') - \
                datetime.timedelta(hours=shift_hour)
        newcols = ['time'] + [symbol+'_'+ptype for ptype in ['O','H','L','C']]
        prices[symbol].columns = newcols
    # Merge data into dataframe
    tmpdf = prices[list(prices.keys())[0]]
    for name in list(prices.keys())[1:]:
        tmpdf = tmpdf.merge(prices[name], how = 'left' ,on = 'time')
    # Return dataframe
    return tmpdf.set_index('time')

# Function to query data for current day from MT5
def query_today_data(symbol, timeframe, shift_hour):
    # Convert the timeframe into an MT5 friendly format
    mt5_timeframe = set_query_timeframe(timeframe)
    # Today's date
    today = datetime.date.today()
    # Today's starting and ending datetimes
    dt_now0 = datetime.datetime(today.year, today.month, today.day, 0,0)
    dt_now = datetime.datetime.today()
    dt_now_5m = datetime.datetime(today.year, today.month, today.day, 
                                  dt_now.hour, int(dt_now.minute/5)*5)
    # Number of todays candles in timeframe units
    n_candles = int((dt_now_5m - dt_now0).seconds/(60*5))+2
    # Retrieve data from MT5
    rates = pd.DataFrame(MetaTrader5.\
                       copy_rates_from_pos(symbol, 
                                           mt5_timeframe, 
                                           1, 
                                           n_candles))
    rates['time'] = pd.to_datetime(rates['time'], unit='s') - \
        datetime.timedelta(hours=shift_hour)
    df = rates[["time", "open", "high", "low", "close"]]
    df.columns = ['time'] + [symbol+'_'+ptype for ptype in ['O','H','L','C']]
    return df.set_index('time')
    
    

# Function to query previous candlestick data from MT5
def query_historic_data(symbol, timeframe, number_of_candles):
    # Convert the timeframe into an MT5 friendly format
    mt5_timeframe = set_query_timeframe(timeframe)
    # Add 1 to number of candles
    number_of_candles = number_of_candles + 1
    # Retrieve data from MT5
    rates = MetaTrader5.copy_rates_from_pos(symbol, mt5_timeframe, 
                                            1, number_of_candles)
    return rates

# Function to get ATR
def get_atr(symbol, timeframe, atr_length):
    # Convert the timeframe into an MT5 friendly format
    mt5_timeframe = set_query_timeframe(timeframe)
    # Add 1 to number of candles
    number_of_candles = atr_length*3
    atr = pd.DataFrame(MetaTrader5.\
                       copy_rates_from_pos(symbol, 
                                           mt5_timeframe, 
                                           1, 
                                           number_of_candles)).\
        ta.atr(length=atr_length, mamode='ema')
    return atr

# Function to retrieve all open orders from MT5
def get_open_orders():
    orders = MetaTrader5.orders_get()
    order_array = []
    for order in orders:
        order_array.append(order[0])
    return order_array

# Function to retrieve all open positions
def get_open_positions_s(symbol):
    # Get position objects
    positions = MetaTrader5.positions_get(group=symbol)
    # Return position objects
    return positions

# Function to retrieve all open positions
def get_open_positions():
    # Get position objects
    positions = MetaTrader5.positions_get()
    # Return position objects
    return positions

# Function to determime timezone shift in hours from MT5 to local
def get_timezone_shift_hour(symbol):
    # MT5 datetime
    datetime_mt5 = pd.to_datetime(
        MetaTrader5.symbol_info(symbol)._asdict()['time'], 
        unit = 's')
    # Local datetime
    datetime_local = datetime.datetime.today()
    # Timezones difference
    tz_diff = datetime_mt5 - datetime_local
    # Timezone hour-shift
    tz_hr_shft = int(round(tz_diff.seconds/(60*60),0))
    return tz_hr_shft

# Function to get EoD PnL by traded asset
def get_eod_total_profit_symbol(symbol):
    # Reverse shift
    hourshift = get_timezone_shift_hour(symbol)
    # Timehandles
    today = datetime.datetime.today()
    today_eod = datetime.datetime(today.year, today.month, today.day,16,0,0)
    to_date = today_eod + datetime.timedelta(hours=hourshift)
    from_date = to_date - datetime.timedelta(days=1)
    # Deals data
    str_grp = r'*'+symbol.replace('USD','')+r'*'
    deals = MetaTrader5.history_deals_get(from_date, to_date, group=str_grp)
    
    # PnL
    total_pnl = 0
    for deal in deals:
        tmpdict = deal._asdict()
        total_pnl+=tmpdict['profit']
        
    return total_pnl

# Function to get EoD PnL
def get_eod_total_profit():
    # Reverse shift
    hourshift = get_timezone_shift_hour('USDMXN')
    # Timehandles
    today = datetime.datetime.today()
    today_eod = datetime.datetime(today.year, today.month, today.day,16,0,0)
    to_date = today_eod + datetime.timedelta(hours=hourshift)
    from_date = to_date - datetime.timedelta(days=1)
    # Deals data
    deals = MetaTrader5.history_deals_get(from_date, to_date)
    
    # PnL
    total_pnl = 0
    for deal in deals:
        tmpdict = deal._asdict()
        total_pnl+=tmpdict['profit']
        
    return total_pnl



