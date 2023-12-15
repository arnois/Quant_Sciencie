# -*- coding: utf-8 -*-
"""
MXN Analysis
Frequency: 5M

@author: jquintero
"""
# In[Installs]
# pip install finplot
# pip install ta

# In[Modules]
import pandas as pd
import mplfinance as mpf
import ta 
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

# In[DataImport]
tmp_fname = 'FX_RATES_5M.xlsx'
tmp_pname = 'H:\\Trading\\db\\'
tmp_path = tmp_pname + tmp_fname
data = pd.read_excel(tmp_path)

# In[DataMgmt1]
tmpdf = data.drop([0,1])
tmpdf = tmpdf.iloc[:,:5]
tmpdf.columns = tmpdf.iloc[0]
tmpdf = tmpdf.drop(2)
tmpdf['Date'] = pd.to_datetime(tmpdf['Dates'])
tmpdf = tmpdf.dropna()
tmpdf.set_index(tmpdf['Date'], inplace=True)
df = tmpdf.copy()
df.drop(['Date','Dates'],axis=1, inplace=True)

del [tmp_fname,tmp_pname,tmp_path,tmpdf]

# In[DataMgmt2]
### filter out weekends
df1 = df[df.index.day_of_week<5]
### filter out non-trading hours
trd_st_hr = "00:00:00"
trd_ed_hr = "16:00:00"
df2 = df1.between_time(trd_st_hr,trd_ed_hr)
df2 = df2.astype(float)
del [trd_ed_hr, trd_st_hr, df, data, df1]

# In[DataExport]
"""
df2.to_csv(r'H:\\Trading\\db\\usdmxn_5M.txt')
"""
# In[DataExportImport]
tmpdf2 = pd.read_csv('H:\\Trading\\db\\usdmxn_5M.txt')
tmpdf2['Date'] = pd.to_datetime(tmpdf2['Date'])
tmpdf2.set_index(tmpdf2['Date'], inplace=True)
tmpdf2.drop('Date', axis=1, inplace=True)
df2 = tmpdf2.copy()
del tmpdf2

# In[UDF]
##############################################################################
### -*- Technical Indicators -*- ###
##############################################################################
def addTA_Volatility(df, ATR_window = 10, p_smoothing = 3):
    df_atr = pd.DataFrame()
    atr = ta.volatility.AverageTrueRange(df.High, df.Low, df.Close, 
                                     window = ATR_window )
    atr_sma = ta.trend.ema_indicator(atr.average_true_range(), p_smoothing)
    df_atr['ATR'] = atr.average_true_range()
    df_atr['ATR_sgnl'] = atr_sma
    return df_atr

def addTA_Channel(df, window = 55):
    df_chnl = pd.DataFrame()
    df_chnl['HH'] = df['High'].shift().rolling(window).max()
    df_chnl['LL'] = df['Low'].shift().rolling(window).min()
    return df_chnl

def addTA_emas(Close, p_fast = 9, p_mid = 21, p_slow = 55):
    emaFast = (ta.trend.EMAIndicator(Close,p_fast)).ema_indicator()
    emaFast.name = 'emaF'
    emaMid = (ta.trend.EMAIndicator(Close,p_mid)).ema_indicator()
    emaMid.name = 'emaM'
    emaSlow = (ta.trend.EMAIndicator(Close,p_slow)).ema_indicator()
    emaSlow.name = 'emaS'
    df_emas = pd.concat([emaFast,emaMid,emaSlow],axis=1)
    return df_emas

def addTA_rsi(Close, p = 9):
    df_rsi = pd.DataFrame()
    df_rsi['rsi'] = ta.momentum.rsi(Close, p)
    return df_rsi

def addTA_tsi(Close, p_smooth = 7, p = 55):
    df_tsi = pd.DataFrame()
    df_tsi['tsi'] = ta.momentum.tsi(Close,window_fast=p_smooth,window_slow=p)
    return df_tsi
    
def addTA(df, window = 55, atr_wdw = 55, atr_smooth = 5):
    dfm = df.copy()
    ### Volatility: 10p-ATR
    df_atr = addTA_Volatility(df, ATR_window=atr_wdw, p_smoothing=atr_smooth)
    dfm['ATR'] = df_atr.ATR
    dfm['ATR_sgnl'] = df_atr.ATR_sgnl
    ### Channel: 55p
    df_chnl = addTA_Channel(df, window)
    dfm['HH'] = df_chnl.HH
    dfm['LL'] = df_chnl.LL
    ### EMAs: (9,21,32)
    df_emas = addTA_emas(df.Close)
    dfm = dfm.merge(df_emas,left_index=True,right_index=True)
    ### Momentum: RSI, TSI
    df_rsi = addTA_rsi(df.Close)
    df_tsi = addTA_tsi(df.Close)
    mmtm = df_rsi.merge(df_tsi, left_index=True, right_index=True)
    dfm = dfm.merge(mmtm,left_index=True,right_index=True)
    dfm.dropna(inplace=True)
    dfm.ATR[dfm.ATR==0] = dfm.ATR.loc[dfm.loc[dfm.ATR!=0].index[0]:].mean()
    return dfm

##############################################################################
### -*- TradingChart Plotting -*- ###
##############################################################################
def finplot(df, startDate = '2020-11-05 06:20', endDate = '2020-11-05 14:00'):
    ### fin plot
    sdt = startDate
    edt = endDate
    ti_chnl = ['LL','HH']
    apdict = [mpf.make_addplot(df[ti_chnl].loc[sdt:edt], linestyle = 'dashed', 
                           color = 'darkcyan'),
          mpf.make_addplot(df['emaS'].loc[sdt:edt], color = 'orange'),
          mpf.make_addplot(df['rsi'].loc[sdt:edt], panel = 1, 
                           color = 'blue'),
          mpf.make_addplot(df['tsi'].loc[sdt:edt], panel = 2, 
                           color = 'darkcyan'),
          mpf.make_addplot(df['ATR'].loc[sdt:edt], panel = 3, 
                           color = 'black'),
          mpf.make_addplot(df['ATR_sgnl'].loc[sdt:edt], panel = 3, 
                           color = 'red', linestyle = 'dashed')
          ]
    mpf.plot(df.loc[sdt:edt], addplot = apdict, type = 'candle', 
         style = 'classic', figscale = 2, panel_ratios = (10,1,1.7,1.8))
    mpf.show()

##############################################################################
### -*- TradingSystem SetUps -*- ###
##############################################################################
def volatBrkt_entry(df):
    ### Volatility
    volBrkt_up = df.ATR > df.ATR_sgnl
    volBrkt_down = df.ATR.shift() < df.ATR_sgnl.shift()
    volBrkt = volBrkt_down & volBrkt_up
    ### Entry Signals
    df['sgnl'] = 0
    #### Buy
    sgnl_buy = volBrkt & (df.Close > df.HH)
    df.loc[sgnl_buy,'sgnl'] = 1
    #### Sell
    sgnl_sell = volBrkt & (df.Close < df.LL)
    df.loc[sgnl_sell,'sgnl'] = -1
    
    return df

##############################################################################
### -*- TradingSystem Exits -*- ###
##############################################################################
#--- function to assess a trade worst path
def maxAdverseExcursion(df, dt_entry, dt_idx, curr_pos):
    px_entry = df.loc[dt_entry,'Close']
    dt_init = (pd.date_range(dt_entry,periods=1)\
               + pd.Timedelta(minutes = 5))[0]
    if curr_pos > 0:    # look for lowest price against trade
        lows = df.loc[dt_init:dt_idx,'Low']
        mae_px = np.min(lows)
        mae_dt = df.loc[df.Low == mae_px].index[0]
        mae = np.max(px_entry - mae_px,0)
    else:               # look for highest price against trade
        highs = df.loc[dt_init:dt_idx,'High']
        mae_px = np.max(highs)
        mae_dt = df.loc[df.High == mae_px].index[0]
        mae = np.max(mae_px - px_entry, 0)
    return mae_px, mae, mae_dt

#--- function to assess a trade best path
def maxSuccessExcursion(df, dt_entry, dt_idx, curr_pos):
    px_entry = df.loc[dt_entry,'Close']
    dt_init = (pd.date_range(dt_entry,periods=1)\
               + pd.Timedelta(minutes = 5))[0]
    if curr_pos > 0:    # look for highest price supporting trade
        highs = df.loc[dt_init:dt_idx,'High']
        mse_px = np.max(highs)
        mse_dt = df.loc[df.High == mse_px].index[0]
        mse = np.max(mse_px - px_entry,0)
    else:               # look for lowest price supporting trade
        lows = df.loc[dt_init:dt_idx,'Low']
        mse_px = np.min(lows)
        mse_dt = df.loc[df.Low == mse_px].index[0]
        mse = np.max(px_entry - mse_px, 0)
    return mse_px, mse, mse_dt

#--- function to assess open trades: check if stopped or update sl
def evalOpenTrade(df, dt_idx, n_trade, curr_pos, entry_px, curr_sl, iSL):
    # check if stopped
    if curr_pos > 0:    # eval lows
        tmp_low = df.loc[dt_idx, 'Low']
        stopped = tmp_low <= curr_sl
    else:               # eval highs
        tmp_high = df.loc[dt_idx,'High']
        stopped = tmp_high >= curr_sl
    if stopped:     # close trade
        # cpnl = np.round(curr_pos*(curr_sl - entry_px),5)
        strExitCols = ['Exit','ExDate']
        tupExitCols = curr_sl, dt_idx
        return stopped, strExitCols, tupExitCols
    else:           # update sl
        # 2.38x ATR
        tmp_atr = df.loc[dt_idx,'ATR']*2.38
        # look for 21-p date
        tmp_sgnl_sl_dt = pd.date_range(dt_idx,periods=1)\
                            - pd.Timedelta(minutes=21*5)
        if curr_pos > 0:    # 21-p low
            tmp_sgnl_sl = df.loc[tmp_sgnl_sl_dt[0]:dt_idx,'Low'].min()
            tmp_newExtreme = df.loc[dt_idx,'High']
            atrTSL = tmp_newExtreme - tmp_atr*curr_pos
            atrTSL = np.max([atrTSL,curr_sl])
            midTSL = np.max([atrTSL,tmp_sgnl_sl])
        else:               # 21-p high
            tmp_sgnl_sl = df.loc[tmp_sgnl_sl_dt[0]:dt_idx,'High'].max()
            tmp_newExtreme = df.loc[dt_idx,'Low']
            atrTSL = tmp_newExtreme - tmp_atr*curr_pos
            atrTSL = np.min([atrTSL,curr_sl])
            midTSL = np.min([atrTSL,tmp_sgnl_sl])
        csl = midTSL
        return stopped, csl

##############################################################################
### -*- TradingSystem Test -*- ###
##############################################################################
#--- blotter
def getBlotter(df, disl = 0.0362, modeMAE = False):
    delta_iSL = disl # (0.0362,0.0656,0.0745)
    n_trade = 0
    curr_pos = 0
    entry_px = 0.0
    curr_sl = 0.0
    curr_idx = 0
    cols_bttr = ['tradeID','Entry','Pos','initSL','Exit','ExDate']
    bttr = pd.DataFrame(columns = cols_bttr)
    for index, row in df.iterrows():
        tmp_sgnl = row.sgnl
        
        if tmp_sgnl != 0:
            # entry signal triggered
            tmp_px = df.loc[index,'Close']
            if curr_pos == 0:   # new trade signal
                n_trade += 1
                curr_idx = index
                # trade side
                curr_pos = tmp_sgnl
                entry_px = tmp_px
                # look for 21-p date
                tmp_sgnl_sl_dt = pd.date_range(index,periods=1)\
                        - pd.Timedelta(minutes=21*5)
                # sl
                if modeMAE:     # init SL determined by MAE analysis
                    curr_sl = entry_px - delta_iSL*curr_pos
                else:           # init SL as rule-defined
                    atrSLd = df.loc[curr_idx, 'ATR']*2.7
                    if curr_pos > 0:    # 21-p low
                            tmp_sgnl_sl =\
                                df.loc[tmp_sgnl_sl_dt[0]:index,'Low'].min()
                    else:               # 21-p high
                            tmp_sgnl_sl =\
                                df.loc[tmp_sgnl_sl_dt[0]:index,'High'].max()
                    initSL1 = np.round(tmp_sgnl_sl,5)
                    initSL2 = entry_px - atrSLd*curr_pos
                    curr_sl = np.mean([initSL1, initSL2])
                # blotter entries
                bttr = bttr.append(pd.DataFrame(index=[index]))
                strEntryCols = ['tradeID','Entry','Pos','initSL']
                tupEntryCols = n_trade,entry_px,int(curr_pos),curr_sl
                bttr.loc[curr_idx,strEntryCols] = tupEntryCols
            else:   # update open trade; new signal, but current trade open
                isl = bttr.loc[curr_idx, 'initSL']
                openTradeRes = evalOpenTrade(df, index, n_trade, curr_pos, 
                                             entry_px, curr_sl, isl)
                if openTradeRes[0]:     # stopped
                    bttr.loc[curr_idx, openTradeRes[1]] = openTradeRes[2]
                    curr_pos, entry_px, curr_sl, curr_idx = np.zeros(4)
                else:
                    curr_sl = openTradeRes[1]
        elif curr_pos != 0:   # eval open trade
            isl = bttr.loc[curr_idx, 'initSL']
            openTradeRes = evalOpenTrade(df, index, n_trade, curr_pos, 
                                             entry_px, curr_sl,isl)
            if openTradeRes[0]:     # stopped
                bttr.loc[curr_idx, openTradeRes[1]] = openTradeRes[2]
                curr_pos, entry_px, curr_sl, curr_idx = np.zeros(4)
            else:
                curr_sl = openTradeRes[1]
    bttr['PnL'] = (bttr.Exit - bttr.Entry)*bttr.Pos
    bttr['pipR'] = (bttr.Entry - bttr.initSL) * bttr.Pos
    bttr['R'] = bttr.PnL/bttr.pipR            
    return bttr

#--- trades max adverse and success excurisons
def getExcursions_time(df, bttr, excursion_time = 240):
    cols_excs = ['MAE','MSE']
    excs = pd.DataFrame(columns = cols_excs)
    for idx, row in bttr.iterrows():
        dt_delta_min = (pd.date_range(idx,periods=1)\
               + pd.Timedelta(minutes = excursion_time))[0]
        curr_pos = row.Pos
        mae = maxAdverseExcursion(df, idx, dt_delta_min, curr_pos)
        mse = maxSuccessExcursion(df, idx, dt_delta_min, curr_pos)
        isValid = (mae[2] != mse[2]) and mae[1] > 0
        if isValid:
            excs = excs.append(pd.DataFrame(index=[idx]))
            tupEntryCols = mae[1], mse[1]
            excs.loc[idx] = tupEntryCols
    excs['mseR'] = excs.MSE / excs.MAE
    return excs.astype(float)

##############################################################################
### -*- TradingSystem R-distribution sim -*- ###
##############################################################################
#--- random samples generator from empirical distribution
def empirical_sample(ecdf, size):
    u = np.random.uniform(0,1,size)
    ecdf_x = np.delete(ecdf.x,0)
    sample = []
    for u_y in u:
        idx = np.argmax(ecdf.y >= u_y)-1
        e_x = ecdf_x[idx]
        sample.append(e_x)
    return pd.Series(sample,name='emp_sample')

#--- R paths generator from empirical distribution
def sim_path_R(ecdf, sample_size=1000, paths_size=1000):
    runs = []
    for n in range(paths_size):
        run = empirical_sample(ecdf,sample_size)
        run.name = run.name + f'{n+1}'
        runs.append(run.cumsum())
    df_runs = pd.concat(runs, axis = 1)
    df_runs.index = df_runs.index + 1
    df_runs = pd.concat([pd.DataFrame(np.zeros((1,paths_size)),
                                      columns=df_runs.columns),
               df_runs])
    return df_runs

#--- MC sims of R-distribution
def sim_cumR(bttr, df, n_paths = 1000):
    recdf = ECDF(bttr['R'])
    N_sim = n_paths
    N_sample =\
        int(bttr.shape[0]*12*30/(df.tail(1).index[0]-df.head(1).index[0]).days)
    runs_myR = sim_path_R(recdf, sample_size=N_sample, paths_size=N_sim)
    runs_myR.plot(title=f'Cumulative R Simulations\n'\
                  f'N(paths)={N_sim}, N(sample) ={N_sample}',
                  legend=False)
    plt.grid(alpha=0.5, linestyle='--')
    #--- simulation results
    simmean = runs_myR.mean(axis=1)
    simstd = runs_myR.std(axis=1)
    simqtl = runs_myR.quantile(q=(0.025,1-0.025),axis=1).T
    simres = [simmean, simmean-2*simstd, simmean+2*simstd, 
              simqtl, bttr['R'].reset_index(drop=True).cumsum()]
    avgsim = pd.concat(simres,axis=1)
    avgsim.columns = ['Avg Path','LB-2std','UB-2std',
                      f'{2.5}%-q',f'{97.5}%-q','Current']
    avgsim.plot(title='Cumulative R Simulations\nMean path\n'\
                f'N(paths)={N_sim}, N(sample) ={N_sample}', 
                style=['-','--','--','--','--'])
    plt.grid(alpha=0.5, linestyle='--')
    return None

# In[Techs]
df3 = addTA(df2,55,32,32) # meanR = 0.0527
df3 = addTA(df2,55,16,32) # meanR = 0.0486

finplot(df3,'2020-11-04 06:05','2020-11-04 16:00')
finplot(df3,'2021-07-06 00:00','2021-07-06 12:00')


# In[Set ups]
"""
\\--- Volatility Breakout System: 
Entry:
    Enter long when an up/down channel-breakout is marked by the close with
    ATR up-xovers it's (ergodic) signal.
    Channel-Breakout := Close > UB or Close < LB
Exits:
    1R (Initial SL): mu (21-period low/high for buy/sell, 2.7atr)
    Trailing SL: 21-period low/high for buy/sell or 2.38atr
Pos.Sizing: 
    ---pending---
"""
df3 = volatBrkt_entry(df3)

### Exits
sgnl_dt = df3.loc[df3['sgnl']!=0].index[0]
sgnl_ = df3.loc[sgnl_dt,'sgnl'] 
sgnl_px = df3.loc[sgnl_dt,'Close']
sgnl_sl_dt = pd.date_range(sgnl_dt, periods = 1) - pd.Timedelta(minutes=21*5)
sgnl_sl = df3.loc[sgnl_sl_dt[0]:sgnl_dt,'Low'].min()
sgnl_R = (sgnl_px - sgnl_sl)*sgnl_
del [sgnl_dt,sgnl_,sgnl_px,sgnl_sl_dt,sgnl_sl,sgnl_R]

"""
n_trade = 0
curr_pos = 0
entry_px = 0.0
curr_sl = 0.0
curr_idx = 0
cols_bttr = ['tradeID','Entry','Pos','initSL','Exit','ExDate']
bttr = pd.DataFrame(columns = cols_bttr)
for index, row in df3.iterrows():
# for index, row in df3.loc[sdt:edt].iterrows():
    tmp_sgnl = row.sgnl
    
    if tmp_sgnl != 0:
        # entry signal triggered
        tmp_px = df3.loc[index,'Close']
        if curr_pos == 0:   # new trade signal
            n_trade += 1
            curr_idx = index
            # trade side
            curr_pos = tmp_sgnl
            entry_px = tmp_px
            # look for 21-p date
            tmp_sgnl_sl_dt = pd.date_range(index,periods=1)\
                    - pd.Timedelta(minutes=21*5)
            # sl 
            atrSLd = df3.loc[curr_idx, 'ATR']*2.7
            if curr_pos > 0:    # 21-p low
                    tmp_sgnl_sl = df3.loc[tmp_sgnl_sl_dt[0]:index,'Low'].min()
            else:   # 21-p high
                    tmp_sgnl_sl = df3.loc[tmp_sgnl_sl_dt[0]:index,'High'].max()
            initSL1 = np.round(tmp_sgnl_sl,5)
            initSL2 = entry_px - atrSLd*curr_pos
            curr_sl = np.mean([initSL1, initSL2])
            # blotter entries
            print(f'New Trade. Date: {index}\n\tEntry: {entry_px}, '\
                  f'Side: {curr_pos}, SL: {curr_sl}, ID: A00{n_trade}\n')
            bttr = bttr.append(pd.DataFrame(index=[index]))
            strEntryCols = ['tradeID','Entry','Pos','initSL']
            tupEntryCols = n_trade,entry_px,int(curr_pos),curr_sl
            bttr.loc[curr_idx,strEntryCols] = tupEntryCols
        else:   # new signal, but current trade open, so update only
            isl = bttr.loc[curr_idx, 'initSL']
            openTradeRes = evalOpenTrade(df3, index, n_trade, curr_pos, 
                                         entry_px, curr_sl, isl)
            if openTradeRes[0]:     # stopped
                bttr.loc[curr_idx, openTradeRes[1]] = openTradeRes[2]
                curr_pos, entry_px, curr_sl, curr_idx = np.zeros(4)
            else:
                curr_sl = openTradeRes[1]
    elif curr_pos != 0:   # eval open trade
        isl = bttr.loc[curr_idx, 'initSL']
        openTradeRes = evalOpenTrade(df3, index, n_trade, curr_pos, 
                                         entry_px, curr_sl,isl)
        if openTradeRes[0]:     # stopped
            bttr.loc[curr_idx, openTradeRes[1]] = openTradeRes[2]
            curr_pos, entry_px, curr_sl, curr_idx = np.zeros(4)
        else:
            curr_sl = openTradeRes[1]
"""
bttr = getBlotter(df3)
bttr['PnL'] = (bttr.Exit - bttr.Entry)*bttr.Pos
bttr['pipR'] = (bttr.Entry - bttr.initSL) * bttr.Pos
bttr['R'] = bttr.PnL/bttr.pipR
bttr.R.astype(float).describe()
minExpR = np.ceil(10*(1-np.mean(bttr.R>0))/np.mean(bttr.R>0))/10

# In[R Multiples Stats]
rhist = bttr['R'].plot.hist(title='R-Multiples', density=True)
cumr = bttr['R'].cumsum().plot(title='Cumulative R-Multiples', 
                               color='darkcyan')

# In[R-Paths Sim]
sim_cumR(bttr,df3)





