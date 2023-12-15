"""
###############################################################################
Copulae models for trading CME Futures
###############################################################################

Statistical arbitrage (algorithmic) trading strategy in diverse asset class
futures. The main idea is to capitalize on strong inconsistent movements
between associated pairs in short periods of time, being no more than 1 hour.

@author: Arnulf QC
@email: arnulf.q@gmail.com
"""
###############################################################################
# MODULES
###############################################################################
import sys
sys.path.append(r'H:\Python\mt5_ea_copmod\\')
import copstrat
import xlwings as xw
###############################################################################
# GLOBVARS
###############################################################################
# Paths to models and resources
filename = 'fut_copmod'
str_path = r'C:\Users\jquintero\db'
week = copstrat.datetime.datetime.today().isocalendar().week
str_modelname = r'H:\Python\models\copmodels_'+r'w'+str(week)+r'.pickle'
# Control
stop_hour = 14
# Futures names
futures = ['ES1', 'NQ1']
###############################################################################
# UDF
###############################################################################
# Function to look for excel file within all sessions
def fetch_xlBook(str_file):
    wb = xw.Book(str_file)
    xl_sess_id = xw.apps.keys()
    for pid in xl_sess_id:
        try:
            wb = xw.apps[pid].books(str_file)
            xl_pid = pid 
            print(f'{str_file} found in {pid} XL session')
            break
        except:
            print(f'{str_file} not in {pid} XL session')
    
    return xl_pid, wb
###############################################################################
# Init Models
###############################################################################
# Trading Week Init
def init_models():
    datapath, dataname, savepath = r'H:\db', 'data_5m', r'H:\Python\models'
    str_file = r'\data_5m_y2023.xlsx'
    symbols = ['NQ1','ES1','TU1','FV1','TY1','RX1','CL1','GC1','USDMXN']
    n_skipRows = 2e4
    str_dbpath = r"H:\db\data_5m.parquet"
    copstrat.update_data_5m(str_path, str_file, 'data', 
                            n_skipRows, str_dbpath)
    copstrat.bulkset_copmodel_5M_fromDotParquet(datapath, dataname, 
                                                symbols, savepath)
###############################################################################
# MAIN 
###############################################################################
if __name__ == '__main__':
    # Excel session for database
    #wb = xw.Book(str_path+'\\'+filename+'.xlsx')
    #xlsessid = wb.app.pid
    xlsessid, wb = fetch_xlBook(filename+'.xlsx')
    
    # Models
    try:
        models = copstrat.pd.read_pickle(str_modelname)
    except FileNotFoundError:
        print('\nUpdating models...\n')
        init_models()
        models = copstrat.pd.read_pickle(str_modelname)
    
    # Session
    today_hour = copstrat.get_session_hour()
    while today_hour < stop_hour:
        # Update session hour
        today_hour = copstrat.get_session_hour()
        # Run program
        copstrat.main_algotrade(feedpath = str_path, feedname = filename,
                                futures = ['ES1', 'NQ1'],
                                models = models, wb = wb)
        # Waiting till next new candle
        n_secs2wait1 = min(abs(copstrat.time2wait_M5().seconds+6),300)
        slpuntl = copstrat.datetime.datetime.today() +\
                      copstrat.datetime.timedelta(seconds=n_secs2wait1)
        print(f"\nSleeping {int(n_secs2wait1)} secs... "+\
              "until {:02d}:{:02d}:{:02d}".\
              format(slpuntl.hour, slpuntl.minute, slpuntl.second))
        copstrat.time.sleep(int(n_secs2wait1))
    # Trading hours over
    if input('Trading is over.\n Close Excel Session? ') == 'y':
        # Close XL session    
        wb.save()   
        xw.apps[xlsessid].api.Quit()

"""
###########################################################################
    # GUI
###########################################################################
import pandastable as pt
models = copstrat.pd.read_pickle(str_modelname)

def gen():
    df = copstrat.get_copmodel_run_fromXL(str_path, filename, 'ES1', models)
    
    dTDa1 = copstrat.tkinter.Toplevel()
    dTDa1.title('TestData')
    dTDaPT = pt.Table(dTDa1, dataframe=df, showtoolbar=True, showstatusbar=True)
    dTDaPT.show()

root=copstrat.tkinter.Tk()
root.title('StatArb Trading: CopModel')  
root.geometry('600x700')  
root.resizable(True, True)
button1=copstrat.tkinter.Button(root, text="Run program", 
                                font = ('arial', 8, 'bold'), 
                                command=gen).place(x=10,y=10)
root.mainloop()
###############################################################################
# DEBUG
###############################################################################
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import tkinter as tk

root = tk.Tk()
root.title('StatArb Trading: CopModel')  
root.geometry('900x800')  
root.resizable(True, True)
button1=tk.Button(root, text="Run program", 
                                font = ('arial', 40, 'bold'), 
                                command=gen).place(x=90,y=50)

lf = tk.LabelFrame(root, text='Plot Area')
lf.grid(row=0, column=0, sticky='nwes', padx=3, pady=3)

t = np.arange(0.0,3.0,0.01)
df = pd.DataFrame({'t':t, 's':np.sin(2*np.pi*t)})

fig = Figure(figsize=(5,4), dpi=100)
ax = fig.add_subplot(111)

df.plot(x='t', y='s', ax=ax)

canvas = FigureCanvasTkAgg(fig, master=lf)
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=0)

root.mainloop()
"""