# -*- coding: utf-8 -*-
"""
@author: arnulf.q@gmail.com
"""
#%%############################################################################
# MODULES
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
# Plotting tools
import matplotlib.pyplot as plt
# Error-warning mgmt
import warnings
warnings.filterwarnings("ignore")
# Display options
pd.set_option('display.float_format', lambda x: '%0.4f' % x)
pd.set_option('display.max_columns', 7)
# Custom modules
import sys
sys.path.append(r'H:\Python\\')
from udf_trading_hyperdrive import csvImport
#%%############################################################################
# DATA IMPORT
# Daily price data from selected fixed income and derivatives securities
data = csvImport(r'H:\db', r'\data_1D.csv')
#%%############################################################################
# UDF
# Function to split dataset into 2 based on time steps taken in the network
def create_dataset(dataset: np.ndarray, time_step: int = 1) -> tuple:
    """
    Parameters
    ----------
    dataset : pd.DataFrame
        Data to split.
    time_step : int, optional
        Number of steps to split data into. The default is 1.

    Returns
    -------
    tuple
        Two arrays of data: the first are the values as input and the second
        the values as output.

    """
    dataX, dataY = [], []
    
    for i in range(len(dataset)-time_step-1):
        A = dataset[i:(i+time_step), 0]
        dataX.append(A)
        dataY.append(dataset[i+time_step, 0])
    
    return np.array(dataX), np.array(dataY)

# Function to visualize CV behavior
def plot_cv_indices(cv: object, n_splits: int, 
                    X: pd.DataFrame, y: pd.Series, 
                    date_col: pd.Series=None) -> None:
    """    
    Args:
        cv: corss-validation object.
        n_splits: number of cv fold splits.
        X, y: data to split in to CV folds.
        date_col: xaxis data

    Returns:
        None.
    """
    from matplotlib.patches import Patch
    cmap_cv = plt.cm.coolwarm
    fig, ax = plt.subplots(1, 1, figsize = (11, 7))
    
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=10, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)


    # Formatting
    yticklabels = list(range(n_splits))
    
    if date_col is not None:
        tick_locations  = ax.get_xticks()
        tick_dates = [" "] +\
            date_col.iloc[list(tick_locations[1:-1])].astype(str).tolist() +\
                [" "]

        tick_locations_str = [str(int(i)) for i in tick_locations]
        new_labels = ['\n\n'.join(x) 
                      for x in zip(list(tick_locations_str), tick_dates) ]
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(new_labels)
    
    ax.set(yticks=np.arange(n_splits+0) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+0.2, -.2])
    ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
              ['Validation set', 'Training set'], loc=(1.02, .8))
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return None
#%%############################################################################
# DATA PREPROCESSING
df_minmax = MinMaxScaler().fit_transform(data)
plt.plot(df_minmax[:,[104,111,112]]);plt.show()
df1 = data['ES1']
sclr_minmax = MinMaxScaler()
df1_minmax = sclr_minmax.fit_transform(df1.values.reshape(-1,1))
#%%############################################################################
# DATA SPLIT
tstep = 100
X, y = create_dataset(df1_minmax, tstep)
# Timeseries CV split
n_splits = 5
my_cv = TimeSeriesSplit(n_splits=n_splits)#max_train_size=int(X.shape[0]*0.65/n_splits), test_size=int(X.shape[0]*0.35/n_splits))
#plot_cv_indices(my_cv, my_cv.n_splits, X, y)
data_cv_split = list(enumerate(my_cv.split(X,y))); data_cv_split.pop(0)
split1_train_idx, split1_test_idx = data_cv_split[0][1]
# Train-Test split
X_train, y_train = X[split1_train_idx,:], y[split1_train_idx]
X_test, y_test = X[split1_test_idx,:], y[split1_test_idx]

#%%############################################################################
# MODEL
# Network params
n_input = X_train.shape[1]
n_epochs = 100
n_batch_size = 128
# Network structure
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(n_input,1)))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
# Network compilation
model.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])
print(model.summary())
# NN train
model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch_size, verbose=1)
# NN test
y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
y_train_pred = sclr_minmax.inverse_transform(y_train_pred)
y_test_pred = sclr_minmax.inverse_transform(y_test_pred)
# NN test metrics
mean_squared_error(df1.iloc[split1_test_idx], y_test_pred)

pd.DataFrame(np.concatenate((df1.iloc[split1_train_idx].values.reshape(-1,1), 
                             y_train_pred),axis=1)).plot()
pd.DataFrame(np.concatenate((df1.iloc[split1_test_idx].values.reshape(-1,1), 
                             y_test_pred),axis=1)).plot()











