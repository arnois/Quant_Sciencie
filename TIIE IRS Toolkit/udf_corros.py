# Code Description
"""
This Module supports TIIE_Trading for dealer market making best prices.
"""
#%%############################################################################
# Modules
###############################################################################
import QuantLib as ql
import numpy as np
import pandas as pd
import networkx as nx
import xlwings as xw
# local modules
import sys, os
str_cwd = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(str_cwd)

#%%############################################################################
# Global variables
###############################################################################
tenor2ql = {'B':ql.Days,'D':ql.Days,'M':ql.Months,'W':ql.Weeks,'Y':ql.Years}
clr_orange_output = (253, 233, 217)

#%%############################################################################
# Spreads Paths for Corros Functions
###############################################################################
def clean_spreads(spreads_corros: pd.DataFrame) -> pd.DataFrame:
    """Cleans spreads DataFrame gotten from Corros file.
    
    DataFrame gotten from Corros file has missing column names and does not
    have the spread tenors separated. Columns Tenor 1 L and Tenor 2 L will be
    created for easier data handling.
    

    Parameters
    ----------
    spreads_corros : pd.DataFrame
        Spreads DataFrame gotten from Corros file.

    Returns
    -------
    spreads_df : pd.DataFrame
        Spreads DataFrame ready to handle.

    """

    i=1
    cols = []
    for c in spreads_corros.columns:
        if c==None:
            c='Tenor'+str(i)
            i=i+1
        cols.append(c)
            
    spreads_corros.columns = cols   
    
    # Tenor column (3-6, 6-9, etc.)
    tenor_column = spreads_corros.columns[1]
    
    # Separate first tenor and second tenor
    spreads_corros['Tenor 1 L'] = spreads_corros[tenor_column].apply(
        lambda x: x.split('-')[0])
    spreads_corros['Tenor 2 L'] = spreads_corros[tenor_column].apply(
        lambda x: x.split('-')[1])

    # Get rid of bad data and convert tenors to int
    spreads_corros = spreads_corros[(spreads_corros['Tenor 1 L'] != '') & 
                                    (spreads_corros['Tenor 2 L'] != '')]
    spreads_corros['Tenor 1 L'] = spreads_corros['Tenor 1 L'].astype(int)
    spreads_corros['Tenor 2 L'] = spreads_corros['Tenor 2 L'].astype(int)

    # String tenors (3m6m, 6m9m, etc.) rename column
    tenor_str_column = spreads_corros.columns[2]  
    spreads_corros = spreads_corros.rename(
        columns = {tenor_str_column: 'Tenor'})

    # Columns we need  
    spreads_df = spreads_corros[['Tenor', 'Tenor 1 L', 'Tenor 2 L', 'Bid', 
                                 'Offer']]

    # Remove rows that don't have neither bid nor offer values
    spreads_df = spreads_df.set_index('Tenor')
    spreads_df.dropna(axis = 0, subset = ['Bid', 'Offer'], how = 'all', 
                      inplace = True)
    
    return spreads_df
#%%
def clean_rates(rates_corros: pd.DataFrame) -> pd.DataFrame:
    """Cleans rates DataFrame gotten from Corros file.
    

    Parameters
    ----------
    rates_corros : pd.DataFrame
        Rates DataFrame gotten from Corros file.

    Returns
    -------
    rates_df : pd.DataFrame
        Clean DataFrame.

    """
    
    # Fill columns with no name
    i=1
    cols = []
    for c in rates_corros.columns:
        if c==None:
            c='Tenor'+str(i)
            i=i+1
        cols.append(c)
    rates_corros.columns = cols
    
    # Tenor (3, 6, 9) and string tenors columns (3m, 6m, 9m)
    tenor_column, tenor_str_column = rates_corros.columns[1:3]
    
    # Drop rows with no tenors
    rates_df = rates_corros.dropna(subset=[tenor_column, tenor_str_column])
    
    # Rename columns
    rates_df = rates_df.rename(columns={tenor_column: 'Tenor_L', 
                                        tenor_str_column: 'Tenor'})
    
    return rates_df
#%%
def create_graphs(spreads_df: pd.DataFrame) -> list:
    """
    

    Parameters
    ----------
    spreads_df : pd.DataFrame
        DataFrame with spreads info.

    Returns
    -------
    list
        List with two graphs:
            bG: bid graph
            oG: offer graph.

    """
    
    bid_spreads = spreads_df['Bid'].tolist()
    offer_spreads = spreads_df['Offer'].tolist()
    
    start_tenors = spreads_df['Tenor 1 L'].tolist()
    end_tenors = spreads_df['Tenor 2 L'].tolist()
    
    bG = nx.DiGraph()
    oG = nx.DiGraph()

    for t in range(0, len(start_tenors)):
        
        # Start and end nodes
        v = start_tenors[t]
        s = end_tenors[t]
        
        # Case short to long tenor
        if v < s:
            if not pd.isnull(bid_spreads[t]) and bid_spreads[t] != '':
                bG.add_edge(v, s, weight = bid_spreads[t])
                oG.add_edge(s, v, weight = -bid_spreads[t])
            if not pd.isnull(offer_spreads[t]) and offer_spreads[t] != '':
                bG.add_edge(s, v, weight = -offer_spreads[t])
                oG.add_edge(v, s, weight = offer_spreads[t])
            
        # Case long to short tenor    
        elif s < v:
            if not pd.isnull(offer_spreads[t]) and offer_spreads[t] != '':
                bG.add_edge(v, s, weight = -offer_spreads[t])
                oG.add_edge(s, v, weight = offer_spreads[t])
            if not pd.isnull(bid_spreads[t]) and bid_spreads[t] != '':
                bG.add_edge(s, v, weight = bid_spreads[t])
                oG.add_edge(v, s, weight = -bid_spreads[t])
    
    return bG, oG
#%%
def get_all_paths(graph: nx.DiGraph, side: str) -> dict:
    """Get all paths from one tenor to another given a graph with spreads info.
    
    Uses a directed graph with spreads as edge's weight and tenors as nodes to
    get all possible paths between all nodes.
    

    Parameters
    ----------
    graph : nx.DiGraph
        Directed Graph with spreads information.
    side : str
        Indicates if we are on the Bid side or on the Offer side.

    Returns
    -------
    dict
        Dictionary with all possible paths. The keys are the tenors, and the
        values are DataFrames with all possible paths starting from the key
        tenor to the rest.

    """
    
    # Dictionaries to save paths for each starting node
    paths_dict = {} 
    
    # For each node we get all the paths starting in that node to all others
    for n in graph.nodes():
        
        # Lists for all paths starting at n and all the weights
        all_paths = []
        all_weights = []
        
        for v in graph.nodes():
            
            # Paths and weights going from n to v
            paths = [path for path in nx.all_simple_paths(graph, n, v)]
            weights = [nx.path_weight(graph, path, 'weight') for path in paths]
            
            # Save paths and weights to list
            all_paths.extend(paths)
            all_weights.extend(weights)
            
        # Save all paths starting at n in dictionary
        paths_dict[n] = pd.DataFrame({side + ' Path': all_paths, 
                                      side + ' Spread': all_weights})
    
    return paths_dict
#%%
def best_paths_fn(rates_df: pd.DataFrame, graph: nx.DiGraph, side: str, 
                  paths_dict: dict) -> dict:
    """
    

    Parameters
    ----------
    rates_df : pd.DataFrame
        DataFrame with rates.
    graph : nx.DiGraph
        Directed Graph with spreads information.
    side : str
        Indicates if we are on the Bid side or on the Offer side.
    paths_dict : dict
        Dictionary with all possible paths. The keys are the tenors, and the
        values are DataFrames with all possible paths starting from the key
        tenor to the rest.

    Returns
    -------
    dict
        Dictionary with best paths. The keys are the tenors, and the
        values are DataFrames with the best paths starting from the key
        tenor to the rest.

    """
    
    best_paths_dic = {}
    
    # Get best bid/offer rates starting from known node (tenor, bid)
    for i in range(0, rates_df.shape[0]):
        
        row = rates_df.iloc[i]
        tenor = row.Tenor_L
        
        if side == 'Bid':
            rate = row.Bid
            
        elif side == 'Offer':
            rate = row.Offer
        
        
        # If there were no spreads from known node the graph will not have it 
        # as a node
        try:
            paths_start = paths_dict[tenor]
        except:
            rate_start = rates_df[rates_df['Tenor_L']==tenor][side].values[0]
            best_paths_dic[tenor] = pd.DataFrame(
                {'Start': tenor, 'End': tenor, side + ' Path': [[int(tenor)]], 
                 side: [rate_start], side + ' Spread': [0], 'Length': [0]})
            continue
        
        # If bid rate is nan we can't use it as a known node
        if not pd.isnull(rate) and rate != '':
            
            # Dictionary to save bid paths starting in known node
            paths = {}
            
            # Iterate over all nodes
            for k in paths_dict.keys():
                
                # Paths starting in known node and ending in specific node k
                paths_k = [(path, nx.path_weight(graph, path, 'weight')) for 
                           path in paths_start[side + ' Path'] if path[-1]==k]
                
                # DataFrame with all paths starting in known node and ending in 
                # specific node with sum of spreads
                k_df = pd.DataFrame({side + ' Path': [p for (p, w) in paths_k], 
                                      side + ' Spread': [w for (p, w) 
                                                         in paths_k]})
                
                # Bid rate of end node will be start bid plus sum of spreads
                k_df[side]  = [np.round(rate + w/100, 7) for w in 
                               k_df[side + ' Spread']]
                
                # Save DataFrame in dictionary. DataFrame with bid paths 
                # starting in known node and ending in k.
                paths[k] = k_df
            
            # DataFrame to save best paths for all nodes starting from known 
            # node
            best_paths = pd.DataFrame(columns = ['Start','End', side + ' Path', 
                                                 side])
            
            # Bid path keys are the nodes gotten from the spreads
            for k in paths.keys():
                
                # Bid paths that end in node k.
                options = paths[k]
                # Best path is the one with maximum bid.
                if side == 'Bid':
                    best_path = options[options[side]==np.round(
                        options[side].max(), 7)]
                elif side == 'Offer':
                    best_path = options[options[side]==np.round(
                        options[side].min(), 7)]
                # Get length of best paths to get the shortest one
                best_path['Length'] = best_path[side + ' Path'].apply(
                    lambda x: len(x)-1)
                best_path = best_path[best_path['Length'] == \
                                      best_path['Length'].min()]
                
                # Column with end node
                best_path.insert(0, 'End', k)
                # Column with start node
                best_path.insert(0, 'Start', tenor)
                
                # Concat DataFrame to save all best paths starting in known 
                # node
                best_paths = pd.concat([best_paths, best_path])
            
            # Add trivial path in case it is the best
            best_paths = pd.concat([best_paths, 
                                    pd.DataFrame(
                                        {'Start': [tenor], 'End': [tenor], 
                                         side + ' Path': [[int(tenor)]], 
                                         side: [rate], side + ' Spread': [0], 
                                         'Length': [0]})])
            
            # Drop duplicates to get only one path
            best_paths = best_paths.drop_duplicates(subset = 'End')
            
            # Save to dictionary
            best_paths_dic[tenor] = best_paths.sort_values(by = 'End')
    
    return best_paths_dic
#%%
def format_paths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gives acceptable format to paths DataFrame to write it in Excel file.
    
    Since xlwings can't write a list as values in Excel, you have to convert
    lists with paths into strings.'
    

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with bid and offer paths.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with only string values to write it in Corros Excel file.

    """
    df = df.fillna('')
    df['Bid Path'] = [', '.join(map(str, df['Bid Path'].tolist()[i])) 
                                   for i in range(0, len(df['Bid Path']))]

    df['Offer Path'] = [', '.join(map(str, df['Offer Path'].tolist()[i])) 
                                     for i in range(0, len(df['Offer Path']))]

    df['Bid Path'] = ['['+s+']' for s in df['Bid Path']]
    df['Offer Path'] = ['['+s+']' for s in df['Offer Path']]
    df['Bid Path'] = np.select([df['Bid']==''], [''], df['Bid Path'])
    df['Offer Path'] = np.select([df['Offer']==''], [''], df['Offer Path'])
    df['Bid Spread'] = np.select([df['Bid Path']==''], [''], df['Bid Spread'])
    df['Offer Spread'] = np.select([df['Offer Path']==''], [''], 
                                   df['Offer Spread'])
    
    return df
#%%
def corros_fn(corros_book: xw.Book) -> tuple:
    """Fills Corros Excel file with best bid/offer spreads and paths.
    

    Parameters
    ----------
    corros_book : xw.Book
        Corros Excel file.

    Returns
    -------
    tuple
        Tuple of three elements that include the following:
            best_spreads_copy: pd.DataFrame with best bid/offer spreads and 
            paths.
            paths_data_copy: pd.DataFrame with all the paths starting from each
            tenor to the rest.
            closes_df: pd.DataFrame with close rates.

    """
    
    best_sheet = corros_book.sheets('BEST')
    #best_sheet.activate()
    best_sheet.api.Calculate()

    row_spreads = best_sheet.range('V52').end('down').row
    
    # Check if there is already any data
    if row_spreads < 75:
        spreads_corros = best_sheet.range('U51:Y'+str(row_spreads)).options(
            pd.DataFrame, header=1, index=False).value
        spreads_df = clean_spreads(spreads_corros)
    
    # Case no data available
    else:
        spreads_df = pd.DataFrame(
            columns = ['Tenor 1 L', 'Tenor 2 L', 'Bid', 'Offer'])

    row_rates = best_sheet.range('V28').end('down').row
    rates_corros = best_sheet.range('U27:Y'+str(row_rates)).options(
        pd.DataFrame, header=1, index=False).value

    # DataFrame of bid and offer starting rates
    rates_df = clean_rates(rates_corros)

    bG, oG = create_graphs(spreads_df)
                
    # Dictionaries to save paths for each starting node            
    bid_paths_dict = get_all_paths(bG, 'Bid')
    offer_paths_dict = get_all_paths(oG, 'Offer')

        
    # Get best bid/offer paths and merge them in same dictionary
    best_paths_bid_dic = best_paths_fn(rates_df, bG, 'Bid', bid_paths_dict)
    best_paths_offer_dic = best_paths_fn(rates_df, oG, 'Offer', 
                                         offer_paths_dict)

    all_keys = set().union(best_paths_bid_dic.keys(), 
                           best_paths_offer_dic.keys())
    best_paths_dic = {}

    for k in all_keys:
        try:
            b_df = best_paths_bid_dic[k]
        except:
            o_df = best_paths_offer_dic[k]
            b_df = pd.DataFrame({'Start': [k]*o_df.shape[0], 
                                 'End': o_df['End'].tolist(), 
                                 'Bid Path': ['']*o_df.shape[0], 
                                 'Bid': ['-']*o_df.shape[0], 
                                 'Bid Spread': ['-']*o_df.shape[0], 
                                 'Length': ['']*o_df.shape[0]})
        try:    
            o_df = best_paths_offer_dic[k]
        except:
            b_df = best_paths_bid_dic[k]
            o_df = pd.DataFrame({'Start': [k]*b_df.shape[0], 
                                 'End': b_df['End'].tolist(), 
                                 'Offer Path': ['']*b_df.shape[0], 
                                 'Offer': ['-']*b_df.shape[0], 
                                 'Offer Spread': ['-']*b_df.shape[0], 
                                 'Length': ['']*b_df.shape[0]})
            
        best_df = b_df.merge(o_df, how = 'outer', left_on = 'End', 
                             right_on = 'End')
        best_df['Start_x'] = np.select([best_df['Start_x'].isna()], 
                                       [best_df['Start_y']], 
                                       best_df['Start_x'])
        best_df['Start_y'] = np.select([best_df['Start_y'].isna()], 
                                       [best_df['Start_x']], 
                                       best_df['Start_y'])
        best_paths_dic[k] = best_df
    
    # Create DataFrame with all paths
    paths_df = pd.DataFrame()

    for k in best_paths_dic.keys():
        paths_df = pd.concat([paths_df, best_paths_dic[k]])
    
    
    paths_data = paths_df[['Start_x', 'End', 'Bid Path', 'Bid', 
                           'Bid Spread', 'Offer Path', 'Offer', 
                           'Offer Spread']]

    paths_data = paths_data.rename(columns = {'Start_x': 'Start'})
    
    tenors = [str(int(paths_data['Start'].tolist()[i])) + 'm' + \
              str(int(paths_data['End'].tolist()[i]))  + 'm' for i in 
              range(0, paths_data.shape[0])]
        
    paths_data.insert(2, 'Spread Tenor', tenors)
    
    # Create DataFrames with best bid and offer rates
    best_spreads_bid = pd.DataFrame()
    best_spreads_offer = pd.DataFrame()  
      
    for v in paths_df['End'].unique():
        
        df_a = paths_df[paths_df['End']==v]
        
        # Get maximum bid (ignoring blank rates)
        df_b = df_a[df_a['Bid'] != '-']
        best_b = df_b[df_b['Bid']==df_b['Bid'].max()]
        
        # If there are two paths with same rate, get the shortest one
        best_b['Length'] = best_b['Bid Path'].apply(lambda x: len(x))
        best_b = best_b[best_b['Length'] == best_b['Length'].min()]
        best_b = best_b.drop_duplicates(subset = 'End')
        
        # Get minimum offer (ignoring blank rates)
        df_o = df_a[df_a['Offer'] != '-']
        best_o = df_o[df_o['Offer']==df_o['Offer'].min()]
        
        # If there are two paths with same rate, get the shortest one
        best_o['Length'] = best_o['Offer Path'].apply(lambda x: len(x))
        best_o = best_o[best_o['Length'] == best_o['Length'].min()]
        best_o = best_o.drop_duplicates(subset = 'End')
        
        best_spreads_bid = pd.concat([best_spreads_bid, best_b])
        best_spreads_offer = pd.concat([best_spreads_offer, best_o])
    
    # Create one unified DataFrame
    best_spreads_df = best_spreads_bid[['End', 'Bid Path', 'Bid', 
                                        'Bid Spread']]\
        .merge(best_spreads_offer[['End', 'Offer Path', 'Offer', 
                                   'Offer Spread']],  how='outer', 
               left_on='End', right_on='End')
        
    best_spreads_df = best_spreads_df.rename(columns={'End': 'Tenor'})
    best_spreads_df.sort_values(by = 'Tenor', inplace = True)
    rates = rates_df[['Tenor', 'Bid', 'Offer']]


    best_spreads_copy = best_spreads_df.copy()
    best_spreads_copy = format_paths(best_spreads_copy)

    complete_tenors_range = best_sheet.range('AF28').end('down').row
    complete_tenors = best_sheet.range('AF28:AF'+
                                       str(complete_tenors_range)).value

    missing_tenors = [t for t in complete_tenors if t not in 
                      best_spreads_copy['Tenor'].tolist()]
    missing_df = pd.DataFrame({'Tenor': missing_tenors, 'Bid Path': '', 
                               'Bid': '', 'Bid Spread': '', 'Offer Path': '', 
                               'Offer': '', 'Offer Spread': ''})

    best_spreads_copy = pd.concat([best_spreads_copy, missing_df], 
                                  ignore_index=True)
    best_spreads_copy = best_spreads_copy.sort_values(by='Tenor')

    best_sheet.range('AG52:AH74').clear_contents()
    best_sheet.range('AE52:AE74').clear_contents()
    best_sheet.range('AL52:AM74').clear_contents()
    
    # Replace blanks with "-" so Excel doesn't count them as zeroes
    best_sheet.range('AG52').value = best_spreads_copy[['Bid', 'Offer']]\
        .replace('', '-').values
        
    # Write results in Corros Excel file
    best_sheet.range('AE52').value = best_spreads_copy[['Tenor']].values
    best_sheet.range('AL52').value = best_spreads_copy[['Bid Path']]\
        .replace('[]', '').values
    best_sheet.range('AM52').value = best_spreads_copy[['Offer Path']].\
        replace('[]', '').values

    # Write best paths in Corros Excel file
    paths_data.index = range(0, paths_data.shape[0])
    paths_data_copy = paths_data.copy()
    paths_data_copy = format_paths(paths_data_copy)

    paths_sheet = corros_book.sheets('Paths')
    paths_sheet.clear_contents()
    paths_sheet['A1'].options(pd.DataFrame, header=1, index=False, 
                              expand='table').value = \
        paths_data_copy.replace('[]', '').sort_values(by='Start')
    tenors_start=paths_data_copy['Start'].unique()
    tenors_start.sort()
    paths_sheet.range('K2').value = tenors_start.reshape(-1, 1)
    best_sheet.api.Calculate()
    
    # Get close rates
    close_row = best_sheet.range('AE52').end('down').row
    tenors_close = best_sheet.range('AE52:AE'+str(close_row)).value   
    closes = best_sheet.range('AJ52:AJ'+str(close_row)).value
    closes_df = pd.DataFrame({'Tenor': tenors_close, 'Close': closes})
    
    return best_spreads_copy, paths_data_copy, closes_df
#%%
def fill_rates(wb: xw.Book, best_spreads: pd.DataFrame, 
               closes_df: pd.DataFrame) -> None:
    """Fills rates in TIIE_IRS_Data with best rates gotten from spreads.
    

    Parameters
    ----------
    wb : xw.Book
        TIIE_IRS_Data Excel file.
    best_spreads : pd.DataFrame
        DataFrame with best bid/offer rates gotten from spreads.
    closes_df : pd.DataFrame
        DataFrame with close rates in case some tenors are empty.

    Returns
    -------
    None

    """
    
    rates_sheet = wb.sheets('MXN_TIIE')
    rates_sheet.activate()
    prev_rates = rates_sheet.range('G1:I15').options(
        pd.DataFrame, header=1, index=False).value
    prev_rates.set_index('Tenor', inplace=True)
    best_spreads_copy = best_spreads.copy()
    best_spreads_copy.set_index('Tenor', inplace=True)
    
    new_rates = prev_rates.merge(best_spreads_copy[['Bid', 'Offer']], 
                                 how='left', left_index=True, right_index=True)
    
    new_rates = new_rates.reset_index()
    
    # Fill tenors greater than 10Y that didn't have a rate via spreads using 
    # close rates
    close_rates = new_rates.merge(closes_df, how='left', left_on='Tenor', 
                                  right_on='Tenor')
    rate130_close = close_rates[close_rates['Tenor']==130]['Close'].values[0]
    spread130 = []
    
    # Non essential tenors don't have to be filled
    for i in range(0, close_rates.shape[0]):
        if close_rates['Tenor'][i] in prev_rates.index:
            try:
                spread130.append(close_rates['Close'][i] - rate130_close)
            except:
                spread130.append('')
        else:
            pass
    # Spreads between 10Y and the rest of the tenors
    new_rates['Spread130'] = spread130
    
    # Bid rate for 10Y
    rate130_b = new_rates[new_rates['Tenor']==130]['Bid_y'].values[0]
    if rate130_b != '':
    
        conditions_b = [new_rates['Tenor']==1, 
                      (new_rates['Tenor']<=130) & (new_rates['Bid_y']==''), 
                      (new_rates['Tenor']<=130) & (new_rates['Bid_y'].isna()),
                      (new_rates['Tenor']>130) & (new_rates['Bid_y']==''),
                      (new_rates['Tenor']>130) & (new_rates['Bid_y'].isna())]
    
        options_b = [new_rates['Bid_x'], np.nan, np.nan, 
                     rate130_b+new_rates['Spread130'], 
                     rate130_b+new_rates['Spread130']]
    
    else:
        conditions_b = [new_rates['Tenor']==1, 
                      (new_rates['Tenor']<=130) & (new_rates['Bid_y']==''), 
                      (new_rates['Tenor']<=130) & (new_rates['Bid_y'].isna())]
        
        options_b = [new_rates['Bid_x'], np.nan, np.nan]
    
    # Offer rate for 10Y
    rate130_o = new_rates[new_rates['Tenor']==130]['Offer_y'].values[0]
    if rate130_o != '':
    
        conditions_o = [
            new_rates['Tenor']==1, 
            (new_rates['Tenor']<=130) & (new_rates['Offer_y']==''), 
            (new_rates['Tenor']<=130) & (new_rates['Offer_y'].isna()),
            (new_rates['Tenor']>130) & (new_rates['Offer_y']==''),
            (new_rates['Tenor']>130) & (new_rates['Offer_y'].isna())]
    
        options_o = [new_rates['Offer_x'], np.nan, np.nan, 
                     rate130_o+new_rates['Spread130'], 
                     rate130_o+new_rates['Spread130']]
    
    else:
        conditions_o = [
            new_rates['Tenor']==1, 
            (new_rates['Tenor']<=130) & (new_rates['Offer_y']==''), 
            (new_rates['Tenor']<=130) & (new_rates['Offer_y'].isna())]
        
        options_o = [new_rates['Offer_x'], np.nan, np.nan]
    
    new_rates['Bid'] = np.select(conditions_b, options_b, new_rates['Bid_y'])
    new_rates['Offer'] = np.select(conditions_o, options_o, 
                                   new_rates['Offer_y'])
    # Handling NaN values
    idx_NaN = new_rates[['Bid','Offer']].isna().any(axis=1)
    new_rates.loc[idx_NaN,['Bid','Offer']] = new_rates.loc[idx_NaN,['Bid_x','Offer_x']]
    
    # print out new marks
    rates_sheet.range('H2').value = new_rates[['Bid', 'Offer']].values
    # print out paths
    rates_sheet.range('S3').value =  best_spreads[['Bid Path']].values
    rates_sheet.range('T3').value =  best_spreads[['Offer Path']].values
