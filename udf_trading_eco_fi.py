# -*- coding: utf-8 -*-
"""
UDF for trading and ETF Portfolio construction

@author: jquintero
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

###############################################################################
# SUPPORT
###############################################################################
# function to manage data and column ranges
def sliceDataFrame(df, dt_start = None, dt_end = None, lst_securities = None):
    # columns verification
    lstIsEmpty = (lst_securities is None) or (lst_securities == [])
    if lstIsEmpty:
        tmpsecs = df.columns.tolist()
    else:
        tmpsecs = lst_securities
    
    # column-filtered df
    tmpdfret = df[tmpsecs]

    # date range verification
    if (dt_start is None) or not (np.any(tmpdfret.index >= pd.to_datetime(dt_start))):
        tmpstr1 = df.index.min()
    else:
        tmpstr1 = dt_start
    if (dt_end is None) or not (np.any(tmpdfret.index >= pd.to_datetime(dt_end))):
        tmpstr2 = df.index.max()
    else:
        tmpstr2 = dt_end
        
    return tmpdfret.loc[tmpstr1:tmpstr2,tmpsecs].dropna()

# function to add text labels in scatter plot
def scttrplt_addtext(axis, xs, ys, labels):
    k=0
    for x,y in zip(xs,ys):
        label = labels[k]
        axis.annotate(label, # this is the text
                     (x,y), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     alpha = 0.75,
                     ha='center')
        k+=1
        
###############################################################################
# SERIES PLOTTING
###############################################################################
def normLinplots(df, dt_start = None, dt_end = None, lst_securities = None, 
                 plt_size = (9,6), plt_cmap = 'Accent'):
    """
    Returns: 
        Linear plot of selected securities between the selected periods 
        in the input DataFrame.
    """
    # securities and date ranges verification
    df_ = sliceDataFrame(df=df, dt_start=dt_start, 
                         dt_end=dt_end, lst_securities=lst_securities)
    df_ret = df_
    
    # return-normalized data
    normdata = df_ret
    
    # plotting
    ax = normdata.plot(figsize=plt_size, colormap=plt_cmap)
    plt.title('', 
              size = 20)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('')
    plt.tight_layout()
    plt.show()
    return None

###############################################################################
# SERIES ANALYSIS
###############################################################################
# funciton to plot correlation matrix among securities
def plot_corrmatrix(df, dt_start = None, dt_end = None, 
                    lst_securities = None, plt_size = (10,8)):
    """
    Returns: Correlation matrix among securities changes.
    """
    from seaborn import heatmap
    # securities and date ranges verification
    df_ = sliceDataFrame(df=df, dt_start=dt_start, 
                         dt_end=dt_end, lst_securities=lst_securities)
    tmpdfret = df_
    
    # corrmatrix
    plt.figure(figsize=plt_size)
    heatmap(
        tmpdfret.corr(method='spearman'),        
        cmap='RdBu', 
        annot=True, 
        vmin=-1, vmax=1,
        fmt='.2f')
    plt.title('Spearman', size=20)
    plt.tight_layout()
    plt.show()
    return None

# function to visualize return series returns boxplots
def boxplot_rets(df, dt_start = None, dt_end = None, 
                 lst_securities = None, plt_size = (10,8), str_ttl=''):
    """
    Returns: 
        Plots each series changes' boxplots.
    """
    # securities and date ranges verification
    df_ = sliceDataFrame(df=df, dt_start=dt_start, 
                         dt_end=dt_end, lst_securities=lst_securities)
    tmp_df = df_
    
    # boxplot
    bp = tmp_df.plot.box(figsize=plt_size)
    bp.set_title(str_ttl, size = 20)
    bp.set_ylabel('Value')
    plt.xticks(rotation=90)
    bp.title.set_size(20)
    plt.tight_layout()
    plt.show()
    return None

# function to get pairwise scatterplots matrix
def scatterplot(df, dt_start = None, dt_end = None, 
                lst_securities = None, plt_size = (10,8)):
    """
    Returns: 
        Pairwise rate changes scatterplot matrix
    """
    from seaborn import pairplot
    # securities and date ranges verification
    df_ = sliceDataFrame(df=df, dt_start=dt_start, 
                         dt_end=dt_end, lst_securities=lst_securities)
    tmp_df = df_
    
    # scatterplot matrix
    pairplot(tmp_df)
    plt.tight_layout()
    plt.show()
    return None

# function to get statistics for different variables in a dataframe
def statistics(data):
    """
    Returns: Summary stats for each serie in the input DataFrame.
    """
    from scipy import stats
    tmpmu = np.round(data.mean().values,4)
    tmptmu_5pct = np.round(stats.trim_mean(data, 0.05),4)
    tmpiqr = np.round(data.apply(stats.iqr).values,4)
    tmpstd = np.round(data.std().values,4)
    tmpskew = np.round(data.apply(stats.skew).values,4)
    tmpme = np.round(data.median().values,4)
    tmpkurt = np.round(data.apply(stats.kurtosis).values,4)
    tmpcol = ['Mean', '5% Trimmed Mean', 'Interquartile Range', 
              'Standard Deviation', 'Skewness', 'Median', 'Kurtosis']
    tmpdf = pd.DataFrame([tmpmu, tmptmu_5pct, tmpiqr,
                        tmpstd, tmpskew, tmpme, tmpkurt]).T
    tmpdf.columns = tmpcol
    tmpdf.index = data.columns
    return tmpdf

# function for the shapiro-wilk test for normality
def test_shapiro(data):
    """
    Returns: Most recognized normality test results for the input Serie.
    """
    from scipy.stats import shapiro
    pvalue_limit = 0.05
    stat, p = shapiro(data)
    dec = 'Do not reject H0'
    name = data.name
    if p<= pvalue_limit:
        dec = 'Reject H0'
    print('Shapiro-Wilk (Normality) Test.'+\
          '\n\tH0: Data comes from Normal distribution.')
    print(f'\t\t{name}: {dec}')
    return None

###############################################################################
# DATA PROCESSING
###############################################################################
# function to import csv data
def csvImport(str_path, str_filename):
    import datetime as dt
    dateparse = lambda x: dt.datetime.strptime(x, '%d/%m/%Y')
    tmpdf = pd.read_csv(str_path+str_filename, parse_dates=['date'], 
                        date_parser=dateparse)
    tmpdf = tmpdf.set_index(['date'])
    return tmpdf

###############################################################################
# ML METHODS
###############################################################################
# function to plot biplot
def biplot(axis,score,coeff,pcax,pcay,labels=None):
    pca1=pcax-1
    pca2=pcay-1
    xs = score[:,pca1]
    ys = score[:,pca2]
    #n=score.shape[1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    axis.scatter(xs*scalex,ys*scaley,c='darkcyan',edgecolors='gray')
    for i in range(n):
        axis.arrow(0, 0, coeff[i,pca1], coeff[i,pca2], color='r', alpha=0.35) 
        if labels is None:
            axis.text(coeff[i,pca1]* 1.15, 
                      coeff[i,pca2] * 1.15, "Var"+str(i+1), 
                      color='g', ha='center', va='center')
        else:
            axis.text(coeff[i,pca1]* 1.15, 
                      coeff[i,pca2] * 1.15, labels[i], 
                      color='black', ha='center', va='center')
    axis.set_xlim(-1,1)
    axis.set_ylim(-1,1)
    axis.set_xlabel("PC{}".format(pcax), fontsize=15)
    axis.set_ylabel("PC{}".format(pcay), fontsize=15)
    
    return None

# function to assess best number of clusters in data returns
def preclustering(df, dt_start = None, dt_end = None, 
                  lst_securities = [], plt_size = (15,7), str_mode = 'sclr'):
    """
    Returns: 
        Distorsion, Silhouette and Calinski Harabasz elbow methods 
        for scoring optimal number of clusters.
    """
    from sklearn.cluster import KMeans
    from yellowbrick.cluster import KElbowVisualizer
    #from scipy.cluster import hierarchy as shc
    
    # securities and date ranges verification
    df_ = sliceDataFrame(df=df, dt_start=dt_start, 
                         dt_end=dt_end, lst_securities=lst_securities)
    if str_mode == 'chg':
        tmpdfret = df_.diff().dropna()*100
    elif str_mode == 'sclr':
        from sklearn.preprocessing import StandardScaler
        sclr = StandardScaler()
        tmpdfret = sclr.fit_transform(df_)
    elif str_mode == 'norm':
        from sklearn.preprocessing import Normalizer
        nrm = Normalizer()
        tmpdfret = nrm.fit_transform(df_)
    
    # data proc
    X = tmpdfret.T
    ub = tmpdfret.shape[1]
    
    # kmeans clustering
    model = KMeans()
    
    # clustering plot
    fig, ax = plt.subplots(nrows=1,ncols=3,figsize=plt_size)
    plt.subplots_adjust(hspace=0.4,wspace=0.2)
    ax[0].title.set_text('Distorsion')
    ax[0].set_xlabel('N Clusters')
    ax[0].set_ylabel('Score')
    ax[1].title.set_text('Silhouette')
    ax[2].title.set_text('Calinski Harabasz')
    v1=KElbowVisualizer(model, k=(2,ub), timings= False, ax=ax[0]).fit(X)
    v2=KElbowVisualizer(model, k=(2,ub), timings= False, 
                        metric='silhouette', ax=ax[1]).fit(X)
    v3=KElbowVisualizer(model, k=(2,ub), timings= False, 
                        metric='calinski_harabasz', ax=ax[2]).fit(X)
    fig.suptitle('Score Elbow for KMeans Clustering', size=20)
    return None

# function to get clusters in data 
def cluster_kmeans(df, n_clusters = 2, dt_start = None, dt_end = None, 
                   lst_securities = [], iclrmap = False, str_mode = 'chg'):
    """
    Returns: 
        Clusters, via KMeans, in input DataFrame
        along with the fitted model.
    """
    from sklearn.cluster import KMeans
    # securities and date ranges verification
    df_ = sliceDataFrame(df=df, dt_start=dt_start, 
                         dt_end=dt_end, lst_securities=lst_securities)
    if str_mode == 'chg':
        tmpdfret = df_.diff().dropna()*100
    elif str_mode == 'sclr':
        from sklearn.preprocessing import StandardScaler
        sclr = StandardScaler()
        tmpdfret = pd.DataFrame(sclr.fit_transform(df_), index = df_.index,
                                columns = df_.columns)
    elif str_mode == 'norm':
        from sklearn.preprocessing import Normalizer
        nrm = Normalizer()
        tmpdfret = pd.DataFrame(nrm.fit_transform(df_), index = df_.index,
                                columns = df_.columns)
    
    # clustering by kmeans
    kmeans = KMeans(n_clusters = n_clusters,  
                    max_iter=800, tol=1e-12, random_state=0)
    kmeansfit = kmeans.fit(tmpdfret.T.to_numpy())
    
    # cluster segmentation
    l = []
    clusters = []
    for i in range(n_clusters):
        l = np.concatenate((l,np.where(kmeansfit.labels_ == i)[0]))
        clusters.append(np.where(kmeansfit.labels_ == i)[0].tolist())
    
    #  cluster labels array
    clusters = np.array([np.array(x) for x in clusters], dtype=object)
    
    # securities in each cluster
    k=0
    cluster_sec = {}
    for cluster in clusters:
        cluster_sec[f'Cluster{k}'] = df_.columns[cluster].tolist()
        k+=1
        
    # plot intercluster distance map
    if iclrmap:
        from yellowbrick.cluster import InterclusterDistance
        # clustering model and visualizer
        visualizer = InterclusterDistance(kmeans)
        visualizer.fit(tmpdfret.T)
        visualizer.show() 
        plt.show()
        
    return {'model': kmeansfit, 'cluster_set': cluster_sec}

# function to plot dendrograms for hierarchical clustering
def hierClustDendro(df, dt_start = None, dt_end = None, lst_securities = []):
    from scipy.cluster import hierarchy as shc
    # securities and date ranges verification
    df_ = sliceDataFrame(df=df, dt_start=dt_start, 
                         dt_end=dt_end, lst_securities=lst_securities)
    tmpdfret = df_.apply(np.log).diff().dropna()
    
    # dendrogram plot
    plt.figure(figsize=(10, 7))  
    plt.title("Dendrograms", size=20)  
    dend = shc.dendrogram(shc.linkage(tmpdfret.T, method='ward'))
    return None

# function to assess explained variance across PCs
def pca_assess(df, dt_start = None, dt_end = None, 
               lst_securities = [], plt_size = (15,7), str_mode = 'chg'):
    """
    Returns: 
        Explained variance across PCs and number of
        components needed to explain at least 80%.
    """
    from sklearn.decomposition import PCA
    sclr = StandardScaler()
    
    # securities and date ranges verification
    df_ = sliceDataFrame(df=df, dt_start=dt_start, 
                         dt_end=dt_end, lst_securities=lst_securities)
    
    # levels2changes
    if str_mode == 'chg':
        tmpdf = df_.diff().dropna()*100
    elif str_mode == 'ret':
        tmpdf = df_.apply(np.log).diff().dropna()
    else:
        tmpdf = df_.copy()
    
    # data scaling transform
    tmpdf_scld = pd.DataFrame(sclr.fit_transform(tmpdf), 
                              columns=tmpdf.columns,
                              index=tmpdf.index)
    
    # PCA fit
    pca_out = PCA().fit(tmpdf_scld)
    
    # proportion of variance explained
    variance_limit = 0.80
    comps_needed = np.min(
        np.where(
            pca_out.explained_variance_ratio_.cumsum() >= variance_limit)
        )+1

    # cum var expl plot
    plt.axes(facecolor='#EBEBEB')
    plt.grid(color='white', linestyle='-', linewidth=1.5)
    
    plt.plot(list(range(1,pca_out.explained_variance_ratio_.shape[0]+1,1)),
             pca_out.explained_variance_ratio_.cumsum(), 'o-', c = 'darkcyan')
    
    plt.axhline(y = variance_limit, color='r', linestyle='--')
    plt.ylabel('Explained Variance')
    plt.xlabel('Principal Components')
    plt.xticks(list(range(1,pca_out.explained_variance_ratio_.shape[0]+1,1)), 
               size = 7)
    plt.title('Cumulative Explained Variance', size = 15)
    plt.show()
    print(f'Components needed to express {variance_limit:.0%} '+\
          f'(at least) of the variation in the data: {comps_needed: .0f}')
    return None

# function to show PCA loadings amongst clusters
def plt_pcal_cltr(pca_out, loadings_df, loadings, dic_cltr, 
                  d_cltr_cmap, plt_size = (13,5)):
    """
    Returns: First 3 PC loadings amongst clusters.
    """
    #cluster_model = dic_cltr['model']
    cluster_set = dic_cltr['cluster_set']
    # loadings & clusters
    tmpdf = loadings_df.iloc[:,:3]
    ## variables with cluster label
    df_cltset = pd.DataFrame()
    for k,v in cluster_set.items():
        df_cltset = df_cltset.append(pd.DataFrame(
            zip(v, len(v)*[int(k.replace('Cluster',''))]), 
            columns=['variable','cluster']))
    df_cltset = df_cltset.reset_index(drop=True)
    df_cltset['cluster']+=1
    df_cltset = df_cltset.set_index('variable')
    tmpdf = tmpdf.merge(df_cltset, how='left', left_index=True, right_index=True)
    tmpdf = tmpdf.reset_index()
    
    # PCs Variance
    pc1l = round(pca_out.explained_variance_ratio_[0]*100, 2)
    pc2l = round(pca_out.explained_variance_ratio_[1]*100, 2)
    pc3l = round(pca_out.explained_variance_ratio_[2]*100, 2)
    # {1:'red',2:'blue',3:'cyan', 4:'orange',5:'purple',6:'green'}
    cmap = d_cltr_cmap
    
    # xy-lims
    pc1_max = max(abs(np.array([min(loadings[0]), max(loadings[0])])))*1.25
    pc2_max = max(abs(np.array([min(loadings[1]), max(loadings[1])])))*1.25
    pc3_max = max(abs(np.array([min(loadings[2]), max(loadings[2])])))*1.25
    pc1_lims = (-pc1_max,pc1_max)
    pc2_lims = (-pc2_max,pc2_max)
    pc3_lims = (-pc3_max,pc3_max)
    
    # plot figure config
    plt.style.use('seaborn')
    #axlims = (-1, 1)
    mrks = 25
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', 
                          linestyle='') for color in cmap.values()]
    ## PC1 vs PC2
    ax = plt.scatter(x=loadings[0], y=loadings[1], 
                  c=tmpdf['cluster'].map(cmap), s=mrks)
    ax.axes.set_xlabel(f'PC1 ({pc1l}%)')
    ax.axes.set_ylabel(f'PC2 ({pc2l}%)')
    ax.axes.set_xlim(pc1_lims)
    ax.axes.set_ylim(pc2_lims)
    scttrplt_addtext(ax.axes, loadings[0], loadings[1], tmpdf['variable'])
    ax.axes.set_title('PCA Loadings', size=20)
    plt.legend(markers, cmap.keys(), numpoints=1, title='Cluster', 
               bbox_to_anchor=(1.11,0.95))
    plt.tight_layout()
    plt.show()
    
    # PC1 vs PC3
    ax = plt.scatter(x=loadings[0], y=loadings[2], 
                  c=tmpdf['cluster'].map(cmap), s=mrks)
    ax.axes.set_xlabel(f'PC1 ({pc1l}%)')
    ax.axes.set_ylabel(f'PC3 ({pc3l}%)')
    ax.axes.set_xlim(pc1_lims)
    ax.axes.set_ylim(pc3_lims)
    scttrplt_addtext(ax.axes, loadings[0], loadings[2], tmpdf['variable'])
    ax.axes.set_title('PCA Loadings', size=20)
    plt.legend(markers, cmap.keys(), numpoints=1, title='Cluster', 
               bbox_to_anchor=(1.11,0.95))
    plt.tight_layout()
    plt.show()
    
    # PC3 vs PC2
    ax = plt.scatter(x=loadings[1], y=loadings[2], 
                  c=tmpdf['cluster'].map(cmap), s=mrks)
    ax.axes.set_xlabel(f'PC2 ({pc2l}%)')
    ax.axes.set_ylabel(f'PC3 ({pc3l}%)')
    ax.axes.set_xlim(pc2_lims)
    ax.axes.set_ylim(pc3_lims)
    scttrplt_addtext(ax.axes, loadings[1], loadings[2], tmpdf['variable'])
    ax.axes.set_title('PCA Loadings', size=20)
    plt.legend(markers, cmap.keys(), numpoints=1, title='Cluster', 
               bbox_to_anchor=(1.11,0.95))
    plt.tight_layout()
    plt.show()

    return None

# function to run PCA
def pca_(df, n_comps = 3, dt_start = None, dt_end = None, 
         lst_securities = [], plt_size = (15,7), str_mode = 'chg', 
         plt_pcmap = False):
    """
    Returns: 
        PCA.
    """
    from sklearn.decomposition import PCA
    from seaborn import heatmap
    from sklearn.preprocessing import StandardScaler
    sclr = StandardScaler()
    
    # securities and date ranges verification
    df_ = sliceDataFrame(df=df, dt_start=dt_start, 
                         dt_end=dt_end, lst_securities=lst_securities)
    # level2changes
    if str_mode=='chg':
        tmpdf = df_.diff().dropna()*100
    elif str_mode=='ret':
        tmpdf = df_.apply(np.log).diff().dropna()
    elif str_mode=='':
        tmpdf = df_.copy()
    # scaled data
    sclr.fit(tmpdf)
    tmpdf_scld = pd.DataFrame(sclr.transform(tmpdf), columns=tmpdf.columns,
                              index=tmpdf.index)
    # PCA fit
    pca_out = PCA(n_components=n_comps).fit(tmpdf_scld)
    # PCA scores
    pca_scores = pca_out.transform(tmpdf_scld)
    
    # component loadings or weights 
    #(correlation coefficient between original variables and the component) 
    loadings = pca_out.components_
    num_pc = pca_out.n_features_
    pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
    loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
    loadings_df['variable'] = tmpdf_scld.columns.values
    loadings_df = loadings_df.set_index('variable')

    # components loadings plot
    if plt_pcmap:
        heatmap(loadings_df[[f'PC{i}' for i in range(1,n_comps+1)]], 
                     annot=True, cmap='RdBu', vmin=-1, vmax=1)
        plt.title(f'First {n_comps} Principal Components Loadings\n', size=20)
        plt.tight_layout()
        plt.show()
    return pca_out, loadings_df, loadings, pca_scores

# fucntion that runs PCA score plot + loading plot
def plt_pca_biplots(pca_scores, loadings_df, loadings, dic_cltr):
    """
    Returns: PCA biplots for the first 3 components.
    """
    # loadings & clusters
    cluster_set = dic_cltr['cluster_set']
    tmpdf = loadings_df.iloc[:,:3]
    ## variables with cluster label
    df_cltset = pd.DataFrame()
    for k,v in cluster_set.items():
        df_cltset = df_cltset.append(pd.DataFrame(
            zip(v, len(v)*[int(k.replace('Cluster',''))]), 
            columns=['variable','cluster']))
    df_cltset = df_cltset.reset_index(drop=True)
    df_cltset['cluster']+=1
    df_cltset = df_cltset.set_index('variable')
    tmpdf = tmpdf.merge(df_cltset, how='left', left_index=True, right_index=True)
    tmpdf = tmpdf.reset_index()
    
    # biplots
    ## PC1 vs PC2
    plt.style.use('seaborn')
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10))
    biplot(ax,pca_scores,loadings.T,1,2,labels=tmpdf['variable'])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    fig.suptitle('PCA Scores', size=24)
    fig.tight_layout()
    plt.show()
    ## PC1 vs PC3
    plt.style.use('seaborn')
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10))
    biplot(ax,pca_scores,loadings.T,1,3,labels=tmpdf['variable'])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    fig.suptitle('PCA Scores', size=24)
    fig.tight_layout()
    plt.show()
    ## PC2 vs PC3
    plt.style.use('seaborn')
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10))
    biplot(ax,pca_scores,loadings.T,2,3,labels=tmpdf['variable'])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    fig.suptitle('PCA Scores', size=24)
    fig.tight_layout()
    plt.show()
    
    return None

# function to get first elements that account for at least 80% var explained
def pca_top_comps(pca, feat_names):
    n_comps = pca.n_components_
    tmp_pca_comp = pd.DataFrame(pca.components_.T, index=feat_names, 
                                columns=range(1,n_comps+1))
    dic_topComps = {}
    for n in range(n_comps):
        dic_topComps[n+1] = \
        (tmp_pca_comp**2)[n+1].\
            sort_values(ascending=False)[
                :(tmp_pca_comp**2)[n+1].\
                    sort_values(ascending=False).cumsum().gt(.80).idxmax()
                ].index
    return dic_topComps

##############################################################################
# STUDY PERIODS MGMT
##############################################################################
# Function to get end of month day of given date
def last_day_of_month(str_date):
    # String2Datetime64
    tmpdt = np.datetime64(str_date)
    # 28D + 4D it's always next month
    next_month = np.datetime64(np.datetime_as_string(tmpdt)[:8]+'28') +\
        np.timedelta64(4,'D')
    next_month_day = next_month.astype(object).day
    # Bring back one month
    return next_month - np.timedelta64(next_month_day, 'D') #datetime.timedelta(days=next_month.day)

# Function to get datetime delimiters for train-test split
def dt_delims_train_test(str_end_train, n_yr_train, n_month_test):
    # Train start date
    str_start_train = pd.to_datetime(
        str(str_end_train.year - n_yr_train)+'-'+\
        str(str_end_train.month)+'-'+\
        str(str_end_train.day))

    # Test start date
    test_next_month = str_end_train.month%12+1
    str_start_test = pd.to_datetime(
            str(str_end_train.year + (test_next_month == 1)*1)+'-'+\
            str(test_next_month)+'-1')
    
    # Test end date
    test_Tm2add = str_start_test.month+n_month_test-1
    test_y2add = (test_Tm2add>12)*test_Tm2add//12
    str_end_test = pd.to_datetime(
        f'{str_start_test.year + test_y2add}-{str_start_test.month}-1')
    str_end_test = pd.to_datetime(
        f'{str_end_test.year}-{test_Tm2add}-1')

    # String fmt
    str_start_train = str_start_train.strftime('%Y-%m-%d')
    str_start_test = str_start_test.strftime('%Y-%m-%d')
    str_end_test = str_end_test.strftime('%Y-%m-%d')
    str_end_train = str_end_train.strftime('%Y-%m-%d')
    
    return str_start_train, str_end_train, str_start_test, str_end_test

# Function to set data split dates
def get_train_test_split_dates(df, n_yr_train, n_month_test, n_month_test_roll):
    # Study periods first trainning end date
    dt_first = pd.to_datetime(
        str(df.index[0].year + n_yr_train)+'-'+str(df.index[0].month)+'-1')
    # Total study periods
    n_total_months = int(
        (df.index[-1]-dt_first)/np.timedelta64(1, 'M')/n_month_test_roll
    )
    # Dates splits specs
    colnames = ['train_st','train_ed','test_st','test_e']
    df_dtsplit = pd.DataFrame(columns=colnames)
    dt_train_end = dt_first
    for i in range(n_total_months):
        # Total move fwd
        n_totm2add = 1*i*n_month_test_roll
        # Years to move fwd
        n_y2add = n_totm2add//12
        dt_train_end = pd.to_datetime(
            f'{dt_first.year + n_y2add}-{dt_first.month}-1')
        # Months to move fwd
        n_m2add = n_totm2add%12
        monthnum = dt_train_end.month + n_m2add
        dt_train_end = pd.to_datetime(
            f'{dt_train_end.year + (monthnum>12)*monthnum//12}-'+\
                f'{(monthnum-1)%12+1}-1')
        # Date flags
        df_dtsplit.loc[i+1,:] = dt_delims_train_test(dt_train_end,
                                                     n_yr_train, 
                                                     n_month_test)
    return df_dtsplit

###############################################################################
# MODELING METHODS
###############################################################################
# function to plot model results performance
def plot_model_perf(data_perf):
    import matplotlib.dates as mdates
    from matplotlib.dates import DateFormatter

    str_y = data_perf.columns[0]
    str_yhat = data_perf.columns[1]
    x = data_perf.index.values
    y = data_perf[str_y].values
    yhat = data_perf[str_yhat]
    y1 = data_perf.LB
    y2 = data_perf.UB
    str_ttl = f'{str_y} Prices\nModel Performance'

    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(x, y, color='black')
    ax.fill_between(x, y1, y2, alpha=0.1, 
                   edgecolor='darkcyan', facecolor='gray',
          linewidth=1, linestyle='-', antialiased=True)
    ax.plot(x,yhat, linestyle='--', color='darkcyan')
    ax.set(title=f'{str_y} Prices\nModel Performance')
    ax.legend([str_y, str_yhat], bbox_to_anchor=(1,0.5))
    ax.xaxis.set_major_formatter(DateFormatter("%d/%b/%y"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))

# function to get performance metrics from a regression model
def rgr_perf_metrics(y_true, y_pred, model_name = 'Model'):
    from sklearn import metrics
    """
    Returns: Performance metrics between the real (observed) values and 
    predicted ones from a regression model.
        y_true: array-like holding the real values from the response variable
        y_pred: array-like holding the fitted values from the regression model
    """
    exp_var = metrics.explained_variance_score(y_true, y_pred)
    max_err = metrics.max_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
    r2 = metrics.r2_score(y_true, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
    metric_name = ['Exp.Var.','Max Error','MAE','MSE','RMSE','R2 Score','MAPE']
    perf_metrics = pd.DataFrame([exp_var, max_err, mae, mse, rmse, r2, mape], 
                                index = metric_name, columns = [model_name])
    return perf_metrics

# function to perform np.linspace function applicable to datetime64 datatypes
def linspace_datetime64(start_date, end_date, n):
    """
    Returns: Array of separated dates.
        start_date: numpy.datetime64 starting date.
        end_date: numpy.datetime64 ending date.
        n = integer to determing number of samples to generate.
    """
    return np.linspace(0, 1, n) * (end_date - start_date) + start_date

# function to produce backtesting indexes
def getBacktest_idx(idx_last, n_train, n_test, backwards=False):
    """
    Returns: Indices of each train/test dataset backtest run.
        idx_last: integer with the value of last index in dataframe
    """
    if(backwards):
        idx_df = pd.DataFrame(list(range(idx_last-1,0,-1*n_test)), columns=['test_f'])
        idx_df['test_i'] = idx_df['test_f']-n_test+1
        idx_df['train_f'] = idx_df['test_i']
        idx_df['train_i'] = idx_df['train_f']-n_train+1
        idx_df = idx_df.loc[idx_df['train_i']>=0,]
    else:
        idx_df = pd.DataFrame(list(range(0,idx_last-1,n_test)), columns=['train_i'])
        idx_df['train_f'] = idx_df['train_i']+n_train-1
        idx_df['test_i'] = idx_df['train_f']
        idx_df['test_f'] = idx_df['test_i']+n_test-1
        idx_df = idx_df.loc[idx_df['train_f']<=(idx_last-1),]
    return idx_df

# function to mask spread according to its value relative to mean, ub and lb
def maskSpread(dfz):
    """
    Returns: provided dataframe with additional column masking spread 
    value relative to its limits:
            {buy:-2,xdown:-1,xup:1,sell;2}
    """
    tmpdf = dfz.copy()
    tmpdf['zone'] = -2
    # zone conditions
    condition1 = (tmpdf['z'] > tmpdf['lb']) & (tmpdf['z'] <= tmpdf['z_mu'])
    condition2 = (tmpdf['z'] > tmpdf['z_mu']) & (tmpdf['z'] < tmpdf['ub'])
    condition3 = tmpdf['z'] >= tmpdf['ub']
    # masking 
    tmpdf['zone'].mask(condition1, -1, inplace=True)
    tmpdf['zone'].mask(condition2, 1, inplace=True)
    tmpdf['zone'].mask(condition3, 2, inplace=True)
    return tmpdf

# Function to gen random samples from empirical distribution
def empirical_sample(ecdf, size):
    u = np.random.uniform(0,1,size)
    ecdf_x = np.delete(ecdf.x,0)
    sample = []
    for u_y in u:
        idx = np.argmax(ecdf.y >= u_y)-1
        e_x = ecdf_x[idx]
        sample.append(e_x)
    return pd.Series(sample,name='emp_sample')

# Function for R paths from empirical distribution
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
