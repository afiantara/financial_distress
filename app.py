import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from sqlalchemy import create_engine
from modelling import *
from sklearn.metrics import accuracy_score
from ydata_profiling import ProfileReport



def treatoutliers(df, columns=None, factor=1.5, method='IQR', treament='cap'):
    """
    Removes the rows from self.df whose value does not lies in the specified standard deviation
    :param columns:
    :param in_stddev:
    :return:
    """
#     if not columns:
#         columns = self.mandatory_cols_ + self.optional_cols_ + [self.target_col]
    if not columns:
        columns = df.columns
    
    for column in columns:
        if method == 'STD':
            permissable_std = factor * df[column].std()
            col_mean = df[column].mean()
            floor, ceil = col_mean - permissable_std, col_mean + permissable_std
        elif method == 'IQR':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            floor, ceil = Q1 - factor * IQR, Q3 + factor * IQR
        
        if treament == 'remove':
            df = df[(df[column] >= floor) & (df[column] <= ceil)]
        elif treament == 'cap':
            df[column] = df[column].clip(floor, ceil)
            
    return df

def show_outliers(df,columns):
    booxplot = df.boxplot(column=columns)  
    plt.show()

def add_data(df,cols):
    #df['year'] = df['PeriodeData'].dt.year
    for col in cols:
        df['D_' + col] = df.groupby(['SandiBPR'])[col].diff()
        #df['G_'+col]=df.groupby(['SandiBPR'])[col].pct_change(periods=1)*100
        df['MA_' + col] = df.groupby(['SandiBPR'])[col].transform(lambda x: x.rolling(2, 1).mean())
    df=df.dropna() 
    return df

def read():
    df1 = pd.read_excel("./data uji.xlsx")
    df1 = df1.drop_duplicates(subset=['PeriodeData','SandiBPR'], keep='first')
    df1['KPMM'] = pd.to_numeric(df1['KPMM'], errors='coerce')
    df1['ROA'] = pd.to_numeric(df1['ROA'], errors='coerce')
    df1['BOPO'] = pd.to_numeric(df1['BOPO'], errors='coerce')
    df1['CR'] = pd.to_numeric(df1['CR'], errors='coerce')
    df1['NNPL'] = pd.to_numeric(df1['NNPL'], errors='coerce')
    
    df = pd.read_excel("./Data Train.xlsx")
    df=df.drop(['BDL'], axis=1)
    
    # remove duplicate values in ProductID column
    df = df.drop_duplicates(subset=['PeriodeData','SandiBPR'], keep='first')

    df['KPMM'] = pd.to_numeric(df['KPMM'], errors='coerce')
    df['ROA'] = pd.to_numeric(df['ROA'], errors='coerce')
    df['BOPO'] = pd.to_numeric(df['BOPO'], errors='coerce')
    df['CR'] = pd.to_numeric(df['CR'], errors='coerce')
    df['NNPL'] = pd.to_numeric(df['NNPL'], errors='coerce')
    
    #gabung data uji dan data latih
    frames = [df1, df]
    
    df = pd.concat(frames)
    
    df = df.sort_values(['SandiBPR','PeriodeData']) #first sorte the periode and sandiBPR
    df = df.bfill() #if nan fill with the known previous value
    #df['year'] = df['PeriodeData'].dt.year
    return df

def readCIU():
    df = pd.read_excel("./ciu.xlsx")
    df['CIU']=1.0
    df['SandiBPR']=df['SandiBank']
    return df

def preprocessing():
    df=read()
    df = add_data(df,['KPMM','ROA','BOPO','CR','NNPL'])
    dfCIU=readCIU()
    df['CIU'] = df['SandiBPR'].map(dfCIU.set_index('SandiBPR')['CIU']).fillna(0)
    
    #add flag OJK BDP
    # flag =0 is normal
    # flag =1 is BDP
    # flag =2 is BDR
    
    #Flag== 1
    #a. TKS dengan PK 5 (lima) selama 2 (dua) periode berturut-turut;
    #b. CR rata-rata 3 (tiga) bulan terakhir kurang dari 5%(lima persen); dan/atau
    #c. rasio KPMM kurang dari 12% (dua belas persen).
    
    #flag == 0
    #a. TKS minimal PK 4 (empat);
    #b. CR rata-rata 3 (tiga) bulan terakhir paling sedikit 5% (lima persen); dan
    #c. rasio KPMM paling sedikit 12% (dua belas persen).
    
    #Flag==2
    #a. rasio KPMM menjadi kurang dari sama dengan 2% (dua persen) dan/atau 
    #b. CR rata-rata selama 3 (tiga) bulan terakhir menjadi kurang dari sama dengan 1% (satu persen); atau
    #atau
    #a. mengalami penurunan rasio KPMM dan/atau CR; dan
    #b. tidak mampu meningkatkan rasio KPMM menjadi paling sedikit 12% (dua belas persen) dan/atau CR rata-rata 3 (tiga) bulan terakhir paling sedikit 5% (lima persen).
    df_KPMM = df['KPMM']
    df_MACR = df['MA_CR']
    df_DKPMM = df['D_KPMM']
    df_DCR = df['D_CR']
    
    # List of conditions
    conditions = [
        (((df_KPMM <=2) ^ (df_MACR <=1)) | (((df_DKPMM<0) ^ (df_DCR<0)) & ((df_KPMM<=12) ^ (df_MACR<=5))))
        ,((df_KPMM <12) ^ (df_MACR <5))
    ]

    # List of values to return
    choices  = [2.0, 1.0]
    
    # create a new column in the DF based on the conditions
    df['flag'] = np.select(conditions, choices, 0.0)
    
    df_flag=df['flag']
    df_CIU = df['CIU']
    
    conditions=[
        ((df_flag>0) & (df_CIU>0))
        ,((df_flag==0) & (df_CIU==0))
    ]
    choices=[0.0,0.0]
    
    #df['anomali'] = np.select(conditions,choices,1.0)
    #output class

    conditions=[
        ((df_flag>0) & (df_CIU==0)),
        ((df_flag==0) & (df_CIU==0)),
        ((df_flag>0) & (df_CIU==1)),
        ((df_flag==0) & (df_CIU==1))
    ]
    choices=['Waspada','Normal','Awas','Siaga']
    df['class'] = np.select(conditions,choices)
    return df

def doImputation(df):
    #filter untuk kebutuhan data yang ingin di analisis oleh DNN
    df=df.drop(['PeriodeData','SandiBPR','CIU','flag'], axis=1)
    #transform skewness
    from skew_autotransform import skew_autotransform
    transformedDF = skew_autotransform(df.copy(deep=True), plot = True, exp = False, threshold = 0.5)
    return transformedDF

def drop_tosql(df):
    # Create your connection.
    cnx = sqlite3.connect(':memory:')
    disk_engine = create_engine('sqlite:///data_bpr.db')
    df.to_sql('ratio', disk_engine, if_exists='replace')

def cross_correlation(df):
    import seaborn as sns; sns.set(style="ticks", color_codes=True)
    cols = ['KPMM', 'ROA','BOPO', 'CR', 'NNPL','D_KPMM', 'D_ROA', 'D_BOPO',
    'D_CR', 'D_NNPL','MA_KPMM','MA_ROA','MA_BOPO','MA_CR','MA_NNPL']
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.0)
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    hm = sns.heatmap(cm,
    cbar=True,
    annot=True,
    square=True,
    fmt='.2f',
    annot_kws={'size': 15},
    yticklabels=cols,
    xticklabels=cols)
    plt.tight_layout()
    #plt.savefig('images/10_04.png', dpi=300)
    plt.show()

def analize(df):
    import seaborn as sns; sns.set(style="ticks", color_codes=True)
    g = sns.pairplot(df[['KPMM','ROA','BOPO','CR','NNPL']], diag_kind="kde", height=2)
    plt.show()
    g = sns.pairplot(df[['D_KPMM','D_ROA','D_BOPO','D_CR','D_NNPL']], diag_kind="kde", height=2)
    plt.show()
    g = sns.pairplot(df[['MA_KPMM','MA_ROA','MA_BOPO','MA_CR','MA_NNPL']], diag_kind="kde", height=2)
    plt.show()
    
def create_EDA(df):
    profile = ProfileReport(df, title="Profiling Data Report",explorative=True)
    profile.to_file("REPORT.html")
    
def show_case(df):
    import seaborn as sns
    data = df[df['SandiBPR']==600137].reset_index(drop=True)
    ax = sns.lineplot(x='PeriodeData',y='KPMM', data=data,label = 'KPMM')
    ax.axhline(y = 2,linestyle = "dashed", color = "red",label="KPMM BDR Regulation")
    ax.axhline(y = 12,linestyle = "dashed", color = "orange",label="KPMM BDP Regulation")
     
    ax1= sns.lineplot(x='PeriodeData', y='MA_CR', data=data,label='MA_CR')
    ax1.axhline(y = 1,linestyle = "dashed", color = "blue",label="MA_CR BDR Regulation")
    ax.axhline(y = 5,linestyle = "dashed", color = "yellow",label="MA_CR BDP Regulation")
    ax2= sns.lineplot(x='PeriodeData', y='D_KPMM', data=data,label='D_KPMM')
    ax3= sns.lineplot(x='PeriodeData', y='D_CR', data=data,label='D_CR')
    plt.legend()
    plt.title('{}'.format('600137 - PT BPR Artaprima Danajasa'))
    plt.show()

if __name__=="__main__":
    df=preprocessing()
    #create_EDA(df)
    #analize(df)
    #cross_correlation(df)
    #show_case(df)    
    #drop_tosql(df)
    
    #training(df)
    #training_kfold(df)
    training_automatic_verification(df)
    
    #init()
    #X,Y=load_data(df)
    #load_compile_evaluate_model(X,Y)
    
 