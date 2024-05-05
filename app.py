import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from sqlalchemy import create_engine

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
        df['G_'+col]=df.groupby(['SandiBPR'])[col].pct_change(periods=1)*100
        df['MA_' + col] = df.groupby(['SandiBPR'])[col].transform(lambda x: x.rolling(2, 1).mean())
    df=df.dropna() 
    return df

def read():
    df = pd.read_excel("./Data Train.xlsx")
    df=df.drop(['BDL'], axis=1)
    # remove duplicate values in ProductID column
    df = df.drop_duplicates(subset=['PeriodeData','SandiBPR'], keep='first')

    df['KPMM'] = pd.to_numeric(df['KPMM'], errors='coerce')
    df['ROA'] = pd.to_numeric(df['ROA'], errors='coerce')
    df['BOPO'] = pd.to_numeric(df['BOPO'], errors='coerce')
    df['CR'] = pd.to_numeric(df['CR'], errors='coerce')
    df['NNPL'] = pd.to_numeric(df['NNPL'], errors='coerce')
    df = df.sort_values(['SandiBPR','PeriodeData']) #first sorte the periode and sandiBPR
    df = df.bfill() #if nan fill with the known previous value
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
        ((df_KPMM <=2) & (df_MACR <=1)) | ((df_DKPMM<0) & (df_DCR<0) & (df_KPMM<=12) & (df_MACR<=5))
        ,(df_KPMM <12) & (df_MACR <5)
        

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
    
    df['anomali'] = np.select(conditions,choices,1.0)
    return df

def drop_tosql(df):
    # Create your connection.
    cnx = sqlite3.connect(':memory:')
    disk_engine = create_engine('sqlite:///data_bpr.db')
    df.to_sql('ratio', disk_engine, if_exists='replace')
    
def show_case(df):
    import seaborn as sns
    data = df[df['SandiBPR']==600551].reset_index(drop=True)
    print(data)
    ax = sns.lineplot(x='PeriodeData',y='KPMM', data=data,label = 'KPMM')
    ax.axhline(y = 2,linestyle = "dashed", color = "red",label="KPMM BDR Regulation")
    ax.axhline(y = 12,linestyle = "dashed", color = "orange",label="KPMM BDP Regulation")
     
    ax1= sns.lineplot(x='PeriodeData', y='MA_CR', data=data,label='MA_CR')
    ax1.axhline(y = 1,linestyle = "dashed", color = "blue",label="MA_CR BDR Regulation")
    ax.axhline(y = 5,linestyle = "dashed", color = "yellow",label="MA_CR BDP Regulation")
    ax2= sns.lineplot(x='PeriodeData', y='D_KPMM', data=data,label='D_KPMM')
    ax3= sns.lineplot(x='PeriodeData', y='D_CR', data=data,label='D_CR')
    plt.legend()
    plt.title('{}'.format('600551 - Perumda BPR Bank Purworejo'))
    plt.show()

if __name__=="__main__":
    df=preprocessing()
    show_case(df)    
    drop_tosql(df)
