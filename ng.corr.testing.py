from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings 

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor

def get_data():
    #get train data
    data_path ='data.csv'
    df = pd.read_csv(data_path, index_col=0)
    
    return df

def oneHotEncode(df,colNames):
    for col in colNames:
        if( df[col].dtype != np.dtype('object')):
            dummies = pd.get_dummies(df[col],prefix=col)
            df = pd.concat([df,dummies],axis=1)

            #drop the encoded column
            df.drop([col],axis = 1 , inplace=True)
    return df

def curve_shift(v, by):
    tmp = v.shift(by)
    tmp = tmp.fillna(0)
    return tmp

#Load data into pandas DataFrames
df = get_data()
data = df[['Month', 'Last Future', 'Price Move',
       'Dry Production', 'Residential Consumption',
       'Deliveries to Commercial Consumers', 'Industrial Consumption', 'Deliveries to Electric Power Consumers',
       'Delta Dry Production',
       'Delta Residential Consumption', 'Delta Deliveries to Commercial', 'Delta Industrial Consumption', 
       'Delta Deliveries to Electric Power', 'Pipeline Imports', 'Pipeline Exports', 'Liquefied Exports',
       'Working Storage', 'Net Withdraws', 'Delta Pipeline Imports', 'Delta Pipeline Exports', 
       'Delta Liquefied Exports', 'Delta Working Storage', 'Delta Net Withdraws', 'Non-Comm Net Stded',
       'Delta Non-Comm Net', 'HDD CONUS', 'HDD Delta Avg CONUS']]
data.columns = ['Month', 'Last', 'Move',
       'DryProduction', 'Residential',
       'Commercial', 'Industrial', 'ElectricPower',
       'DeltaDryProduction',
       'DeltaResidential', 'DeltaCommercial', 'DeltaIndustrial', 
       'DeltaElectricPower', 'PipelineImports', 'PipelineExports', 'LNGExports',
       'Storage', 'NetWithdraws', 'DeltaPipelineImports', 'DeltaPipelineExports', 
       'DeltaLNGExports', 'DeltaStorage', 'DeltaNetWithdraws', 'NonCommNetStded',
       'DeltaNonCommNet', 'HDDCONUS', 'HDDCONUSvs5YAvg']

#Season as Spot
data["Season"] = ((data["Month"])/3).apply(np.floor)
data["Season"].replace({0:4}, inplace = True)
data["MoY"] = np.sin(np.pi * (data.Month - 4) / 6)   #*180
data["MTrend"] = np.sin(np.pi * (data.Month - 3) / 6) - data["MoY"]

data.drop(["Month"],axis = 1 , inplace = True)

#data = oneHotEncode(data, ['Season', 'MoSeason'])


train_data = data[pd.to_datetime(data.index) < datetime(2020, 1, 1)]
test_data = data[pd.to_datetime(data.index) >= datetime(2020, 1, 1)]

C_mat = train_data.corr()
fig = plt.figure(figsize = (15,15))

sb.heatmap(C_mat, cmap="PiYG_r", vmax = .8, square = True)
plt.draw()

""" color codes
'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 
'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 
'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 
'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 
'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 
'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 
'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 
'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 
'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 
'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 
'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 
'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 
'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 
'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 
'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 
'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 
'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 
'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
"""




print("Completed")

