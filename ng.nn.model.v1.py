from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings 
import os

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor

def plot_loss(model_history):
    train_loss=[value for key, value in model_history.items() if 'loss' in key.lower()][0]
    valid_loss=[value for key, value in model_history.items() if 'loss' in key.lower()][1]
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    vcolor = 'tab:orange'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(train_loss, '--', color=color, label='Train Loss')
    ax1.plot(valid_loss, color=vcolor, label='Valid Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    plt.legend(loc='upper left')
    plt.title('Model Loss')
    plt.show()

def plot_model_recall_fpr(model_history):
    train_recall=[value for key, value in model_history.items() if 'recall' in key.lower()][0]
    valid_recall=[value for key, value in model_history.items() if 'recall' in key.lower()][1]
    train_fpr=[value for key, value in model_history.items() if 'false_positive_rate' in key.lower()][0]
    valid_fpr=[value for key, value in model_history.items() if 'false_positive_rate' in key.lower()][1]
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Recall', color=color)
    ax1.set_ylim([-0.05,1.05])
    ax1.plot(train_recall, '--', color=color, label='Train Recall')
    ax1.plot(valid_recall, color=color, label='Valid Recall')
    ax1.tick_params(axis='y', labelcolor=color)
    plt.legend(loc='upper left')
    plt.title('Model Recall and FPR')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('False Positive Rate', color=color)  # we already handled the x-label with ax1
    ax2.plot(train_fpr, '--', color=color, label='Train FPR')
    ax2.plot(valid_fpr, color=color, label='Valid FPR')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([-0.05,1.05])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend(loc='upper right')
    plt.show()
    
def get_all_files(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isfile(os.path.join(a_dir, name))]

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

sub = ".\\models\\"
files = get_all_files(sub)
for f in files :
    os.remove(os.path.join(sub, f))

#Load data into pandas DataFrames
# Drop Consumption Summary, 'Delivered to Consumers',  'Delta Delivered to Consumers', 'Consumption', 'DeltaConsumption',
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

# 2. Pre-process Data
#Season as Spot
data["Season"] = ((data["Month"])/3).apply(np.floor)
data["Season"].replace({0:4}, inplace = True)
#Month as solar orbit
data["MoY"] = np.sin(np.pi * (data.Month - 4) / 6)   #*180
data["MTrend"] = np.sin(np.pi * (data.Month - 3) / 6) - data["MoY"]
data["Move"] = np.log(1 + data["Move"] / data["Last"])
#data["Last"] = data["Last"] * 10000

#One Hot Encode Season
data.drop(["Month"],axis = 1 , inplace = True)
data = oneHotEncode(data, ['Season'])

train_data = data[pd.to_datetime(data.index) < datetime(2020, 1, 1)]
train_target = train_data["Move"].to_frame()
train_data.drop(["Move"],axis = 1 , inplace = True)
test_data = data[pd.to_datetime(data.index) >= datetime(2020, 1, 1)]
test_target = test_data["Move"].to_frame()
test_data.drop(["Move"],axis = 1 , inplace = True)

#Scale unbalanced data
scaler = MinMaxScaler().fit(train_data)
# scaler = StandardScaler().fit(x_train)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)



# 3. Make the deep neural network
NN_model = Sequential()
# The input layer
NN_model.add(Input(shape=(train_scaled.shape[1], )))
NN_model.add(Dense(64, kernel_initializer='normal', activation='elu'))
# The hidden layers
NN_model.add(Dense(128, kernel_initializer='normal', activation='elu'))
NN_model.add(Dense(64, kernel_initializer='normal', activation='elu'))
NN_model.add(Dense(16, kernel_initializer='normal', activation='elu'))
# The output layer
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
# Compile the network
NN_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
NN_model.summary()
# Define a checkpoint callback
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint_path = os.path.join('models/', checkpoint_name)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

# 4. Train the model
history = NN_model.fit(train_scaled, train_target, epochs=1000, batch_size=128, validation_split = 0.2, callbacks=callbacks_list).history
plot_loss(history)


# 5. Display model results :
train_output = train_target
test_output = test_target
#test_data.drop(['Move'],axis = 1 , inplace = True)

files = get_all_files(sub)
for f in (f for f in files if f[-4:] == "hdf5"):
    NN_model.load_weights(sub + f) # load it
    NN_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    column = f[8:-6]
    train_output[column] = NN_model.predict(train_scaled)
    test_output[column] = NN_model.predict(test_scaled)

test_monthly = np.exp(test_output)
test_result = test_monthly.cumprod()
test_result = test_result * 2.19
fig, axes = plt.subplots(nrows=2, ncols=1)
test_output.plot(ax=axes[0], use_index=True, legend=False)
test_result.plot(ax=axes[1], use_index=True)

train_monthly = np.exp(train_output)
cols = []
for column in train_monthly.columns:
    if column == "Move":
        cols.append(column)
    else:
        column = column[:-5] + "__" + str(int(train_monthly["Move"].corr(train_monthly[column])*100)) + "%"
        cols.append(column)
train_monthly.columns = cols
train_result = train_monthly.cumprod()
train_result.plot(use_index=True)


print("Completed")

