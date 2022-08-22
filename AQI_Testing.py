# %%
import keras
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

# %%
coords_csv_name = './3kmGrid.csv'
coords_df = pd.read_csv(coords_csv_name)
coords_df = coords_df[['Cell','left','right','bottom','top']]
coords_data = coords_df.to_numpy()

# %%
def createdataset(data,k):
    dataX,dataY=[],[]
    for i in range(data.shape[0]-k):
        x=data[i:i+k]
        y=data[i+k]
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX),np.array(dataY)

# need to update selecting columns for saving row data
def get_cell_list(data, coords_data = coords_df):
    cells={}
    data_len= data.shape[0]
    for row_id in range(data_len):
        row = data.loc[row_id]
        for coords in coords_df.itertuples():
            if coords.bottom<=row.devicelat<=coords.top and coords.left<=row.devicelon<=coords.right:
                if coords.Cell not in cells.keys():
                    cells[coords.Cell] = pd.DataFrame()
                cells[coords.Cell] = cells[coords.Cell].append(row.loc['PM25':'Hum'], ignore_index=True)
                break
    return cells

# %%
def test_cells(data_df, name = '', required_cells = [], look_back = 72, n_feature=4):
    mae_list = []
    rms_list = []
    r2_list = []
    cells_list = []
    error_score = pd.DataFrame()
    cells = get_cell_list(data=data_df)
    for name, items in cells.items():
        if required_cells and name not in required_cells:
            print(f"skipping cell {name}")
            continue
        if items.shape[0]<look_back+1:
            print(f"Dataset for cell '{name}' too small, Size:{items.shape[0]}, Min Required Size:{look_back+1}")
            continue
        predicted_df = pd.DataFrame()
        
        model_name = f"./New_Weights/{name}_CNN_LSTM.h5"
        model = keras.models.load_model(model_name)
        scaler = MinMaxScaler(feature_range =(0,1))
        scaled_items=scaler.fit_transform(items)
        dataset, dataY = createdataset(scaled_items, look_back)
        dataset=np.reshape(dataset,(dataset.shape[0],dataset.shape[1],n_feature,1))
        output=model.predict(dataset)
        y_predicted=scaler.inverse_transform(output)
        y_true = scaler.inverse_transform(dataY)
        y_p_pm=list()
        y_t_pm=list()
        
        for x in range(len(y_predicted)):
            y_p_pm.append(y_predicted[x][2])
            y_t_pm.append(y_true[x][2])
        rms = mean_squared_error(y_t_pm, y_p_pm, squared=False)
        mae = mean_absolute_error(y_t_pm, y_p_pm)
        r2=r2_score(y_t_pm, y_p_pm)
        print(rms, mae, r2)
        cells_list.append(name)
        rms_list.append(rms)
        mae_list.append(mae)
        r2_list.append(r2)
    error_score['cell'] = cells_list
    error_score['rmse'] = rms_list
    error_score['mae'] = mae_list
    error_score['r2'] = r2_list
    error_score.to_csv('./Test_results'+name+'.csv', index=False)


