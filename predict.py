import json
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, log_loss
from sklearn.metrics import accuracy_score
import logging



def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data


def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [USDT]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)


def normalise_zero_base(df):
    return df / df.iloc[0] - 1


def normalise_min_max(df):
    return (df - df.min()) / (data.max() - df.min())


def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)


def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size)
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    # y_train = (train_data['close']>train_data['open'])[window_len:].values * 1
    # y_test = (test_data['close']>test_data['open'])[window_len:].values * 1
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][window_len-1:-1].values - 1
        y_test = y_test / test_data[target_col][window_len-1:-1].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test


def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear',
                     dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(
        input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

def cvtToBinaryOption(y_test, preds):
    binary_test = []
    binary_pred = []
    for i in range(1, len(y_test)):
        if y_test[i]> y_test[i-1]:
            binary_test.append(1.0)
        else:
            binary_test.append(0.0)

    for i in range(1, len(preds)):
        if preds[i]> preds[i-1]:
            binary_pred.append(1.0)
        else:
            binary_pred.append(0.0)
    return binary_pred, binary_test


if __name__ == '__main__':
    token = input("Nhap Token: ")
    endpoint = 'https://min-api.cryptocompare.com/data/v2/histohour'
    res = requests.get(endpoint + '?fsym={}&tsym=USD&limit=1000'.format(token))
    hist = pd.DataFrame(json.loads(res.content)['Data']['Data'])
    hist = hist.set_index('time')
    hist.index += 3600*7
    hist.index = pd.to_datetime(hist.index, unit='s')
    target_col = 'close'
    hist.drop(["conversionType", "conversionSymbol"],
              axis='columns', inplace=True)
    
              
    np.random.seed(42)
    window_len = 20
    test_size = 0.1
    zero_base = True
    lstm_neurons = 100
    epochs = 20
    batch_size = 32
    loss = 'mse'
    dropout = 0.2
    optimizer = 'adam'

    train, test, X_train, X_test, y_train, y_test = prepare_data(
        hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

    model = build_lstm_model(
        X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer)
    history = model.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True)

    targets = test[target_col][window_len:]
    preds = model.predict(X_test).squeeze()
    # binary_pred, binary_test = cvtToBinaryOption(y_test, preds)
    binary_test = (y_test > 0) *1
    binary_pred = (preds > 0) *1
    if binary_pred[-1]== 1:
        trend = 'up'
    else:
        trend = 'down'
    print('-------------------------------------')
    hist.drop(["volumeto"],
              axis='columns', inplace=True)
    trend_coin = (hist['close'] > hist['open']) * 1
    hist['Trend'] = trend_coin
    print('Data about {} most recently: \n'.format(token),hist.tail(5))
    print('Du doan trong khoang tu {} den 1 tieng tiep theo {} se {}'.format(hist.index[-1], token, trend))
    acc = accuracy_score(binary_test, binary_pred)
    print('Do chinh xac trong 100 mau gan day cua Token {}: {:.2f}%'.format(token, acc*100))
    print(binary_pred [ -10:])
    print(binary_test[-10:])
