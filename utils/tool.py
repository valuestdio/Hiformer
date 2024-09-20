import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import h5py
import os

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def metric(pred, label):
    """
    计算MAE, RMSE和MAPE。

    参数:
    pred (np.array): 预测值数组。
    label (np.array): 实际值数组。

    返回:
    tuple: 包含MAE, RMSE和MAPE的元组。
    """
    if len(pred) > 0 and len(label) > 0:
        mae = np.mean(np.abs(pred - label))
        mse = np.mean((pred - label) ** 2)
        rmse = np.sqrt(np.mean((pred - label) ** 2))
        mape = np.mean(np.abs((pred - label) / label)) 

    return mae, rmse, mse

def normalize(x, use_norm=True):
    """
    对二维数组进行均值和标准差归一化。

    参数:
    x (np.array): 形状为 (n_samples, n_features) 的二维数组。

    返回:
    np.array: 归一化后的数组。
    """
    if use_norm:
        # 计算均值和标准差
        means = np.mean(x, axis=0, keepdims=True)
        x_centered = x - means
        stdev = np.sqrt(np.var(x_centered, axis=0, keepdims=True, ddof=0) + 1e-5)
        
        # 归一化
        x_normalized = x_centered / stdev
        return x_normalized
    else:
        return x
    
def h5load(adress):
    data = pd.read_hdf(adress)
    data = data.iloc[:len(data)].sort_index()
    x = data.to_numpy()
    return x

def seq2instance(data, P, Q, slide=2):
    #sliding window
    num_step, dims = data.shape
    num_sample = (num_step - P - Q) // slide + 1
    x = np.zeros(shape=(num_sample, P, dims))
    y = np.zeros(shape=(num_sample, Q, dims))
    for i in range(num_sample):
        start_idx = i * slide
        x[i] = data[start_idx: start_idx + P]
        y[i] = data[start_idx + P: start_idx + P + Q]
    return x, y

def weatherProcess(args, Weather, wtrain_steps, wval_steps, wtest_steps):
    wtrain = Weather[: wtrain_steps]
    wval = Weather[wtrain_steps: wtrain_steps + wval_steps]
    wtest = Weather[-wtest_steps:]

    trainX, _ = seq2instance(wtrain, args.P, args.Q)
    valX, _ = seq2instance(wval, args.P, args.Q)
    testX, _ = seq2instance(wtest, args.P, args.Q)

    wtrain = np.expand_dims(trainX, axis=3)
    wval = np.expand_dims(valX, axis=3)
    wtest = np.expand_dims(testX, axis=3)

    return wtrain, wval, wtest

def windpowerProcess(args, windpower, train_steps, val_steps, test_steps):
    train = windpower[: train_steps]
    val = windpower[train_steps: train_steps + val_steps]
    test = windpower[-test_steps:]

    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)

    trainX = np.expand_dims(trainX, axis=3)
    valX = np.expand_dims(valX, axis=3)
    testX = np.expand_dims(testX, axis=3)
    trainY = np.expand_dims(trainY, axis=3)
    valY = np.expand_dims(valY, axis=3)
    testY = np.expand_dims(testY, axis=3)

    return trainX, trainY, valX, valY, testX, testY

def loadData_weather(args):
    # windpower
    df = pd.read_hdf(args.wind_power_file)
    df = df.iloc[:len(df)].sort_index()
    df.index = pd.to_datetime(df.index)
    wind_power = h5load(args.wind_power_file)
    wind_power_mode1 = h5load(args.wind_power_file_mode1)
    wind_power_mode2 = h5load(args.wind_power_file_mode2)
    wind_power_mode3 = h5load(args.wind_power_file_mode3)
    wind_power_mode4 = h5load(args.wind_power_file_mode4)
    wind_power_mode5 = h5load(args.wind_power_file_mode5)
    wind_power_mode6 = h5load(args.wind_power_file_mode6)
    wind_power_mode7 = h5load(args.wind_power_file_mode7)
    # normalization
    mean, std = np.mean(wind_power), np.std(wind_power)

    wind_power = normalize(wind_power,args.use_norm)
    wind_power_mode1 = normalize(wind_power_mode1,args.use_norm)
    wind_power_mode2 = normalize(wind_power_mode2,args.use_norm)
    wind_power_mode3 = normalize(wind_power_mode3,args.use_norm)
    wind_power_mode4 = normalize(wind_power_mode4,args.use_norm)
    wind_power_mode5 = normalize(wind_power_mode5,args.use_norm)
    wind_power_mode6 = normalize(wind_power_mode6,args.use_norm)
    wind_power_mode7 = normalize(wind_power_mode7,args.use_norm)

    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    wind_train = wind_power[: train_steps]
    wind_val = wind_power[train_steps: train_steps + val_steps]
    wind_test = wind_power[-test_steps:]
    _, trainY = seq2instance(wind_train, args.P, args.Q)
    _, valY = seq2instance(wind_val, args.P, args.Q)
    _, testY = seq2instance(wind_test, args.P, args.Q)
    
    trainX_1, _, valX_1, _, testX_1, _ = windpowerProcess(args,wind_power_mode1, train_steps, val_steps, test_steps)
    trainX_2, _, valX_2, _, testX_2, _ = windpowerProcess(args,wind_power_mode2, train_steps, val_steps, test_steps)
    trainX_3, _, valX_3, _, testX_3, _ = windpowerProcess(args,wind_power_mode3, train_steps, val_steps, test_steps)
    trainX_4, _, valX_4, _, testX_4, _ = windpowerProcess(args,wind_power_mode4, train_steps, val_steps, test_steps)
    trainX_5, _, valX_5, _, testX_5, _ = windpowerProcess(args,wind_power_mode5, train_steps, val_steps, test_steps)
    trainX_6, _, valX_6, _, testX_6, _ = windpowerProcess(args,wind_power_mode6, train_steps, val_steps, test_steps)
    trainX_7, _, valX_7, _, testX_7, _ = windpowerProcess(args,wind_power_mode7, train_steps, val_steps, test_steps)

    trainX = np.concatenate((trainX_1, trainX_2), axis=3).astype(np.float16)
    trainX = np.concatenate((trainX, trainX_3), axis=3).astype(np.float16)
    trainX = np.concatenate((trainX, trainX_4), axis=3).astype(np.float16)
    trainX = np.concatenate((trainX, trainX_5), axis=3).astype(np.float16)
    trainX = np.concatenate((trainX, trainX_6), axis=3).astype(np.float16)
    trainX = np.concatenate((trainX, trainX_7), axis=3).astype(np.float16)
    valX = np.concatenate((valX_1, valX_2), axis=3).astype(np.float16)
    valX = np.concatenate((valX, valX_3), axis=3).astype(np.float16)
    valX = np.concatenate((valX, valX_4), axis=3).astype(np.float16)
    valX = np.concatenate((valX, valX_5), axis=3).astype(np.float16)
    valX = np.concatenate((valX, valX_6), axis=3).astype(np.float16)
    valX = np.concatenate((valX, valX_7), axis=3).astype(np.float16)
    testX = np.concatenate((testX_1, testX_2), axis=3).astype(np.float16)
    testX = np.concatenate((testX, testX_3), axis=3).astype(np.float16)
    testX = np.concatenate((testX, testX_4), axis=3).astype(np.float16)
    testX = np.concatenate((testX, testX_5), axis=3).astype(np.float16)
    testX = np.concatenate((testX, testX_6), axis=3).astype(np.float16)
    testX = np.concatenate((testX, testX_7), axis=3).astype(np.float16)

    # spatial embedding
    f = open(args.SE_file, mode='r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(float(temp[0])), int(float(temp[1]))
    SE = np.zeros(shape=(N, dims), dtype=np.float32)
    for line in lines[1:]:
        temp = line.split(' ')
        index = int(float(temp[0]))
        values = [np.float16(val) for val in temp[1:]]
        SE[int(index)] = values

    # temporal embedding
    Time = df.index
    month = np.reshape(Time.month, newshape=(-1, 1))  # 从索引中提取月份信息
    dayofweek = np.reshape(Time.day, newshape=(-1, 1))
    # timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
    #            // Time.freq.delta.total_seconds()
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // 600
    timeofday = np.reshape(timeofday, newshape=(-1, 1))
    Time = np.concatenate((month, dayofweek, timeofday), axis=-1)
    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps: train_steps + val_steps]
    test = Time[-test_steps:]
    # shape = (num_sample, P + Q, 3)
    trainTE,_ = seq2instance(train, args.P, args.Q)
    valTE,_ = seq2instance(val, args.P, args.Q)
    testTE,_ = seq2instance(test, args.P, args.Q)

    weather_1 = normalize(h5load(args.U10_file))
    weather_2 = normalize(h5load(args.V10_file))
    weather_3 = normalize(h5load(args.U100_file))
    weather_4 = normalize(h5load(args.V100_file))

    wnum_step = weather_1.shape[0]
    wtrain_steps = round(args.train_ratio * wnum_step)
    wtest_steps = round(args.test_ratio * wnum_step)
    wval_steps = wnum_step - wtrain_steps - wtest_steps

    train_1, val_1, test_1 = weatherProcess(args,weather_1, wtrain_steps, wval_steps, wtest_steps)
    train_2, val_2, test_2 = weatherProcess(args,weather_2, wtrain_steps, wval_steps, wtest_steps)
    train_3, val_3, test_3 = weatherProcess(args,weather_3, wtrain_steps, wval_steps, wtest_steps)
    train_4, val_4, test_4 = weatherProcess(args,weather_4, wtrain_steps, wval_steps, wtest_steps)

    trainW = np.concatenate((train_1, train_2), axis=3).astype(np.float16)
    valW = np.concatenate((val_1, val_2), axis=3).astype(np.float16)
    testW = np.concatenate((test_1, test_2), axis=3).astype(np.float16)
    trainW = np.concatenate((trainW, train_3), axis=3).astype(np.float16)
    valW = np.concatenate((valW, val_3), axis=3).astype(np.float16)
    testW = np.concatenate((testW, test_3), axis=3).astype(np.float16)
    trainW = np.concatenate((trainW, train_4), axis=3).astype(np.float16)
    valW = np.concatenate((valW, val_4), axis=3).astype(np.float16)
    testW = np.concatenate((testW, test_4), axis=3).astype(np.float16)

    return (trainX, trainTE, trainY, trainW, valX, valTE, valY, valW, testX, testTE, testY, testW, SE, mean, std)

def load_variable_from_hdf5(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][()]
        return data.astype(np.float32) if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)

def load_alldata(folder_path):
    # 定义变量名列表
    variable_names = ['trainX', 'trainTE', 'trainY', 'trainW', 'valX', 'valTE', 'valY', 'valW', 'testX', 'testTE', 'testY', 'testW', 'SE', 'mean', 'std']

    # 初始化空列表以保存加载的变量
    loaded_variables = []

    # 逐个加载变量
    for name in variable_names:
        # 构造 hdf5 文件路径
        file_path = os.path.join(folder_path, f'{name}.h5')
        # 加载变量
        loaded_variables.append(load_variable_from_hdf5(file_path, name))

    return tuple(loaded_variables)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# plot train_val_loss
def plot_train_val_loss(train_total_loss, val_total_loss, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_total_loss) + 1), train_total_loss, c='b', marker='s', label='Train')
    plt.plot(range(1, len(val_total_loss) + 1), val_total_loss, c='r', marker='o', label='Validation')
    plt.legend(loc='best')
    plt.title('Train loss vs Validation loss')
    plt.savefig(file_path)
    
def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()  # 关闭当前图形以释放内存