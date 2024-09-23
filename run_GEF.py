import argparse
from utils import tool
import time
import random
import torch
import numpy as np
import os
import pandas as pd
from model.model_GEF import Hiformer
from forcasting.long_term import Long_Term_Forecast

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
parser = argparse.ArgumentParser()

# model define
parser.add_argument('--time_slot', type=int, default=10,
                    help='a time step is 10 mins')
parser.add_argument('--P', type=int, default=96,
                    help='history steps')
parser.add_argument('--Q', type=int, default=48,
                    help='prediction steps')
parser.add_argument('--L', type=int, default=5,
                    help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=32,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=16,
                    help='dims of each head attention outputs')
parser.add_argument('--T', type=int, default= 144,
                    help='time step of one day T = 24 * 60 // args.time_slot')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn[default : 2048]')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=50,
                    help='epoch to run')
parser.add_argument('--patience', type=int, default=7,
                    help='patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.002,
                    help='initial learning rate')
parser.add_argument('--min_lr', type=float, default=0.0001,
                    help='min learning rate')
parser.add_argument('--decay_epoch', type=int, default=7,
                    help='decay epoch')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--use_norm', type=bool, default=True, help='normalize')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', type=int, default=1, help='use multi gpu')

# basic config
parser.add_argument('--wind_power_file', default='data/GEFcom/Wind_power.h5',
                            help='wind_power file')
parser.add_argument('--wind_power_file_mode1', default='data/GEFcom/windpower_mode1.h5',
                            help='wind_power file')
parser.add_argument('--wind_power_file_mode2', default='data/GEFcom/windpower_mode2.h5',
                            help='wind_power file')
parser.add_argument('--wind_power_file_mode3', default='data/GEFcom/windpower_mode3.h5',
                            help='wind_power file')
parser.add_argument('--wind_power_file_mode4', default='data/GEFcom/windpower_mode4.h5',
                            help='wind_power file')
parser.add_argument('--wind_power_file_mode5', default='data/GEFcom/windpower_mode5.h5',
                            help='wind_power file')
parser.add_argument('--wind_power_file_mode6', default='data/GEFcom/windpower_mode6.h5',
                            help='wind_power file')
parser.add_argument('--wind_power_file_mode7', default='data/GEFcom/windpower_mode7.h5',
                            help='wind_power file')
parser.add_argument('--U10_file', default='data/GEFcom/U10.h5',
                            help='wind_speed file')
parser.add_argument('--V10_file', default='data/GEFcom/V10.h5',
                            help='wind_speed file')
parser.add_argument('--U100_file', default='data/GEFcom/U100.h5',
                            help='wind_speed file')
parser.add_argument('--V100_file', default='data/GEFcom/V100.h5',
                            help='wind_speed file')
parser.add_argument('--SE_file', default='data/GEFcom/SE.txt',
                            help='spatial embedding file')
parser.add_argument('--train_ratio', type=float, default=0.7,
                            help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.1,
                            help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2,
                            help='testing set [default : 0.2]')
parser.add_argument('--model_file', default='data/GEFcom/Hiformer_best',
                    help='save the model to disk')
parser.add_argument('--best_model_file', default='data/GEFcom/Hiformer_best_all',
                    help='save the model to disk')
parser.add_argument('--data_file', default='./Temporary Files/GEFcom',
                    help='data file')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

# setting record of experiments
setting = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
    args.time_slot,
    args.P,
    args.Q,
    args.L,
    args.K,
    args.d,
    args.batch_size,
    args.max_epoch,
    args.patience,
    args.learning_rate,
    args.decay_epoch,
    args.use_gpu,
    args.gpu,
    args.use_multi_gpu
)

start = time.time()
directory = './figure/{}'.format(setting)
os.makedirs(directory, exist_ok=True)
log_file_path = os.path.join(directory, 'log(windpower)_march_Hiformer_best')
log = open(log_file_path, 'w')
tool.log_string(log, str(args)[10: -1])

# ========================== load model  =============================== #
tool.log_string(log, 'compiling model...')
model = Hiformer(args,bn_decay=0.1)
parameters = tool.count_parameters(model)
tool.log_string(log, 'trainable parameters: {:,}'.format(parameters))

if __name__ == '__main__':

    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    forecast = Long_Term_Forecast(model, args, log)
    start = time.time()
    loss_train, loss_val = forecast.train_model()
    tool.plot_train_val_loss(loss_train, loss_val, os.path.join(directory, 'train_val_loss.png'))
    trainX, valX, testX, train_gts, val_gts, test_gts = forecast.test_model()
    end = time.time()
    train_all = np.sum(trainX, axis=-1)
    val_vall = np.sum(valX, axis=-1)
    test_all = np.sum(testX, axis=-1)
    train_gts = np.sum(train_gts, axis=-1)
    val_gts = np.sum(val_gts, axis=-1)
    test_gts = np.sum(test_gts, axis=-1)
    tool.log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    tool.log_string(log, 'Test: %s\ttest_gts: %s' % (test_all.shape, test_gts.shape))
    log.close()
    l = [train_all, val_vall, test_all, train_gts, val_gts, test_gts]
    name = ['train_all', 'val_vall', 'test_all', 'train_gts', 'val_gts', 'test_gts']
    for i, data in enumerate(l):
        df = pd.DataFrame(data)
        file_path = os.path.join(directory, name[i] + '.csv')
        df.to_csv(file_path, index=False)
    # Plot a comparison chart of the test prediction value and the target value (optional)

    for i in range(100):
        pdf_filename = f'{directory}/{i + 1}.pdf'
        tool.visual(test_gts[i],test_all[i],pdf_filename)

    print("All plots have been saved as individual PDF files in the specified directory.")

