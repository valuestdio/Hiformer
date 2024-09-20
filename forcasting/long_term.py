import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from utils import tool
import time
import math
import datetime

class ExponentialLRSchedulerWithMinLr(_LRScheduler):
    def __init__(self, optimizer, gamma, min_lr, last_epoch=-1):
        self.gamma = gamma
        self.min_lr = min_lr
        super(ExponentialLRSchedulerWithMinLr, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr) for base_lr in self.base_lrs]

class Long_Term_Forecast(object):
    def __init__(self, model, args, log):
        super(Long_Term_Forecast, self).__init__()
        self.model = model
        self.args = args
        self.device_ids = list(range(torch.cuda.device_count()))
        self.log = log

    def _acquire_device(self):
        if self.args.use_gpu:
            if torch.cuda.is_available():
                if self.args.use_multi_gpu and len(self.device_ids) > 1:
                    # Use multiple GPUs
                    device = torch.device(f"cuda:{self.device_ids[0]}")  # Primary device for multi-GPU
                    print("main GPU used",device)
                else:
                    # Use single GPU
                    device = torch.device(f"cuda:{self.device_ids[0]}")  # Single GPU device
            else:
                print("Warning: GPU specified but no GPU available. Using CPU instead.")
                device = torch.device("cpu")
        else:
            # Use CPU
            device = torch.device("cpu")
        
        return device
    
    def _build_model(self, device):
        available_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {available_gpus}")
        if self.args.use_gpu and torch.cuda.is_available():
            if self.args.use_multi_gpu:
                model = nn.DataParallel(self.model, device_ids=self.device_ids)
                model = model.to(device)
            else:
                model = self.model.to(device)
        else:
            model = self.model.to(device)
        
        return model
    
    def _get_data(self):
        (trainX, trainTE, trainY, trainW, valX, valTE, valY,valW, testX, testTE, testY, testW, SE, mean, std) = tool.load_alldata(self.args.data_file)
        
        return trainX, trainTE, trainY, trainW, valX, valTE, valY,valW, testX, testTE, testY, testW, SE, mean, std

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _select_scheduler(self, num_train, optimizer):
        decay_epoch = self.args.decay_epoch
        batch_size = self.args.batch_size
        gamma = 0.5 ** (1 / (decay_epoch * num_train // batch_size))
        min_lr = self.args.min_lr  
        scheduler = ExponentialLRSchedulerWithMinLr(optimizer, gamma, min_lr)
        return scheduler
    
    def train_model(self):

        self.device = self._acquire_device()
        self.model = self._build_model(self.device)
        """
        PyTorch implementation of the training loop
        """
        # ========================== load dataset =============================== #
        tool.log_string(self.log, 'loading data...')
        trainX, trainTE, trainY, trainW, valX, valTE, valY, valW, testX, testTE, testY, testW, SE, mean, std = self._get_data()
        tool.log_string(self.log, 'trainX: %s\ttrainY: %s' % (trainX.shape, trainY.shape))
        tool.log_string(self.log, 'valX:   %s\t\tvalY:   %s' % (valX.shape, valY.shape))
        tool.log_string(self.log, 'testX:  %s\t\ttestY:  %s' % (testX.shape, testY.shape))
        tool.log_string(self.log, 'trainW: %s\t\tvalW: %s\t\ttestW:  %s' % (trainW.shape,valW.shape, testW.shape))
        tool.log_string(self.log, 'trainTE: %s\t\tvalTE: %s\t\ttestTE:  %s' % (trainTE.shape,valTE.shape, testTE.shape))
        tool.log_string(self.log, 'data loaded!')
        del testX, testTE, testY, testW
        num_train = trainX.shape[0]
        num_val = valX.shape[0]
        SE = np.repeat(SE, (self.args.P // SE.shape[1])+1, axis=1)
        SE = SE[:, :self.args.P]
        SE = np.expand_dims(SE, axis=0)
        SE = np.repeat(SE, self.args.batch_size, axis=0)
        SE = torch.tensor(SE, dtype=torch.float32).to(self.device)
        #mean = torch.tensor(mean, dtype=torch.float32).to(self.device)
        #std = torch.tensor(std, dtype=torch.float32).to(self.device)
        train_num_batch = math.floor(num_train / self.args.batch_size)
        val_num_batch = math.floor(num_val / self.args.batch_size)
        train_sample = train_num_batch*self.args.batch_size
        val_sample = val_num_batch*self.args.batch_size

        wait = 0
        val_loss_min = float('inf')
        best_model_wts = None
        train_total_loss = []
        val_total_loss = []
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = GradScaler()#Mixed Precision
        scheduler = self._select_scheduler(num_train, optimizer)
        tool.log_string(self.log, '**** training model ****')
        print(self.args.max_epoch)
        for epoch in range(self.args.max_epoch):
            print("------------------")
            print(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + " | epoch: " + str(epoch))
            if wait >= self.args.patience:
                tool.log_string(self.log, 'early stop at epoch: %04d' % (epoch))
                break
            
            # Shuffle the training data
            permutation = np.random.permutation(num_train)
            trainX = trainX[permutation]
            trainTE = trainTE[permutation]
            trainW = trainW[permutation]
            trainY = trainY[permutation]

            # Training phase
            self.model.train()
            start_train = time.time()
            train_loss = 0.0
            for batch_idx in range(train_num_batch):
                print(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + " | epoch: " + str(epoch) + " first loop batch idx: " + str(batch_idx))
                start_idx = batch_idx * self.args.batch_size
                end_idx = min(num_train, (batch_idx + 1) * self.args.batch_size)

                batchX = torch.tensor(trainX[start_idx:end_idx], dtype=torch.float32).to(self.device)
                batchTE = torch.tensor(trainTE[start_idx:end_idx], dtype=torch.float32).to(self.device)
                batchW = torch.tensor(trainW[start_idx:end_idx], dtype=torch.float32).to(self.device)
                batchY = torch.tensor(trainY[start_idx:end_idx], dtype=torch.float32).to(self.device)
                optimizer.zero_grad()
                with autocast():
                    output = self.model(batchX, SE, batchTE, batchW)
                    #output = self._renorlinear(output, mean, std)
                    batch_loss = criterion(output, batchY)
                    train_loss += batch_loss.item() * (end_idx - start_idx)
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # 手动调整批归一化动量
                bn_momentum = max(0.1, 0.5 * (0.99 ** (epoch * train_num_batch + batch_idx)))
                for m in self.model.modules():
                    if isinstance(m, nn.BatchNorm1d):
                        m.momentum = bn_momentum

                print(f'Training batch: {batch_idx+1} in epoch:{epoch}, training batch loss:{batch_loss:.4f}')

                del batchX, batchTE, batchW, batchY, output, batch_loss
            train_loss /= train_sample
            train_total_loss.append(train_loss)
            end_train = time.time()

            # Validation phase
            self.model.eval()
            start_val = time.time()
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx in range(val_num_batch):
                    print(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + " | epoch: " + str(epoch) + " second loop batch idx: " + str(batch_idx))
                    start_idx = batch_idx * self.args.batch_size
                    end_idx = min(num_val, (batch_idx + 1) * self.args.batch_size)

                    batchX = torch.tensor(valX[start_idx:end_idx], dtype=torch.float32).to(self.device)
                    batchTE = torch.tensor(valTE[start_idx:end_idx], dtype=torch.float32).to(self.device)
                    batchW = torch.tensor(valW[start_idx:end_idx], dtype=torch.float32).to(self.device)
                    batchY = torch.tensor(valY[start_idx:end_idx], dtype=torch.float32).to(self.device)

                    output = self.model(batchX, SE, batchTE, batchW)
                    #output = self._renorlinear(output, mean, std)
                    pred = output.detach().cpu()
                    true = batchY.detach().cpu()
                    batch_loss = criterion(pred, true)
                    val_loss += batch_loss.item() * (end_idx - start_idx)
                    del batchX, batchTE, batchW, batchY, output, pred, true, batch_loss
            val_loss /= val_sample
            val_total_loss.append(val_loss)
            end_val = time.time()

            tool.log_string(
                self.log,
                '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
                (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
                self.args.max_epoch, end_train - start_train, end_val - start_val))
            tool.log_string(
                self.log, 'train loss: %.4f, val_loss: %.4f' % (train_loss, val_loss))

            if val_loss <= val_loss_min:
                tool.log_string(
                    self.log,
                    'val loss decrease from %.4f to %.4f, saving model to %s' %
                    (val_loss_min, val_loss, self.args.model_file))
                wait = 0
                val_loss_min = val_loss
                best_model_wts = self.model.state_dict()
                torch.save(self.model.state_dict(), self.args.model_file)
            else:
                wait += 1
            print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()}")
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model, self.args.model_file)
        tool.log_string(self.log, f'Training and validation are completed, and model has been stored as {self.args.model_file}')
        torch.cuda.empty_cache()
        return train_total_loss, val_total_loss

    def test_model(self):
        """
        PyTorch implementation of the testing loop
        """
        self.device = self._acquire_device()
        trainX, trainTE, trainY, trainW, valX, valTE, valY,valW, testX, testTE, testY, testW, SE, mean, std = self._get_data()
        SE = np.repeat(SE, (self.args.P // SE.shape[1])+1, axis=1)
        SE = SE[:, :self.args.P]
        SE = np.expand_dims(SE, axis=0)
        SE = np.repeat(SE, self.args.batch_size, axis=0)
        SE = torch.tensor(SE, dtype=torch.float32).to(self.device)
        tool.log_string(self.log, '**** testing model ****')
        tool.log_string(self.log, 'loading model from %s' % self.args.model_file)
        model = torch.load(self.args.model_file)
        tool.log_string(self.log, 'model restored!')
        tool.log_string(self.log, 'evaluating...')
        model.to(self.device)
        model.eval()

        # Testing phase for train, val and test data
        def evaluate_model(dataX, dataTE, dataW):
            predictions = []
            with torch.no_grad():
                num_data = dataX.shape[0]
                num_batch = math.floor(num_data / self.args.batch_size)
                for batch_idx in range(num_batch):
                    start_idx = batch_idx * self.args.batch_size
                    end_idx = min(num_data, (batch_idx + 1) * self.args.batch_size)
                    batchX = torch.tensor(dataX[start_idx:end_idx], dtype=torch.float32).to(self.device)
                    batchTE = torch.tensor(dataTE[start_idx:end_idx], dtype=torch.float32).to(self.device)
                    batchW = torch.tensor(dataW[start_idx:end_idx], dtype=torch.float32).to(self.device)
    
                    output = model(batchX, SE, batchTE, batchW)
                    #output = self._renorlinear(output, mean, std)
                    output = output.detach().cpu().numpy()
                    predictions.append(output)
                    del batchX, batchTE, batchW, output
                predictions = torch.from_numpy(np.concatenate(predictions, axis=0))
            return predictions

        start_test = time.time()

        trainPred = evaluate_model(trainX, trainTE, trainW).numpy()
        valPred = evaluate_model(valX, valTE, valW).numpy()
        testPred = evaluate_model(testX, testTE, testW).numpy()

        end_test = time.time()
        trainY = trainY[:trainPred.shape[0], ...]
        valY = valY[:valPred.shape[0], ...]
        testY = testY[:testPred.shape[0], ...]
        
        train_mae,  train_rmse, train_mse = tool.metric(trainPred, trainY)
        val_mae, val_rmse, val_mse   = tool.metric(valPred, valY)
        test_mae, test_rmse, test_mse  = tool.metric(testPred, testY)

        tool.log_string(self.log, 'testing time: %.1fs' % (end_test - start_test))
        tool.log_string(self.log, '                MAE\t\tRMSE\t\tMSE')
        tool.log_string(self.log, 'train            %.2f\t\t%.2f\t\t%.2f' % 
                        (train_mae, train_rmse, train_mse))
        tool.log_string(self.log, 'val              %.2f\t\t%.2f\t\t%.2f' %
                        (val_mae, val_rmse, val_mse))
        tool.log_string(self.log, 'test             %.2f\t\t%.2f\t\t%.2f' %
                        (test_mae, test_rmse, test_mse))

        tool.log_string(self.log, 'performance in each prediction step')
        MAE, RMSE, MSE = [], [], []
        for q in range(self.args.Q):
            mae, rmse, mse = tool.metric(testPred[:, q], testY[:, q])
            MAE.append(mae)
            RMSE.append(rmse)
            MSE.append(mse)
            tool.log_string(self.log, 'step: %02d          %.2f\t\t%.2f\t\t%.2f' %
                            (q + 1, mae, rmse, mse))
        average_mae = np.mean(MAE)
        average_rmse = np.mean(RMSE)
        average_mse = np.mean(MSE)
        tool.log_string(self.log, 'average:          %.2f\t\t%.2f\t\t%.2f' %
                        (average_mae,  average_rmse, average_mse))

        end = time.time()
        tool.log_string(self.log, 'test time: %.1fmin' % ((end - start_test) / 60))
        torch.cuda.empty_cache()
        return trainPred, valPred, testPred, trainY, valY, testY
