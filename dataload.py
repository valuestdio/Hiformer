import utils.tool
import os
import h5py
import gc 
from run import args
class load_data():

    def __init__(self):

        self.args = args

    # 定义保存单个变量到 HDF5 文件的函数
    def save_variable_to_hdf5(self,variable, file_path, dataset_name):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset(dataset_name, data=variable)
        return

    def main(self):
        # 获取数据
        (trainX, trainTE, trainY, trainW, valX, valTE, valY, valW, testX, testTE, testY, testW, SE, mean, std) = utils.tool.loadData_weather(self.args)

        # 设置临时文件夹路径为 "Temporary Files"
        temp_dir = 'Temporary Files'

        # 确保临时文件夹存在，如果不存在则创建
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # 保存每个变量到以其名字命名的 h5 文件中
        variables = [trainX, trainTE, trainY, trainW, valX, valTE, valY, valW, testX, testTE, testY, testW, SE, mean, std]
        variable_names = ['trainX', 'trainTE', 'trainY', 'trainW', 'valX', 'valTE', 'valY', 'valW', 'testX', 'testTE', 'testY', 'testW', 'SE', 'mean', 'std']

        for var, name in zip(variables, variable_names):
            self.save_variable_to_hdf5(var, os.path.join(temp_dir, f'{name}.h5'), name)
            del var  # 删除变量
            gc.collect()  # 垃圾回收

        return

# 创建 load_data 类的实例
data_loader = load_data()

# 调用 main 方法来执行主程序
data_loader.main()
