from utils import *
import os
import re
import pandas as pd
from matplotlib import pyplot as plt
import os


def is_exist_target_df(path):
    file_pth = 'target_' + path + '.csv'
    if os.path.exists(file_pth):
        os.remove(file_pth)


if __name__ == '__main__':

    base_path = "C:/Users/Lenovo/yan/D/document/"
    path_lists = [i for i in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, i)) and '_' not in i and '.csv' not in i]

    for path in path_lists:
        path_list = os.listdir(os.path.join(base_path, path))
        path_list.sort(key=lambda x : [int(s) for s in re.findall(r'\d+', x)] )

        is_exist_target_df(path)

        target_df = pd.DataFrame(columns=['a_n_minus', 'r2_minus', 'a_plus', 'r2_plus', 'slope_ratio', 'balance_point',
                                          'disgust_a_n_minus', 'disgust_r2_minus', 'disgust_a_plus', 'disgust_r2_plus',
                                          'disgust_slope_ratio', 'disgust_balance_point'])
        temp_path = path
        path = os.path.join(base_path, path)

        for filename in path_list:
            #print(filename)
            #print(os.path.join(path,filename))
            #f = open(os.path.join(path,filename))
            #with open(os.path.join(path,filename)) as f:
            data = pd.read_csv(os.path.join(path,filename))
            #print(data)  #是一个字符串格式
            # 收益率与总量
            print("收益率与总量")
            x_n = data.close[data.close < 0]
            y_n = data.su[data.close < 0]
            x = data.close[data.close > 0]
            y = data.su[data.close > 0]
            _, temp_df = my_analysis(False, x_n, y_n, x, y, hi='Kay')

            print("厌恶")
            x_n = data.close[data.close < 0]
            y_n = data.disgust[data.close < 0]
            x = data.close[data.close > 0]
            y = data.disgust[data.close > 0]
            r2, disgust_temp_df = my_analysis(True, x_n, y_n, x, y, hi='Kay')

            temp = pd.concat([temp_df, disgust_temp_df], axis=1)
            target_df = target_df.append(temp, ignore_index=True)

        target_file_pth = 'target_' + temp_path + '.csv'
        target_df.to_csv(target_file_pth, index=False)
        print(f'{temp_path}文件夹数据写入成功')