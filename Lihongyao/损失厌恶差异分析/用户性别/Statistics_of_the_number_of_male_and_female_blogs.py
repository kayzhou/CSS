import pandas as pd
import os


# 将文件gender列数据放入列表list_data中
import csv
from collections import Counter

def get_date(pth):
    splited_str = pth.split('/')[-1]
    dot_index = splited_str.index('.')

    date_result = splited_str[:dot_index]

    return date_result

def count_gender(csvfile):
    def read_file_csv(csvfile):
        list_data = []
        with open(csvfile, encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:  # 将csv 文件中的数据保存到list_data中
                list_data.append(row[1])
                blog_time = row[0][0]
        return list_data  # 返回想要获取的那一列数据


    result_dict = Counter(read_file_csv(csvfile))
    result_dict = {
        'date': get_date(csvfile),
        'f': result_dict['f'],
        'm': result_dict['m']
    }

    target = pd.DataFrame(result_dict, index=[-1])
    #count = target.to_csv('count_gender.csv', encoding='utf-8-sig')
    return target


base_pth = 'C:/Users/Lenovo/yan/D/金融数据整理（2021）/bmgl/day_bmgl/'
csv_list = os.listdir(base_pth)
csv_list = [base_pth+i for i in csv_list]

target = pd.DataFrame(columns=[
        'date', 'f','m'
    ])

for _csv in csv_list:
    try:
        print(f"正在执行:{_csv}")
        temp_csv = count_gender(_csv)
        target = target.append(temp_csv, ignore_index=True)
    except Exception as e:
        print(f"出错了:{e}")
target.to_csv('C:/Users/Lenovo/yan/D/金融数据整理（2021）/bmgl/count_gender.csv', index=False)

# if __name__ == '__main__':
#     print(count_gender('C:/Users/Lenovo/yan/D/金融数据整理（2021）/bmgl/day_bmgl/2014-12-01.csv'))