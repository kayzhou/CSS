import pandas as pd
import os


def get_date(pth):
    splited_str = pth.split('/')[-1]
    dot_index = splited_str.index('.')
    date_result = splited_str[:dot_index]
    return date_result


def count_fans(csvfile):
    data = pd.read_csv(csvfile,encoding = 'utf-8')
    data['fol1'] = data['followers_count'][(data['followers_count']>=0)&(data['followers_count']<100)]
    data['fol2'] = data['followers_count'][(data['followers_count']>=100)&(data['followers_count']<10000)]
    data['fol3'] = data['followers_count'][data['followers_count']>=10000]
    result_dict = {
        'date': get_date(csvfile),
        'fol1': len(data['fol1'])-data['fol1'].isnull().sum(axis=0),
        'fol2': len(data['fol2'])-data['fol2'].isnull().sum(axis=0),
        'fol3': len(data['fol3'])-data['fol3'].isnull().sum(axis=0)
    }

    target = pd.DataFrame(result_dict, index=[-1])
    return target


base_pth = 'C:/Users/Lenovo/yan/D/金融数据整理（2021）/bf/day_bf/'
csv_list = os.listdir(base_pth)
csv_list = [base_pth+i for i in csv_list]

target = pd.DataFrame(columns=[
        'date', 'fol1','fol2','fol3'
    ])

for _csv in csv_list:
    try:
        print(f"正在执行:{_csv}")
        temp_csv = count_fans(_csv)
        target = target.append(temp_csv, ignore_index=True)
    except Exception as e:
        print(f"出错了:{e}")
target.to_csv('C:/Users/Lenovo/Desktop/count_fans.csv', index=False)
