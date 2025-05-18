import pandas as pd
import os


# 列表area_list中存放省份或直辖市的数据,city_list存放其他各市和区的数据
import csv
from collections import Counter

def get_date(pth):
    splited_str = pth.split('/')[-1]
    dot_index = splited_str.index('.')
    date_result = splited_str[:dot_index]

    return date_result

def count_city(csvfile):

    def read_file_csv(csvfile):
        area_list = []
        with open(csvfile, encoding='utf-8-sig') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:  # 将csv 文件中的数据保存到list_data中
                area_list.append(row[4])
                blog_time = row[0][0]
        return area_list

    def read_csv_city(csvfile):
        city_list = []
        with open(csvfile, encoding='utf-8-sig') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:  # 将csv 文件中的数据保存到list_data中
                city_list.append(row[5])
                blog_time = row[0][0]
        return city_list  # 返回想要获取的数据


    result_dict = Counter(read_file_csv(csvfile))
    city_dict = Counter(read_csv_city(csvfile))
    final_dict = {
        'date': get_date(csvfile),
        #京津冀城市群
        'beijing': result_dict['北京'],'tianjin': result_dict['天津'],'hebei': result_dict['河北'],
        #长三角城市群
        'shanghai': result_dict['上海'],'nanjing':city_dict['南京'],'suzhou': city_dict['苏州'],'wuxi': city_dict['无锡'],
        'changzhou': city_dict['常州'],'nantong': city_dict['南通'],'yangzhou': city_dict['扬州'],
        'yancheng': city_dict['盐城'],'taizhou': city_dict['泰州'],'zhenjiang': city_dict['镇江'],'hangzhou': city_dict['杭州'],
        'ningbo': city_dict['宁波'],'jiaxing': city_dict['嘉兴'],'huzhou': city_dict['湖州'],'shaoxing': city_dict['绍兴'],
        'jinhua': city_dict['金华'],'zhoushan': city_dict['舟山'],
        'taizhou_zhejiang': city_dict['台州'],'hefei':city_dict['合肥'],'wuhu':city_dict['芜湖'],
        'maanshan': city_dict['马鞍山'],'tongling':city_dict['铜陵'],'anqing':city_dict['安庆'],
        'chuzhou': city_dict['滁州'],'chizhou':city_dict['池州'],'xuancheng':city_dict['宣城'],
        #珠三角城市群
        'guangzhou': city_dict['广州'],'foshan':city_dict['佛山'],'zhaoqing':city_dict['肇庆'],
        'shenzhen': city_dict['深圳'],'dongguan':city_dict['东莞'],'huizhou':city_dict['惠州'],
        'zhuhai': city_dict['珠海'],'zhongshan':city_dict['中山'],'jiangmen':city_dict['江门'],
        #长江中游城市群
        'wuhan': city_dict['武汉'],'huangshi':city_dict['黄石'],'ezhou':city_dict['鄂州'],
        'huanggang': city_dict['黄冈'],'xiaogan':city_dict['孝感'],'xianning':city_dict['咸宁'],
        'xiantao': city_dict['仙桃'],'qianjiang':city_dict['潜江'],'tianmen':city_dict['天门'],
        'xiangyang': city_dict['襄阳'],'yichang':city_dict['宜昌'],'jingzhou':city_dict['荆州'],
        'jingmen': city_dict['荆门'],'changsha':city_dict['长沙'],'zhuzhou':city_dict['株洲'],
        'xiangtan': city_dict['湘潭'],'yueyang':city_dict['岳阳'],'yiyang':city_dict['益阳'],
        'changde': city_dict['常德'],'hengyang':city_dict['衡阳'],'loudi':city_dict['娄底'],
        'nanchang': city_dict['南昌'],'jiujiang':city_dict['九江'],'jingdezhen':city_dict['景德镇'],
        'yingtan': city_dict['鹰潭'],'xinyu':city_dict['新余'],'yichun':city_dict['宜春'],
        'pingxiang': city_dict['萍乡'],'shangrao':city_dict['上饶'],'fuzhou':city_dict['抚州'],
        'jian': city_dict['吉安'],
        #成渝城市群
        'yuzhong': city_dict['渝中区'],'wanzhou':city_dict['万州区'],'qianjiang_chongqing':city_dict['黔江区'],
        'fuling': city_dict['涪陵区'],'dadukou':city_dict['大渡口区'],'jiangbei':city_dict['江北区'],
        'shapingba': city_dict['沙坪坝区'],'jiulongpo':city_dict['九龙坡区'],'nanan':city_dict['南岸区'],
        'beibei': city_dict['北碚区'],'qijiang':city_dict['綦江县'],'dazu':city_dict['大足县'],
        'yubei': city_dict['渝北区'],'banan':city_dict['巴南区'],'changshou':city_dict['长寿区'],
        'jiangjin': city_dict['江津市'],'hechuan':city_dict['合川区'],'yongchuan':city_dict['永川区'],
        'nanchuan': city_dict['南川区'],'tongnan':city_dict['潼南县'],'tongliang':city_dict['铜梁县'],
        'rongchang': city_dict['荣昌县'],'bishan':city_dict['璧山县'],'liangping':city_dict['梁平县'],
        'fengdu': city_dict['丰都县'],'dianjiang':city_dict['垫江县'],'zhongxian':city_dict['忠县'],
        'kaixian': city_dict['开县'],'yunyang':city_dict['云阳县'],
        'chengdu': city_dict['成都'],'zigong':city_dict['自贡'],'luzhou':city_dict['泸州'],
        'deyang': city_dict['德阳'],'mianyang':city_dict['绵阳'],'suining':city_dict['遂宁'],
        'neijiang': city_dict['内江'],'leshan':city_dict['乐山'],'nanchong':city_dict['南充'],
        'meishan': city_dict['眉山'],'yibin':city_dict['宜宾'],'guangan':city_dict['广安'],
        'dazhou': city_dict['达州'],'yaan':city_dict['雅安'],'ziyang':city_dict['资阳'],
        #哈长城市群
        'haerbin': city_dict['哈尔滨'],'daqing':city_dict['大庆'],'qiqihaer':city_dict['齐齐哈尔'],
        'suihua': city_dict['绥化'],'mudanjiang':city_dict['牡丹江'],
        'changchun':city_dict['长春'],'jilin_jilin': city_dict['吉林'],'siping':city_dict['四平'],
        'liaoyuan':city_dict['辽源'],'songyuan': city_dict['松原'],'yanbianchaoxian':city_dict['延边朝鲜族自治州'],
        #中原城市群
        'zhengzhou': city_dict['郑州'],'kaifeng':city_dict['开封'],'luoyang':city_dict['洛阳'],'nanyang':city_dict['南阳'],
        'anyang': city_dict['安阳'],'shangqiu':city_dict['商丘'],'xinxiang':city_dict['新乡'],'pingdingshan':city_dict['平顶山'],
        'xuchang': city_dict['许昌'],'jiaozuo':city_dict['焦作'],'zhoukou':city_dict['周口'],'xinyang':city_dict['信阳'],
        'zhumadian': city_dict['驻马店'],'hebi':city_dict['鹤壁'],'puyang':city_dict['濮阳'],'luohe':city_dict['漯河'],
        'sanmenxia': city_dict['三门峡'],'jiyuan':city_dict['济源'],
        'changzhi': city_dict['长治'],'jincheng':city_dict['晋城'],'yuncheng':city_dict['运城'],
        'xingtai': city_dict['邢台'],'handan':city_dict['邯郸'],
        'liaocheng': city_dict['聊城'],'heze':city_dict['菏泽'],
        'huaibei': city_dict['淮北'],'bengbu':city_dict['蚌埠'],'suzhou_anhui':city_dict['宿州'],
        'fuyang': city_dict['阜阳'],'bozhou':city_dict['亳州']

    }

    target = pd.DataFrame(final_dict, index=[-1])
    return target


base_pth = 'C:/Users/Lenovo/yan/D/金融数据整理（2021）/bmgl/day_bmgl_city/'
csv_list = os.listdir(base_pth)
csv_list = [base_pth+i for i in csv_list]

target = pd.DataFrame(columns=[
    'date', 'beijing','tianjin','hebei','shanghai','nanjing','wuxi','changzhou','suzhou','nantong',
    'yancheng','yangzhou','zhenjiang','taizhou','hangzhou','ningbo','jiaxing','huzhou','shaoxing','jinhua',
    'zhoushan','taizhou_zhejiang','hefei','wuhu','maanshan','tongling','anqing','chuzhou','chizhou','xuancheng',
    'guangzhou','foshan','zhaoqing','shenzhen','dongguan','huizhou','zhuhai','zhongshan','jiangmen',
    'wuhan','huangshi','ezhou','huanggang','xiaogan','xianning','xiantao','qianjiang','tianmen','xiangyang',
    'yichang','jingzhou','jingmen','changsha','zhuzhou','xiangtan','yueyang','yiyang','changde','hengyang',
    'loudi','nanchang','jiujiang','jingdezhen','yingtan','xinyu','yichun','pingxiang','shangrao','fuzhou',
    'jian','yuzhong','wanzhou','qianjiang_chongqing','fuling','dadukou','jiangbei','shapingba','jiulongpo',
    'nanan','beibei','qijiang','dazu','yubei','banan','changshou','jiangjin','hechuan','yongchuan','nanchuan',
    'tongnan','tongliang','rongchang','bishan','liangping','fengdu','dianjiang','zhongxian','kaixian',
    'yunyang','chengdu','zigong','luzhou','deyang','mianyang','suining','neijiang','leshan','nanchong',
    'meishan','yibin','guangan','dazhou','yaan','ziyang','haerbin','daqing','qiqihaer','suihua','mudanjiang',
    'changchun','jilin_jilin','siping','liaoyuan','songyuan','yanbianchaoxian','zhengzhou','kaifeng','luoyang',
    'nanyang','anyang','shangqiu','xinxiang','pingdingshan','xuchang','jiaozuo','zhoukou','xinyang','zhumadian',
    'hebi','puyang','luohe','sanmenxia','jiyuan','changzhi','jincheng','yuncheng','xingtai','handan',
    'liaocheng','heze','huaibei','bengbu','suzhou_anhui','fuyang','bozhou'

])

for _csv in csv_list:
    try:
        print(f"正在执行:{_csv}")
        temp_csv = count_city(_csv)
        target = target.append(temp_csv, ignore_index=True)
    except Exception as e:
        print(f"出错了:{e}")
target.to_csv('C:/Users/Lenovo/Desktop/urban.csv',
              encoding='utf-8-sig',index=False)

# if __name__ == '__main__':
#     print(count_city('C:/Users/Lenovo/yan/D/金融数据整理（2021）/bmgl/day_bmgl_city/2014-12-01.csv'))