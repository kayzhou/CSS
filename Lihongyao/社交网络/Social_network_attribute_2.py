import pandas as pd
import os
from utils import main_loop


if __name__ == '__main__':

    base_pth = 'C:/Users/Lenovo/yan/D/金融数据整理（2021）/file_retweet/union_group/'
    csv_list = os.listdir(base_pth)
    csv_list = [base_pth+i for i in csv_list]

    target_csv = pd.DataFrame(columns=[
        'number_of_nodes', 'number_of_edges', 'network_density',
        'number_connected_components', 'number_of_nodes_for_largest_subgraph',
        'number_of_edges_for_largest_subgraph', 'sub_density', 'diameter',
        'average_shortest_path_length', 'degree_assortativity_coefficient',
        'average_clustering','average_pageRank_score',
        'average_degree_centrality''average_degree','average_degree_sub'
    ])

    for _csv in csv_list:
        try:
            print(f"正在执行:{_csv}")
            temp_csv = main_loop(_csv)
            target_csv = target_csv.append(temp_csv, ignore_index=True)
        except Exception as e:
            print(f"出错了:{e}")
    target_csv.to_csv('target.csv', index=False)
