import csv
import networkx as nx
import numpy as np
import pandas as pd


#计算平均PageRank得分
#def get_average_pageRank_score(H):
    pr = nx.pagerank(H)
    #return np.average([pagerank_value for _,pagerank_value in pr.items()])

#找到最大的PageRank值与先找出子图中最大的节点，在找最大节点对应的pagerank值，发现二者结果一致（下面两个代码选择一个即可）
# def get_maxpageRank_score(H):
#     pr_ = nx.pagerank(H)
#     a = [pagerank_value for _,pagerank_value in pr_.items()]
#     sort_ = sorted(a,reverse=True)
#     return sort_[0]
#
#
# def get_pageRank_score(H):
#     degree_ = nx.degree(H)
#     d_degree = dict(degree_)
#     maxnode = list(d_degree.keys())[list(d_degree.values()).index(max(d_degree.values()))]
#     pr_ = nx.pagerank(H)
#     a = [pagerank_value for _,pagerank_value in pr_.items() if _ == maxnode]
#     return a

#计算average degree centrality指标
def get_average_degree_centrality(H):
    dc = nx.degree_centrality(H)
    return np.average([de_ce for _,de_ce in dc.items()])


#计算average degree指标
def get_average_degree(G):
    de = nx.degree(G)
    dict_de = dict(de)
    return np.average([de_ for _,de_ in dict_de.items()])

#计算最大子图的average degree
def get_average_degree_sub(H):
    desub = nx.degree(H)
    dict_desub = dict(desub)
    return np.average([de_sub for _,de_sub in dict_desub.items()])

#格式化输出日期
def get_date(pth):
    splited_str = pth.split('/')[-1]
    dot_index = splited_str.index('.')

    date_result = splited_str[:dot_index]

    return date_result

#创建网络
def main_loop(csv_pth):
    with open(csv_pth, 'r', encoding='UTF-8') as nodecsv:
        nodereader = csv.reader(nodecsv)  # 不读表头
        nodes = [n for n in nodereader][1:]

    # 边
    with open(csv_pth, 'r', encoding='UTF-8') as edgecsv:
        edgereader = csv.reader(edgecsv)
        edges_all = [tuple(e) for e in edgereader][1:]

    edges = [i[1:3] for i in edges_all]  # 提取后两列的数据
    node_names = [n[1] for n in nodes]

    G = nx.Graph()
    G.add_nodes_from(node_names)  # 添加节点
    G.add_edges_from(edges)  # 添加边

    lst = list(G.subgraph(c) for c in nx.connected_components(G))  # 提取图中所有连通子图，返回一个列表，默认按照结点数量由大到小排序
    H = lst[0]  # 取顶点数最多连通子图来分析

#下面是保存csv文件所需的字段

    pagerank = get_pageRank_score(H)
    number_of_nodes = nx.number_of_nodes(G)
    number_of_edges = nx.number_of_edges(G)
    network_density = nx.density(G)
    number_connected_components = nx.number_connected_components(G)
    number_of_nodes_for_largest_subgraph = nx.number_of_nodes(H)
    number_of_edges_for_largest_subgraph = nx.number_of_edges(H)
    sub_density = nx.density(H)
    diameter = nx.diameter(H)
    average_shortest_path_length = nx.average_shortest_path_length(H)
    degree_assortativity_coefficient = nx.degree_assortativity_coefficient(H)
    average_clustering = nx.average_clustering(H)
    average_pageRank_score = get_average_pageRank_score(H)
    average_degree_centrality = get_average_degree_centrality(H)

    average_degree = get_average_degree(G)
    average_degree_sub = get_average_degree_sub(H)

    temp_dict = {
        'date': get_date(csv_pth),
        'pagerank' : pagerank,
        'number_of_nodes': number_of_nodes,
        'number_of_edges': number_of_edges,
        'network_density': network_density,
        'number_connected_components': number_connected_components,
        'number_of_nodes_for_largest_subgraph': number_of_nodes_for_largest_subgraph,
        'number_of_edges_for_largest_subgraph': number_of_edges_for_largest_subgraph,
        'sub_density': sub_density,
        'diameter': diameter,
        'average_shortest_path_length': average_shortest_path_length,
        'degree_assortativity_coefficient': degree_assortativity_coefficient,
        'average_clustering': average_clustering,
        'average_pageRank_score': average_pageRank_score,
        'average_degree_centrality': average_degree_centrality
        'average_degree':average_degree,
        'average_degree_sub':average_degree_sub
    }
    temp_dataframe = pd.DataFrame(temp_dict, index=[0])

    return temp_dataframe

