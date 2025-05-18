import numpy as np
import math
import lmfit
import json
import os

import pandas as pd
from pandas import Series
from scipy import optimize
from matplotlib import ticker
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.optimize import leastsq


def union_data(x_data, y_data, posi):
    """
    间隔为0.2
    """
    interval = 0.2
    X = []
    Y = []
    if posi == 1:
        x_y = {round(i, 1): [] for i in np.arange(0, 6.4 + interval, interval)}
    elif posi == -1:
        x_y = {round(i, 1): [] for i in np.arange(-8.0, interval, interval)}

    for x, y in zip(x_data, y_data):
        k = round(int(x / interval) * interval, 1)  # 取 interval
        try:
            x_y[k].append(y)
        except:
            pass

    for x in x_y:
        y_list = x_y[x]
        if y_list:
            y = np.array(y_list).mean()
        else:
            y = -1
        #         x_y[x] = y
        X.append(x)
        Y.append(y)

    last = 10000
    for i in range(len(Y)):
        if Y[i] != -1:
            last = Y[i]
        if Y[i] == -1:  # 找到第一个0
            for j in range(i + 1, len(Y)):
                if Y[j] != -1:  # 找到最近的一个非0
                    now = Y[j]
                    break
            else:
                for m in range(i, len(Y)):
                    Y[m] = last
                return X, Y

            gap = (now - last) / (j - i + 1)
            for k in range(i, j):
                Y[k] = last + gap * (k - i + 1)
    return X, Y


def my_line(is_disgust, x_n, y_n, x, y, **kw):
    '''
    牛熊市下用户行为的差异
    :param x_n: 收益率小于0
    :param y_n:
    :param x: 收益率大于0
    :param y:
    :return:
    '''

    def linear(input_x, input_y):
        '''
        线性回归
        :param x:
        :param y:
        :return: 拟合参数
        '''
        clf = LinearRegression()
        X = [[i] for i in input_x.tolist()]
        y = input_y.tolist()
        clf.fit(X, y)
        return clf.coef_[0], clf.intercept_, clf.score(X, y)

    def liner_model(k, b):
        def model(x):
            return b + k * x

        return model

    res = {}
    plt.figure(figsize=(12, 5))

    plt.scatter(x_n, y_n, alpha=0.3, label='$r < 0$')
    plt.scatter(x, y, alpha=0.3, label='$r > 0$')

    union_x_n, union_y_n = union_data(x_n, y_n, posi=-1)
    union_x, union_y = union_data(x, y, posi=1)
    plt.plot(union_x_n, union_y_n, c='grey')
    plt.plot(union_x, union_y, c='grey')

    # 正
    print('return > 0')

    # 画拟合直线
    print('---------- 收益率 < 0 ----------')
    a_n, b_n, r2 = linear(x_n, y_n)

    a_n_minus = a_n
    r2_minus = r2

    print('相关性：', x_n.corr(y_n))
    print('参数：', a_n, b_n)
    res['k_n'] = a_n
    res['b_n'] = b_n
    print('R^2：', r2)
    mod_n = liner_model(a_n, b_n)
    p_0 = mod_n(0)
    p_8 = mod_n(-8)
    p_4 = mod_n(4)
    plt.plot([-8, 0], [p_8, p_0], '-', c='purple', linewidth=2)
    plt.plot([0, 4], [p_0, p_4], 'r--', linewidth=2)

    pars = lmfit.Parameters()
    pars.add_many(('k', 1000), ('b', 1000))

    def residual(p):
        return p['k'] * x_n + p['b'] - y_n

    mini = lmfit.Minimizer(residual, pars)
    result = mini.minimize()
    print(lmfit.fit_report(result.params))

    print('---------- 收益率 > 0 ----------')
    a, b, r2 = linear(x, y)

    a_plus = a
    r2_plus = r2

    print('相关性：', x.corr(y))
    print('参数：', a, b)
    res['k'] = a
    res['b'] = b
    print('R^2：', r2)
    mod = liner_model(a, b)
    p_0 = mod(0)
    p_8 = mod(8)
    plt.plot([0, 8], [p_0, p_8], 'r-', linewidth=1.8)

    pars = lmfit.Parameters()
    pars.add_many(('k', -1000), ('b', 1000))

    def residual(p):
        return p['k'] * x + p['b'] - y

    mini = lmfit.Minimizer(residual, pars)
    result = mini.minimize()
    print(lmfit.fit_report(result.params))

    print('- - - - - - - - - - 损失厌恶指数 - - - - - - - - - -')
    print('斜率比例：', a_n / a)
    slope_ratio = a_n / a
    the_point = (b - b_n) / (a_n - a)
    balance_point = the_point
    print('平衡点：', the_point)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    plt.scatter(the_point, mod(the_point), c='k', linewidths=3, zorder=100, label='balance point')

    if is_disgust:
        temp_dict = {'disgust_a_n_minus': a_n_minus, 'disgust_r2_minus': r2_minus, 'disgust_a_plus': a_plus, 'disgust_r2_plus': r2_plus,
                     'disgust_slope_ratio': slope_ratio, 'disgust_balance_point': balance_point}
        temp_df = pd.DataFrame(temp_dict, index=[0])
    else:
        temp_dict = {'a_n_minus': a_n_minus, 'r2_minus': r2_minus, 'a_plus': a_plus, 'r2_plus': r2_plus,
                     'slope_ratio': slope_ratio, 'balance_point': balance_point}
        temp_df = pd.DataFrame(temp_dict, index=[0])

    plt.xlim(-9, 9)
    plt.xlabel('$return$', fontsize=15)
    if 'ylabel' in kw:
        plt.ylabel(kw['ylabel'], fontsize=15)
    else:
        plt.ylabel('amount of tweets', fontsize=15)
    plt.legend(loc=1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=12)
    #     plt.grid()
    if 'out' in kw:
        plt.savefig(kw['out'], dpi=300)
    # plt.show()
    # plt.close('all')

    # -----------------------------------------------------

    print('收益率为0：', mod_n(0) / mod(0))
    print('收益率为1：', mod_n(-1) / mod(1))
    print('收益率为2：', mod_n(-2) / mod(2))
    print('收益率为4：', mod_n(-4) / mod(4))
    print('收益率为6：', mod_n(-6) / mod(6))

    plt.figure(figsize=(6, 4))
    plt.xlabel('$r$', fontsize=15)
    #     plt.xlim(0, 10)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    x = np.arange(0, 8, 0.1)
    y = [mod_n(-i) / mod(i) for i in x]
    res['y_ratio'] = y
    plt.plot(x, y)
    # plt.show()
    # plt.close('all')

    plt.figure(figsize=(6, 4))
    plt.xlabel('$r$', fontsize=15)
    #     plt.xlim(0, 10)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    len_union_x = len(union_x)
    #     print(union_x_n, union_y_n)
    #     print(union_x, union_y)
    union_y_n.reverse()
    union_ratio = [union_y_n[i] / union_y[i] for i in range(len_union_x)]
    res['union_y_ratio'] = union_ratio
    res['union_x'] = union_x
    plt.plot(union_x, union_ratio)
    #     plt.legend(loc=1)
    # plt.show()
    # plt.close('all')

    res['union_y_n'] = union_y_n
    res['union_y'] = union_y

    # 返回可以评估
    res['ratio'] = a_n / a
    res['point'] = the_point

    return res, temp_df


def my_nonlinear_model(p):
    def this_model(x):
        return p['A1'] * (x ** p['A2']) + p['b']

    return this_model


def my_nonline(x_n, y_n, x, y, **kw):
    """
    拟合损失厌恶曲线
    注意：需要初始化参数
    """
    plt.figure(figsize=(12, 5))
    plt.scatter(x_n, y_n, alpha=0.3, label='$r < 0$')
    plt.scatter(x, y, alpha=0.3, label='$r > 0$')

    union_x_n, union_y_n = union_data(x_n, y_n, posi=-1)
    union_x, union_y = union_data(x, y, posi=1)
    plt.plot(union_x_n, union_y_n, c='grey')
    plt.plot(union_x, union_y, c='grey')

    # 正
    print('return > 0')
    pars = lmfit.Parameters()
    pars.add_many(('A1', 1000), ('A2', 1), ('b', 4000))

    def residual(p):
        return p['A1'] * (x ** p['A2']) + p['b'] - y

    mini = lmfit.Minimizer(residual, pars)
    result = mini.minimize()
    print(lmfit.fit_report(result.params))
    # ci1 = lmfit.conf_interval(mini, result)
    # lmfit.printfuncs.report_ci(ci1)

    mod_1 = my_nonlinear_model(result.params)
    x_fit = np.arange(0, 6, 0.1)
    y_fit = [mod_1(x) for x in x_fit]
    plt.plot(x_fit, y_fit, 'r-', linewidth=2)

    print('- * - ' * 15)

    # 负
    print('return < 0')
    pars = lmfit.Parameters()
    pars.add_many(('A1', -kw['k_n']), ('A2', 1.5), ('b', kw['b_n']))
    x_n = [-x for x in x_n]

    def residual(p):
        return p['A1'] * (x_n ** p['A2']) + p['b'] - y_n

    mini = lmfit.Minimizer(residual, pars)

    try:
        result = mini.minimize()
        print(lmfit.fit_report(result.params))
        # ci2 = lmfit.conf_interval(mini, result)
        # lmfit.printfuncs.report_ci(ci2)

        mod_2 = my_nonlinear_model(result.params)
        x_fit = np.arange(0, 8, 0.1)
        y_fit = [mod_2(x) for x in x_fit]
        x_fit = [-x for x in x_fit]
        plt.plot(x_fit, y_fit, '-', c='purple', linewidth=2)

        # 画图配置及输出
        plt.xlim(-9, 9)
        plt.xlabel('$return$', fontsize=15)
        if 'ylabel' in kw:
            plt.ylabel(kw['ylabel'], fontsize=15)
        else:
            plt.ylabel('amount of tweets', fontsize=15)
        plt.legend(loc=1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=12)
        #     plt.grid()
        if 'out' in kw:
            plt.savefig(kw['out'], dpi=300)
        # plt.show()
        # plt.close('all')

        # 比值
        print('收益率为0.5：', mod_2(0.5) / mod_1(0.5))
        print('收益率为1：', mod_2(1) / mod_1(1))
        print('收益率为2：', mod_2(2) / mod_1(2))
        print('收益率为4：', mod_2(4) / mod_1(4))
        print('收益率为6：', mod_2(6) / mod_1(6))

        plt.figure(figsize=(6, 4))
        plt.xlabel('$r$', fontsize=15)
        #     plt.xlim(0, 10)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        x = np.arange(0, 8, 0.1)
        y = [mod_2(i) / mod_1(i) for i in x]
        kw['non_y_ratio'] = y
        plt.plot(np.arange(0, 10, 1), [1] * 10)
        plt.plot(x, y)
        plt.plot(kw['union_x'], kw['union_y_ratio'])
        #     plt.legend(loc=1)
        plt.plot(np.arange(0, 10, 1), [1] * 10, alpha=0.1, linewidth=4)
        # plt.show()
        # plt.close('all')

        return kw
    except Exception as e:
        print(f"出錯了:{e}")

# 合并线性和非线性分析


def my_analysis(is_disgust, x_n, y_n, x, y, **kw):
    res, temp_df = my_line(is_disgust, x_n, y_n, x, y, **kw)
    res = dict(kw, **res)
    res = my_nonline(x_n, y_n, x, y, **res)
    return res, temp_df