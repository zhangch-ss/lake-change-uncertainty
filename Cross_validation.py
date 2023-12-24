import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from scipy.optimize import curve_fit
import os
import math
# data_cryosat = pd.read_excel('gh_Initial_point_set/Cross_validation.xlsx', header=None, sheet_name='CryoSat2')
# print(data_cryosat)
# data_icesat = pd.read_excel('gh_Initial_point_set/Cross_validation.xlsx', header=None, sheet_name='ICESat2')
# data_sentinel = pd.read_excel('gh_Initial_point_set/Cross_validation.xlsx', header=None, sheet_name='Sentinel3')
# for i in range(10):
#     random.randint(0, 9)
#
# sample_cryosat = random.shuffle(data_cryosat)
# print(sample_cryosat)
# sample_icesat = random.sample(list(data_icesat), 4)
# sample_sentinel = random.sample(list(data_sentinel), 14)
# print(sample_cryosat)
#
#
#
# np.get_include()
# 读数据——面积-水位数据对



def Leave_One_Out_batch(x, y, n):
    x = np.array(x, dtype=float)  # transform your data in a numpy array of floats
    y = np.array(y, dtype=float)  # so the curve_fit can work
    AE_list = []
    for i in range(len(x)):
        # print(i)
        y_ = y[i]
        x_ = x[i]
        xn = np.delete(x, i)
        yn = np.delete(y, i)
        # print(len(xn))
        parameter = np.polyfit(xn, yn, n)
        p = np.poly1d(parameter)
        # 相对误差的绝对值
        ae = np.abs(y_ - p(x_))
        AE_list.append(ae)
    return AE_list, np.mean(AE_list)


def Leave_One_Out(data, n):
    x = data.iloc[:, 0]
    # print(x)
    y = data.iloc[:, 1]
    x = np.array(x, dtype=float)  # transform your data in a numpy array of floats
    y = np.array(y, dtype=float)  # so the curve_fit can work
    AE_list = []
    AE_list1 = []
    for i in range(len(x)):
        # print(i)
        y_ = y[i]
        x_ = x[i]
        xn = np.delete(x, i)
        yn = np.delete(y, i)
        # print(len(xn))
        parameter = np.polyfit(xn, yn, n)
        p = np.poly1d(parameter)
        # 相对误差的绝对值
        ae1 = np.abs(y_ - p(x_))
        ae = y_ - p(x_)
        AE_list.append(ae)
        AE_list1.append(ae1)
    return AE_list, np.mean(AE_list1)


def log_func(x, a, b):
    return a*np.log(x) + b


def exp_func(x, a, b):
    return a*np.exp(b*x)

# def Exponential_function(x, a, b):
#     function_ = a * np.log(x) + b
#     # function_ = a * (np.exp(b * x))
#     return function_


def liuyi_batch(x, y, before_e=True):
    # x = data.iloc[:, 0]
    # # print(x)
    # y = data.iloc[:, 1]
    # x = np.array(x, dtype=float)  # transform your data in a numpy array of floats
    # y = np.array(y, dtype=float)  # so the curve_fit can work
    AE_list = []
    AE_list1 = []
    for i in range(len(x)):
        # print(i)
        y_ = y[i]
        x_ = x[i]
        # print(x_)
        xn = np.delete(x, i)
        yn = np.delete(y, i)
        # print(len(xn))
        if before_e:
            popt, pcov = curve_fit(exp_func, xn, yn, p0=(min(xn), before_e))
            # print(popt)
            y_pre = exp_func(x_, popt[0], popt[1])
            # 相对误差的绝对值
            ae1 = np.abs(y_ - y_pre)
            ae = y_ - y_pre
            AE_list.append(ae)
            AE_list1.append(ae1)
        else:
            popt, pcov = curve_fit(log_func, xn, yn)
            # print(popt)
            y_pre = log_func(x_, popt[0], popt[1])
            # 相对误差的绝对值
            ae1 = np.abs(y_ - y_pre)
            ae = y_ - y_pre
            AE_list.append(ae)
            AE_list1.append(ae1)
    MAE=np.mean(AE_list1)
    return AE_list, MAE

def liuyi(data, before_e=True):
    x = data.iloc[:, 0]
    # print(x)
    y = data.iloc[:, 1]
    x = np.array(x, dtype=float)  # transform your data in a numpy array of floats
    y = np.array(y, dtype=float)  # so the curve_fit can work
    AE_list = []
    AE_list1 = []
    for i in range(len(x)):
        # print(i)
        y_ = y[i]
        x_ = x[i]
        # print(x_)
        xn = np.delete(x, i)
        yn = np.delete(y, i)
        # print(len(xn))
        if before_e:
            popt, pcov = curve_fit(exp_func, xn, yn, p0=(min(xn), before_e))
            # print(popt)
            y_pre = exp_func(x_, popt[0], popt[1])
            # 相对误差的绝对值
            ae1 = np.abs(y_ - y_pre)
            ae = y_ - y_pre
            AE_list.append(ae)
            AE_list1.append(ae1)
        else:
            popt, pcov = curve_fit(log_func, xn, yn)
            # print(popt)
            y_pre = log_func(x_, popt[0], popt[1])
            # 相对误差的绝对值
            ae1 = np.abs(y_ - y_pre)
            ae = y_ - y_pre
            AE_list.append(ae)
            AE_list1.append(ae1)
    MAE=np.mean(AE_list1)
    return AE_list, MAE

# def liuyi_(data):
#     x = data.iloc[:, 0]
#     y = data.iloc[:, 1]
#     x = np.array(x, dtype=float)  # transform your data in a numpy array of floats
#     y = np.array(y, dtype=float)  # so the curve_fit can work
#     AE_list = []
#     for i in range(len(x)):
#         # print(i)
#         y_ = y[i]
#         x_ = x[i]
#         # print(x_)
#         xn = np.delete(x, i)
#         yn = np.delete(y, i)
#         # print(len(xn))
#         popt, pcov = curve_fit(Exponential_function, xn, yn)
#         # print(popt)
#         y_pre = Exponential_function(x_, popt[0], popt[1])
#         print(y_pre)
#         # 相对误差的绝对值
#         ae = np.abs(y_ - y_pre)
#         AE_list.append(ae)
#     np.savetxt(os.path.join(result_path, 'AE_list_Exponential.txt'), AE_list)
#     print(np.mean(AE_list))


if __name__ == 'main':
    data = pd.read_excel('J:/湖泊面积_水位数据对/Tuosu/Tuosu.xlsx', header=None)
    print(data)
    # # liuyi_(data)
    # AE_list1 = Leave_One_Out(data, 1)
    # AE_list2 = Leave_One_Out(data, 2)
    # AE_list3 = Leave_One_Out(data, 3)
    #
    # result_path = os.path.abspath(os.path.dirname(data_file))
    #
    # np.savetxt(os.path.join(result_path, 'AE_list_Linear.txt'), AE_list1)
    # np.savetxt(os.path.join(result_path, 'AE_list_Quadratic_polynomial.txt'), AE_list2)
    # np.savetxt(os.path.join(result_path, 'AE_list_Cubic_polynomial.txt'), AE_list3)