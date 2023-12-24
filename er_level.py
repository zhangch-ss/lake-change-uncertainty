import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import sympy as sp
from sympy import poly
from Cross_validation import liuyi_batch, Leave_One_Out_batch
import os
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
from matplotlib.pyplot import MultipleLocator


# import scipy.special.


# print(area_level.head())

# area = area_level.iloc[:, 0]
# level = area_level.iloc[:, 1]
# 幂函数
def pow_func(x, a, b):
    return a * np.power(x, b)


# 指数函数
def exp_func(x, a, b):
    return a * np.exp(b * x)


# 对数函数
def log_func(x, a, b):
    return a * np.log(x) + b


# 指数函数导数
def D_exp_func(x, a, b):
    return a * (np.exp(b * x) * b)


# 幂函数导数
def D_pow_func(x, a, b):
    return a * b * np.power(x, b - 1)


# 对数函数导数
def D_log_func(x, a, b):
    return a / x


# 绘制幂函数曲线
def pow_curve(x, y):
    global AE_list_cv, MAE_cv
    # print(9e-6)
    popt, pcov = curve_fit(pow_func, x, y)
    # print(popt)
    # 获取0到最大面积的数值
    # x_0 = np.linspace(0, x.max() + (x.max() - x.min()) / 3, 9999)
    x_0 = np.linspace(x.min() - (x.max() - x.min()) / 3,
                      x.max() + (x.max() - x.min()) / 6, 9999)
    # 求该区间的导数值
    diffy_result = D_pow_func(x_0, popt[0], popt[1])
    # print(diffy_result)
    # 若该区间的导数全为0
    AE_list_nocv, MAE_nocv = liuyi_batch(x, y, before_e=False)
    if (np.array(diffy_result) > 0).all():
        AE_list_cv, MAE_cv = liuyi_batch(x, y, before_e=False)
    else:
        print('该函数非单调递增！')
    R2 = np.corrcoef(y, pow_func(x, popt[0], popt[1]))[0, 1] ** 2
    # print(R2)
    RMSE = metrics.mean_squared_error(y, pow_func(x, popt[0], popt[1])) ** 0.5
    return popt, R2, RMSE, AE_list_cv, MAE_cv, AE_list_nocv, MAE_nocv


# 绘制对数函数曲线
def log_curve(x, y):
    global AE_list_cv, MAE_cv
    # print(9e-6)
    popt, pcov = curve_fit(log_func, x, y)
    # print(popt)
    # x_0 = np.linspace(0, x.max() + (x.max() - x.min()) / 3, 9999)
    x_0 = np.linspace(x.min() - (x.max() - x.min()) / 2,
                      x.max() + (x.max() - x.min()) / 6, 9999)
    diffy_result = D_log_func(x_0, popt[0], popt[1])
    # print(diffy_result)
    AE_list_nocv, MAE_nocv = liuyi_batch(x, y, before_e=False)
    if (np.array(diffy_result) > 0).all():
        AE_list_cv, MAE_cv = liuyi_batch(x, y, before_e=False)
    else:
        print('该函数非单调递增！')
    R2 = np.corrcoef(y, log_func(x, popt[0], popt[1]))[0, 1] ** 2
    # print(R2)
    RMSE = metrics.mean_squared_error(y, log_func(x, popt[0], popt[1])) ** 0.5
    return popt, R2, RMSE, AE_list_cv, MAE_cv, AE_list_nocv, MAE_nocv


# 绘制指数函数曲线
def exp_curve(x, y, before_e):
    global MAE_cv, AE_list_cv
    # print(9e-6)
    popt, pcov = curve_fit(exp_func, x, y, p0=(min(x), before_e))
    # print(popt)
    # x_0 = np.linspace(0, x.max() + (x.max() - x.min()) / 3, 9999)
    x_0 = np.linspace(x.min() - (x.max() - x.min()) / 2,
                      x.max() + (x.max() - x.min()) / 6, 9999)
    diffy_result = D_exp_func(x_0, popt[0], popt[1])
    # print(diffy_result)
    AE_list_nocv, MAE_nocv = liuyi_batch(x, y, before_e)
    if (np.array(diffy_result) > 0).all():
        AE_list_cv, MAE_cv = liuyi_batch(x, y, before_e)
    else:
        print('该函数非单调递增！')
    R2 = np.corrcoef(y, exp_func(x, popt[0], popt[1]))[0, 1] ** 2
    # print(R2)
    RMSE = metrics.mean_squared_error(y, exp_func(x, popt[0], popt[1])) ** 0.5
    return popt, R2, RMSE, AE_list_cv, MAE_cv, AE_list_nocv, MAE_nocv


# 绘制多项式曲线1-5
def Polynomial_curve(x, y, deg):
    global MAE_cv, AE_list_cv
    parameter = np.polyfit(x, y, deg)
    # print(parameter)
    p = np.poly1d(parameter)
    # print(p)
    diffy = np.polyder(p, 1)
    # print(diffy)
    diffy_result = []
    # x_0 = np.linspace(0, x.max() + (x.max() - x.min()) / 3, 9999)
    x_0 = np.linspace(x.min() - (x.max() - x.min()) / 2,
                      x.max() + (x.max() - x.min()) / 6, 9999)
    for area_i in x_0:
        diffy_result.append(np.polyval(diffy, area_i))
    # print((np.array(diffy_result) > 0).all())
    AE_list_nocv, MAE_nocv = Leave_One_Out_batch(x, y, deg)
    if (np.array(diffy_result) > 0).all():
        AE_list_cv, MAE_cv = Leave_One_Out_batch(x, y, deg)
    else:
        print('该函数非单调递增！')
    R2 = np.corrcoef(y, p(x))[0, 1] ** 2
    RMSE = metrics.mean_squared_error(y, p(x)) ** 0.5
    return parameter, R2, RMSE, AE_list_cv, MAE_cv, AE_list_nocv, MAE_nocv


# 最优曲线选择
def Optimal_curve(excel_path):
    area_level = pd.read_excel(excel_path, header=None)
    x_ = area_level.iloc[:, 0]
    y_ = area_level.iloc[:, 1]
    x = np.array(x_, dtype=float)  # transform your data in a numpy array of floats
    y = np.array(y_, dtype=float)  # so the curve_fit can work
    MAE_Polynomial = 99
    parameter_Polynomial = []
    R2_Poly = 0
    RMSE_Poly = 0
    deg_Polynomial = 0
    for deg_i in range(1, 6):
        parameter, R2, RMSE, AE_list_cv, MAE_cv, AE_list_nocv, MAE_nocv = Polynomial_curve(x, y, deg_i)
        # print(MAE)
        np.savetxt(os.path.join(os.path.abspath(os.path.dirname(excel_path)), 'AE_list_deg%s.txt' % str(deg_i)),
                   AE_list_nocv)
        if MAE_Polynomial > MAE_cv:
            MAE_Polynomial = MAE_cv
            parameter_Polynomial = parameter
            R2_Poly = R2
            RMSE_Poly = RMSE
            deg_Polynomial = deg_i
    # 输出最优曲线交叉验证后的MAE和曲线参数
    # print(MAE_Polynomial, parameter_Polynomial)
    # np.savetxt(os.path.join(os.path.abspath(os.path.dirname(excel_path)), 'parameter_final_%s.txt' % str(deg_Polynomial)), parameter_Polynomial)

    popt_exp, R2_exp, RMSE_exp, AE_list_cv_exp, MAE_cv_exp, AE_list_nocv_exp, MAE_nocv_exp = exp_curve(x, y, 1e-5)
    np.savetxt(os.path.join(os.path.abspath(os.path.dirname(excel_path)), 'AE_list_exp.txt'), AE_list_nocv_exp)
    popt_log, R2_log, RMSE_log, AE_list_cv_log, MAE_cv_log, AE_list_nocv_log, MAE_nocv_log = log_curve(x, y)
    np.savetxt(os.path.join(os.path.abspath(os.path.dirname(excel_path)), 'AE_list_log.txt'), AE_list_nocv_log)
    popt_pow, R2_pow, RMSE_pow, AE_list_cv_pow, MAE_cv_pow, AE_list_nocv_pow, MAE_nocv_pow = pow_curve(x, y)
    np.savetxt(os.path.join(os.path.abspath(os.path.dirname(excel_path)), 'AE_list_pow.txt'), AE_list_nocv_pow)

    # MAE_min = 0
    # parameter_optimal = {}
    if MAE_Polynomial > MAE_cv_exp:
        MAE_min = MAE_cv_exp
        parameter_optimal = {'Func_type': 'exp', 'Parameters': popt_exp, 'R2': R2_exp, 'RMSE_exp': RMSE_exp,
                             'MAE_min': MAE_min}
        # print(MAE_min, parameter_optimal)
    else:
        MAE_min = MAE_Polynomial
        parameter_optimal = {'Func_type': 'ploy', 'Parameters': parameter_Polynomial, 'R2': R2_Poly,
                             'RMSE_Poly': RMSE_Poly, 'MAE_min': MAE_min}
        # print(MAE_min, parameter_optimal)
    if MAE_min > MAE_cv_log:
        MAE_min = MAE_cv_log
        parameter_optimal = {'Func_type': 'log', 'Parameters': popt_log, 'R2': R2_log, 'RMSE_log': RMSE_log,
                             'MAE_min': MAE_min}
        # print(MAE_min, parameter_optimal)
    if MAE_min > MAE_cv_pow:
        MAE_min = MAE_cv_pow
        parameter_optimal = {'Func_type': 'pow', 'Parameters': popt_pow, 'R2': R2_pow, 'RMSE_pow': RMSE_pow,
                             'MAE_min': MAE_min}
        # print(MAE_min, parameter_optimal)
    return parameter_optimal


# 计算水位
def Calcu_level(area, func_parameter):
    if func_parameter['Func_type'] == 'ploy':
        p = np.poly1d(func_parameter['Parameters'])
        level = p(area)
        # print(level)
    elif func_parameter['Func_type'] == 'exp':
        level = exp_func(area, func_parameter['Parameters'][0], func_parameter['Parameters'][1])
    elif func_parameter['Func_type'] == 'log':
        level = log_func(area, func_parameter['Parameters'][0], func_parameter['Parameters'][1])
    else:  # power
        level = pow_func(area, func_parameter['Parameters'][0], func_parameter['Parameters'][1])
    return np.array(level)


# 计算蓄变量
def Calcu_LWSC(area, level):
    V_list = []
    for numb_i in range(len(level) - 1):
        V = ((level[numb_i + 1] - level[numb_i]) / 3) * (
                area[numb_i] + area[numb_i + 1] + np.sqrt(area[numb_i] * area[numb_i + 1])) * 1000000
        V_list.append(V)
    # print(len(V_list))
    V_change = np.insert(np.cumsum(V_list), 0, 0)
    return V_change


# 绘制曲线
def plot_curve(excel_path, fig, func_parameter, index, name):
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    area_level = pd.read_excel(excel_path, header=None, names=['area', 'level'])
    # print(area_level.head())
    area_level = area_level.sort_values(by="area")
    # print(area_level.head())
    x_ = area_level.iloc[:, 0]
    y_ = area_level.iloc[:, 1]
    x = np.array(x_, dtype=float)  # transform your data in a numpy array of floats
    y = np.array(y_, dtype=float)  # so the curve_fit can work
    x_p = np.linspace(x.min(), x.max(), 100)
    level = Calcu_level(x_p, func_parameter)
    ax = fig.add_subplot(4, 4, index + 1)
    # print(x.shape)
    # print(level.shape)
    plt.plot(x_p, level, color='r', label=str(name))
    plt.scatter(x, y, s=2)
    plt.yticks(fontsize=8)
    # print(i)
    # if index > 11:
    #     plt.xticks(fontsize=8)
    # else:
    #     plt.xticks([])
    plt.legend()
    # plt.title(label=str(name))
    y_major_locator = MultipleLocator((np.nanmax(level) - np.nanmin(level)) / 5)
    ax.yaxis.set_major_locator(y_major_locator)
    x_major_locator = MultipleLocator((np.nanmax(x_p) - np.nanmin(x_p)) / 4)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    fig.text(0.5, 0.04, 'Lake area (km$^2$)', ha='center', fontsize=14)
    fig.text(0.065, 0.5, 'Lake water level (m)', va='center', rotation='vertical', fontsize=14)

    # plt.xlabel('Date (Year)', fontsize=18)
    # plt.ylabel('LWS change (km$^3$)', fontsize=18)


if __name__ == '__main__':
    all_lake_water_storage_change = []
    Data_pair_path = '湖泊面积水位数据对'  # 用于构建曲线
    Data_pair_name_list = os.listdir(Data_pair_path)
    # Area_time_series_path = 'J:/GSW_monthly'
    Area_time_series_path = 'GSW'
    # Area_time_series_name_list = os.listdir(Area_time_series_path)
    fig = plt.figure(figsize=(20, 10))
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    all_lake_level_range_1987_2020 = []
    er_lake_area_all = pd.read_excel('GSW/GSW_gahai/er_area_zl.xlsx',
                                     sheet_name='er_lake_area_all')
    er_lake_level_list = []
    LWS_list = []
    date_list = []
    for data_pair_name in Data_pair_name_list:
        print(data_pair_name)
        func_parameter = Optimal_curve(os.path.join(Data_pair_path, data_pair_name, data_pair_name + '.xlsx'))
        # 或者用自己的方程，按照一下格式定义即可
        # func_parameter = {'Func_type': 'pow', 'Parameters': popt_pow, 'R2': R2_pow, 'RMSE_pow': RMSE_pow,
        #                              'MAE_min': MAE_min}
        print(func_parameter)
        f = open("new_func_parameter.txt", "a")  # 利用追加模式,参数从w替换为a即可
        f.write(str(func_parameter) + '\n')
        f.close()
        Area_time_series = pd.read_excel(
            os.path.join(Area_time_series_path, 'GSW_' + data_pair_name, 'lake_area.xlsx'),
            sheet_name='Sheet1', header=None)
        # 获取湖泊面积
        Area_time_series = Area_time_series.interpolate()
        # print(Area_time_series)
        Area = Area_time_series[~np.isnan(Area_time_series.iloc[:, 1])].iloc[:, 1]
        Area = np.array(Area)
        # print(Area)
        date = Area_time_series[~np.isnan(Area_time_series.iloc[:, 1])].iloc[:, 0]
        # print(date)
        # 计算湖泊水位
        lake_level = Calcu_level(Area, func_parameter)
        # print(Data_pair_name_list.index(data_pair_name) + 1)
        er_lake_area = er_lake_area_all.iloc[:, Data_pair_name_list.index(data_pair_name) + 1]
        # print(Area, er_lake_area)
        area_wai = Area + er_lake_area
        area_nei = Area - er_lake_area
        lake_level_wai = Calcu_level(area_wai, func_parameter)
        lake_level_nei = Calcu_level(area_nei, func_parameter)
        er_lake_level = (lake_level_wai - lake_level_nei) / 2
        # print(lake_level_wai, lake_level_nei)
        er_lake_level_list.append(er_lake_level)

        range_1987_2020 = lake_level[-1] - lake_level[0]
        all_lake_level_range_1987_2020.append(range_1987_2020)
        plot_curve(os.path.join(Data_pair_path, data_pair_name, data_pair_name + '.xlsx'), fig, func_parameter,
                   Data_pair_name_list.index(data_pair_name), data_pair_name)
        # fig.text(0.5, 0.04, 'Date (year)', ha='center', fontsize=14)
        # fig.text(0.065, 0.5, 'Lake water level (m)', va='center', rotation='vertical', fontsize=14)

        df = pd.DataFrame(lake_level, columns=['lake_level'])
        df.insert(0, 'date', np.array(date))
        df.to_excel(os.path.join(os.path.abspath(
            os.path.dirname(os.path.join(Area_time_series_path, 'GSW_' + data_pair_name, 'lake_area.xlsx'))),
            'lake_level.xlsx'),
            index=None)
        # 计算湖泊蓄水量
        lake_water_storage_change = Calcu_LWSC(Area, lake_level)
        LWS_list.append(lake_water_storage_change)
        # lake_water_storage_change_date = np.concatenate((np.array(date).reshape(-1, 1), lake_water_storage_change.reshape(-1, 1)), axis=1)
        # print(lake_water_storage_change_date)
        # print(np.array(date).reshape(-1, 1).shape, lake_water_storage_change.reshape(-1, 1).shape)
        # all_lake_water_storage_change.append(lake_water_storage_change_date)
        df1 = pd.DataFrame(lake_water_storage_change, columns=['LWS_change'])
        df1.insert(0, 'date', np.array(date))
        df1.to_excel(
            os.path.join(Area_time_series_path, 'GSW_' + data_pair_name, 'lake_water_storage_change.xlsx'),
            index=None)
        date_list.append(np.array(date))

        # print(lake_level)
        # print(V_change)
    # df2 = pd.DataFrame(np.array(all_lake_water_storage_change).reshape(-1, 15))
    # df2.to_excel('all_lake_water_storage_change.xlsx', header=None, index=None)
    print(np.array(er_lake_level_list).transpose(1, 0).shape)
    # plt.show()
    # fig.savefig('Qaidam/Qaidam_curve_four.png', dpi=300, bbox_inches="tight")
    np.savetxt('lake_level_range_1987_2020.txt', all_lake_level_range_1987_2020)
    df2 = pd.DataFrame(np.array(er_lake_level_list).transpose(1, 0))

    df2.insert(0, 'date', np.array(date))
    df2.to_excel('error_lake_level.xlsx', index=None)

    df3 = pd.DataFrame(np.array(LWS_list).transpose(1, 0))
    df3.to_excel('LWS_list.xlsx', index=None)
