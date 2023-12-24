import sympy as sp
import numpy as np
import pandas as pd


def er_LWS_change(a1_, a2_, e1_, e2_, er_a1,  er_a2, er_e1, er_e2):
    # 构架湖泊蓄变量方程
    a1, a2, e1, e2 = sp.symbols('a1 a2 e1 e2')
    lwsc = (e2 - e1) * (a1 + a2 + sp.sqrt(a1 * a2)) / 3
    # 构建每个参数的偏导方程
    lwsc_a1_f = sp.diff(lwsc, a1)
    lwsc_a2_f = sp.diff(lwsc, a2)
    lwsc_e1_f = sp.diff(lwsc, e1)
    lwsc_e2_f = sp.diff(lwsc, e2)

    # print(lwsc_a1_f)
    # print(lwsc_a2_f)
    # print(lwsc_e1_f)
    # print(lwsc_e2_f)
    # 湖泊面积和水位数值
    # 对每个偏导方程赋值
    lwsc_a1_v = float(lwsc_a1_f.evalf(subs={a1: a1_, a2: a2_, e1: e1_, e2: e2_}))
    print(lwsc_a1_v)
    lwsc_a2_v = float(lwsc_a2_f.evalf(subs={a1: a1_, a2: a2_, e1: e1_, e2: e2_}))
    lwsc_e1_v = float(lwsc_e1_f.evalf(subs={a1: a1_, a2: a2_, e1: e1_, e2: e2_}))
    lwsc_e2_v = float(lwsc_e2_f.evalf(subs={a1: a1_, a2: a2_, e1: e1_, e2: e2_}))

    # print(lwsc_a1_v, lwsc_a2_v, lwsc_e1_v, lwsc_e2_v)

    # 构造LWS误差方程
    # 标准误差传递
    er_lws = sp.sqrt(pow(er_e1, 2) * pow(lwsc_e1_v, 2) + pow(er_e2, 2) * pow(lwsc_e2_v, 2) +
                     pow(er_a1, 2) * pow(lwsc_a1_v, 2) + pow(er_a2, 2) * pow(lwsc_a2_v, 2))

    # er_lws = er_a1*abs(lwsc_a1_v) + er_a2*abs(lwsc_a2_v) + er_e1*abs(lwsc_e1_v) + er_e2*abs(lwsc_e2_v)
    return er_lws
    # print(er_lws)


lake_area = pd.read_excel('GSW/GSW_gahai/lake_area_zl.xlsx', sheet_name='lake_area_all')
lake_level = pd.read_excel('GSW/GSW_gahai/lake_level.xlsx')
er_area = pd.read_excel('GSW/GSW_gahai/lake_area_zl.xlsx', sheet_name='er_lake_area_all')
er_level = pd.read_excel('error_lake_level.xlsx')
print(er_area)
# def er_LWS_change(a1_, a2_, e1_, e2_, er_a1,  er_a2, er_e1, er_e2):
lake_area = lake_area.iloc[:, 1:]*1000000
lake_area[lake_area == 0] = 0.000001
lake_level = lake_level.iloc[:, 1:]
er_area = er_area.iloc[:, 1:]*1000000
er_area[er_area == 0] = 0.000001
er_level = er_level.iloc[:, 1:]
er_level[er_level == 0] = 0.000001
er_LWSC_list_list = []
for i in range(len(lake_area.iloc[0, :])):
    er_LWSC_list = [0]
    for j in range(len(lake_area) - 1):
        # print(er_level.iloc[j + 1, i])
        print(i, j)
        er_LWSC = er_LWS_change(lake_area.iloc[j, i],
                                lake_area.iloc[j + 1, i],
                                lake_level.iloc[j, i],
                                lake_level.iloc[j + 1, i],
                                er_area.iloc[j, i],
                                er_area.iloc[j + 1, i],

                                er_level.iloc[j, i],
                                er_level.iloc[j + 1, i])
        er_LWSC_list.append(er_LWSC)
    er_LWSC_list_list.append(er_LWSC_list)
er_LWSC_arr = np.array(er_LWSC_list_list).transpose(1, 0)/1000000000
df = pd.DataFrame(er_LWSC_arr.astype('float'))
df.to_excel('error_LWS_change.xlsx', index=None)
print(er_LWSC_arr.shape)
