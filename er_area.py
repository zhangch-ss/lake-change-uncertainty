from read_write_img import read_img, write_img
import numpy as np
import os
import pandas as pd
import cv2
from skimage.measure import label
import matplotlib.pyplot as plt
from numpy import ma


def max_connected_domain(mask_sel):
    labeled_img, num = label(mask_sel, background=0, return_num=True)
    # plt.imshow(labeled_img)
    # plt.show()
    # print(labeled_img)
    # plt.figure(), plt.imshow(labeled_img, 'gray')
    max_label = 0
    max_num = 0
    for i in range(1, num + 1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    return lcc


def uncertainties(img_path):
    im_proj, im_geotrans, im_data = read_img(img_path)
    new_data = np.copy(im_data)
    # print(im_data)

    ret, thresh = cv2.threshold(im_data, 1, 255, 0)
    lcc = max_connected_domain(thresh)
    mask_x1 = ma.array(im_data, mask=~lcc)

    water_index = np.where(mask_x1 == 2)
    # print(water_index)
    kernel = [[0, 1, 0],
              [1, 1, 1],
              [0, 1, 0]]
    num = 0
    for i in range(len(water_index[0])):
        if im_data[water_index[0][i] - 1:water_index[0][i] + 2,
           water_index[1][i] - 1:water_index[1][i] + 2].shape == (3, 3):
            if np.nansum(kernel * im_data[water_index[0][i] - 1:water_index[0][i] + 2,
                                  water_index[1][i] - 1:water_index[1][i] + 2]) != 10:
                num += 1
                # new_data[water_index[0][i], water_index[1][i]] = 255
                # write_img(os.path.join(), im_proj, im_geotrans, new_data)
    uncert_num = num / 2
    uncert_area = uncert_num * 30 * 30 / 1000000
        # print(uncert_area)
    # return uncert_area, new_data
    return uncert_area


if __name__ == '__main__':
    dri = 'GSW'
    # print(dri)
    dri_list = os.listdir(dri)
    # print(dri_list)
    # GSW_dri = 'J:/GSW_ALL/GSW_Gahai/data'
    for d_ in dri_list:
        print(os.path.join(dri, d_, 'data'))
        GSW_list = os.listdir(os.path.join(dri, d_, 'data'))
        uncert_area_list = []
        date_list = []
        for GSW in GSW_list:
            print(d_, GSW)
            un_area = uncertainties(os.path.join(dri, d_, 'data', GSW))
            # un_area, new_data = uncertainties(os.path.join(dri, d_, 'data', GSW))
            # un_tif_save = os.path.join(dri, d_, 'uncertainties')
            # if not os.path.exists(os.path.join(un_tif_save)):
            #     os.makedirs(os.path.join(un_tif_save))
            # cv2.imwrite(os.path.join(un_tif_save, GSW), new_data)
            uncert_area_list.append(un_area)
            date_list.append(GSW.rstrip('.tif'))
        all_data = np.column_stack((date_list, uncert_area_list))
        df = pd.DataFrame(all_data)
        # print(os.path.join(dri, d_, 'uncertainties.xlsx'))
        writer = pd.ExcelWriter(os.path.join(dri, d_, 'er_area.xlsx'))
        df.columns = ['Date', 'Uncert_area']
        df.to_excel(writer, header=True, index=False)
        writer.save()
        writer.close()
