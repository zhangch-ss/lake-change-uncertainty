from osgeo import gdal


def read_img(filename):
    data = gdal.Open(filename)  # 打开文件
    im_width = data.RasterXSize  # 读取图像行数
    im_height = data.RasterYSize  # 读取图像列数

    im_geotrans = data.GetGeoTransform()
    # 仿射矩阵，左上角像素的大地坐标和像素分辨率。
    # 共有六个参数，分表代表左上角x坐标；东西方向上图像的分辨率；如果北边朝上，地图的旋转角度，0表示图像的行与x轴平行；左上角y坐标；
    # 如果北边朝上，地图的旋转角度，0表示图像的列与y轴平行；南北方向上地图的分辨率。
    im_proj = data.GetProjection()  # 地图投影信息
    im_data = data.ReadAsArray(0, 0, im_width, im_height)  # 此处读取整张图像
    # ReadAsArray(<xoff>, <yoff>, <xsize>, <ysize>)
    # 读出从(xoff,yoff)开始，大小为(xsize,ysize)的矩阵。
    del data  # 释放内存，如果不释放，在arcgis，envi中打开该图像时会显示文件被占用

    return im_proj, im_geotrans, im_data


def write_img(filename, im_proj, im_geotrans, im_data):
    # filename-创建的新影像
    # im_geotrans,im_proj该影像的参数，im_data，被写的影像
    # 写文件，以写成tiff为例
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64
    # 判断栅格数据的类型
    # print(im_data.dtype.name)
    # if 'int8' or 'uint8' in im_data.dtype.name:
    #     print('11111111111')

    if 'int8' in im_data.dtype.name or 'uint8' in im_data.dtype.name:
        # print('1111111111111111')
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        if im_data.min() < 0:
            datatype = gdal.GDT_Int16
        else:
            datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # print(datatype)
    if len(im_data.shape) == 3:  # len(im_data.shape)表示矩阵的维数
        im_bands, im_height, im_width = im_data.shape  # （维数，行数，列数）
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape  # 一维矩阵

    #         创建文件
    driver = gdal.GetDriverByName('GTiff')  # 数据类型必须有，因为要计算需要多大内存空间
    data = driver.Create(filename, im_width, im_height, im_bands, datatype)
    data.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    data.SetProjection(im_proj)  # 写入投影
    if im_bands == 1:
        data.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            data.GetRasterBand(i + 1).WriteArray(im_data[i])
    del data