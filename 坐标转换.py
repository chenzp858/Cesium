import math
import os

import numpy as np
import pandas as pd
import scipy.linalg as linalg
from ReadData import *
from pyproj import Transformer
from pyproj import CRS


# 计算真实坐标数据
def CalRealCoordinates(surfacer, wellr, wellzs):  # 分别输入地层界面点，相对地层坐标点，真实地层坐标点
    res = []
    l0 = (wellr[0][3] - wellr[0][5]) / (wellzs[0][2] - wellzs[0][3])  # 参考勘探点长度
    z0 = wellr[0][3] - (wellzs[0][2] * l0)  # Z轴参考0点
    for i in range(len(surfacer)):  # 遍历地层点
        temp = []
        for j in range(len(wellr)):  # 每口井要遍历到刚好大于当前地层点的位置(起点)
            if wellr[-1][2] < surfacer[i][2]:  # 点坐标超出最右侧钻孔时
                j -= 1
                temp.append(surfacer[i][0])
                temp.append(surfacer[i][1])
                x = ((surfacer[i][2] - wellr[j - 1][2]) * (wellzs[j][0] - wellzs[j - 1][0]) / (
                        wellr[j][2] - wellr[j - 1][2])) + wellzs[j - 1][0]
                y = ((surfacer[i][2] - wellr[j - 1][2]) * (wellzs[j][1] - wellzs[j - 1][1]) / (
                        wellr[j][2] - wellr[j - 1][2])) + wellzs[j - 1][1]
                z = (surfacer[i][3] - z0) / l0
                temp.append(x)
                temp.append(y)
                temp.append(z)
                res.append(temp.copy())
                temp.clear()
                break
            if wellr[j][2] > surfacer[i][2]:  # 正常点坐标处理
                temp.append(surfacer[i][0])
                temp.append(surfacer[i][1])
                x = ((surfacer[i][2] - wellr[j - 1][2]) * (wellzs[j][0] - wellzs[j - 1][0]) / (
                        wellr[j][2] - wellr[j - 1][2])) + wellzs[j - 1][0]
                y = ((surfacer[i][2] - wellr[j - 1][2]) * (wellzs[j][1] - wellzs[j - 1][1]) / (
                        wellr[j][2] - wellr[j - 1][2])) + wellzs[j - 1][1]
                z = (surfacer[i][3] - z0) / l0
                temp.append(x)
                temp.append(y)
                temp.append(z)
                res.append(temp.copy())
                temp.clear()
                break
    return res


# 坐标旋转
def Rotate_Point(point, axis, r):  # 点坐标（x,y,z），旋转轴，旋转角度（点绕轴转）
    if axis.lower() == "x":
        ax = [1, 0, 0]
    elif axis.lower() == "y":
        ax = [0, 1, 0]
    elif axis.lower() == "z":
        ax = [0, 0, 1]
    else:
        raise ValueError("Only x, y or z are supported.")
    radian = r * math.pi / 180  # 角度转为弧度
    rot_matrix = linalg.expm(np.cross(np.eye(3), ax / linalg.norm(ax) * radian))  # 计算旋转矩阵
    new_point = np.dot(rot_matrix, point)  # 计算新坐标
    return new_point


# 旋转地层模型
def Rotate_Formations(data_path, filename, startindex, endindex, axis, r):
    """
    要求数据表头为["序号", "图层", "X", "Y", "Z"]
    """
    data = pd.read_csv(data_path + filename, encoding="utf-8-sig")
    points = np.array(data)
    new_points = []
    for point in points:
        temp = []
        newpoint = Rotate_Point(point[startindex:endindex + 1], axis, r)
        temp.extend(newpoint)
        new_points.append(temp)
    new_points1 = pd.DataFrame(new_points)
    res = pd.concat([data, new_points1], axis=1)
    res.to_csv(data_path + f"Rotate_{filename}", encoding="utf-8-sig")


# 反向旋转地层模型
def Back_Rotate_Formations(data_path, filename, axis, r):
    csv_data = read_TS(data_path, filename)
    for key in csv_data.keys():
        data = csv_data[key][0]  # 提取每个地层模型的值数据
        print(data)
        points = np.array(data.iloc[:, 1:4])
        new_points = []
        for point in points:
            new_point = Rotate_Point(point, axis, r)
            new_points.append(new_point)
        new_points1 = pd.DataFrame(new_points, columns=["X", "Y", "Z"])
        res = pd.concat([data.iloc[:, 0], new_points1, data.iloc[:, 4:]], axis=1).apply(pd.to_numeric)
        csv_data[key] = res
    return csv_data


# CGCS2000 xy坐标转为WGS84经纬网
def cgcs_to_wgs(arr):
    crs_CGCS2000 = CRS.from_wkt(
        'PROJCS["CGCS_2000_3_Degree_GK_CM_117E",GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",117.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')  # degree
    crs_WGS84 = CRS.from_epsg(4326)
    from_crs = crs_CGCS2000
    to_crs = crs_WGS84
    transformer = Transformer.from_crs(from_crs, to_crs)
    new_x, new_y = transformer.transform(arr[0], arr[1])  # new_x,new_y即为转换后的坐标，也可以分别使用数组
    return [new_x, new_y]


# 隧道相对坐标向绝对坐标转换
def to_real_cor(linedata, distance):
    for key in linedata.keys():
        points = linedata[key][0]
        segs = linedata[key][2]
        _sum = 0
        for index in range(segs.shape[0]):
            start, end = segs.loc[index, ['start', 'end']]
            sx, sy, sz = points.loc[start, ['X', 'Y', 'Z']]
            ex, ey, ez = points.loc[end, ['X', 'Y', 'Z']]
            se = ((ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2) ** 0.5
            if _sum + se >= distance:
                dis = distance - _sum
                r = get_unit([sx,sy,sz],[ex,ey,ez])
                x = (dis / se) * (ex - sx) + sx
                y = (dis / se) * (ey - sy) + sy
                z = (dis / se) * (ez - sz) + sz
                return x, y, z, r
            else:
                _sum += se
        return _sum, 0, 0, 0


def get_unit(start, end):
    s = np.array(start)
    e = np.array(end)
    v = e - s
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError('向量模长为0!')
    unit = v / norm
    return list(unit)


def get_angle(x, y):
    r = np.degrees(np.arctan(y / x))
    return r


if __name__ == "__main__":
    pass
    """
    data_path = "读取数据/"
    out_path = "数据处理/"
    data = pd.read_csv(data_path + "lines预处理（准备坐标转换）.csv", encoding="utf-8-sig")
    ellipsedata = pd.read_csv(data_path + "ellipses.csv", encoding="utf-8-sig")
    # 声明北线钻井相对坐标数据集
    nwellr = np.array(data.iloc[0:70, 0:8])
    # 声明南线钻井相对坐标数据集
    swellr = np.array(data.iloc[70:139, 0:8])
    # 声明北线钻井真实坐标数据集
    nwellzs = np.array(data.iloc[0:70, 9:14])
    # 声明南线钻井真实坐标数据集
    swellzs = np.array(data.iloc[70:139, 9:14])
    # 声明北线溶洞中心点相对坐标数据集
    nellipses = np.array(ellipsedata.iloc[:365, :])
    # 声明南线溶洞相对坐标数据集
    sellipses = np.array(ellipsedata.iloc[365:804, :])
    # 声明北线溶洞长轴矢量点相对坐标集
    nellipseslong = np.array(ellipsedata.iloc[:365, :4])
    # 声明北线溶洞短轴矢量点相对坐标集
    nellipsesshort = np.array(ellipsedata.iloc[:365, :4])
    for i in range(len(nellipses)):
        nellipseslong[i][2] += nellipses[i][5]
        nellipseslong[i][3] += nellipses[i][6]
        nellipsesshort[i][2] += nellipses[i][8]
        nellipsesshort[i][3] += nellipses[i][9]
    # 声明南线溶洞长轴矢量点相对坐标集
    sellipseslong = np.array(ellipsedata.iloc[365:804, :4])
    # 声明南线溶洞短轴矢量点相对坐标集
    sellipsesshort = np.array(ellipsedata.iloc[365:804, :4])
    for i in range(len(sellipses)):
        sellipseslong[i][2] += sellipses[i][5]
        sellipseslong[i][3] += sellipses[i][6]
        sellipsesshort[i][2] += sellipses[i][8]
        sellipsesshort[i][3] += sellipses[i][9]

    columns = ["序号", "图层", "圆心X", "圆心Y", "圆心Z"]
    columnslong = ["序号", "图层", "长轴X", "长轴Y", "长轴Z"]
    columnsshort = ["序号", "图层", "短轴X", "短轴Y", "短轴Z"]

    data_n = pd.DataFrame(CalRealCoordinates(nellipses, nwellr, nwellzs), columns=columns)
    long_n = pd.DataFrame(CalRealCoordinates(nellipseslong, nwellr, nwellzs), columns=columnslong)
    short_n = pd.DataFrame(CalRealCoordinates(nellipsesshort, nwellr, nwellzs), columns=columnsshort)
    data_n = pd.concat([data_n, long_n.iloc[:, 2:], short_n.iloc[:, 2:]], axis=1)

    data_s = pd.DataFrame(CalRealCoordinates(sellipses, swellr, swellzs), columns=columns)
    long_s = pd.DataFrame(CalRealCoordinates(sellipseslong, swellr, swellzs), columns=columnslong)
    short_s = pd.DataFrame(CalRealCoordinates(sellipsesshort, swellr, swellzs), columns=columnsshort)
    data_s = pd.concat([data_s, long_s.iloc[:, 2:], short_s.iloc[:, 2:]], axis=1)

    # data_n.to_csv(out_path + "北线溶洞真实坐标点.csv", encoding="utf-8-sig")
    # data_s.to_csv(out_path + "南线溶洞真实坐标点.csv", encoding="utf-8-sig")


    # 声明北线地层线起点相对坐标数据集
    nsurfacer1 = np.array(data.iloc[139:188, :4])
    nsurfacer2 = np.array(data.iloc[139:188, [0, 1, 4, 5]])
    # 声明南线地层线起点相对坐标数据集
    ssurfacer1 = np.array(data.iloc[188:238, :8])
    ssurfacer2 = np.array(data.iloc[188:238, :8])

    columns1 = ["序号", "图层", "起点X", "起点Y", "起点Z"]
    columns2 = ["序号", "图层", "终点X", "终点Y", "终点Z"]

    line_n1 = pd.DataFrame(CalRealCoordinates(nsurfacer1, nwellr, nwellzs), columns=columns1)
    line_n2 = pd.DataFrame(CalRealCoordinates(nsurfacer2, nwellr, nwellzs), columns=columns2)
    line_n = pd.concat([line_n1, line_n2.iloc[:, 2:]], axis=1)
    line_s1 = pd.DataFrame(CalRealCoordinates(ssurfacer1, swellr, swellzs), columns=columns1)
    line_s2 = pd.DataFrame(CalRealCoordinates(ssurfacer1, swellr, swellzs), columns=columns2)
    line_s = pd.concat([line_s1, line_s2.iloc[:, 2:]], axis=1)

    # line_n.to_csv(out_path + "北线地层点XY坐标.csv", encoding="utf-8-sig")
    # line_s.to_csv(out_path + "南线地层点XY坐标.csv", encoding="utf-8-sig")
    """

    # data_path = "古生界坐标旋转/"
    # # filename = "formations.csv"
    # # Rotate_Formations(data_path, filename, 4, 6, "y",-90)
    # rfilename = "surfaces24_9_02.ts"
    # wfilename = "Rotate_surfaces24_9_02.ts"
    #
    # def write_TS(data_path, rfilename, wfilename):  # 将转换后的点云数据坐标及原有数据重新写入新的文件
    #     with open(data_path + rfilename, "r", encoding="utf-8") as f:
    #         with open(data_path + wfilename, "a", encoding="utf-8") as w:
    #             w.seek(0)  # 定位至文件开头
    #             w.truncate()  # 清空文件
    #             for line in f:
    #                 if not line.strip():
    #                     continue
    #                 line_type, *values = line.split()
    #                 if line_type == "VRTX" or line_type == "PVRTX":
    #                     values[1], values[3] = str(float(values[3]) * -1), values[1]
    #                     w.write(f"{line_type} {' '.join(values)}\n")
    #                     continue
    #                 w.write(line)  # 其他类型的数据直接写入
    #         w.close()
    #     f.close()
    #
    #
    # write_TS(data_path, rfilename, wfilename)

    # ts_path = "./ts/"
    # files = os.listdir(ts_path)
    # for file in files:
    #     filepath = ts_path + file
    #     csv_data = Back_Rotate_Formations('./ts/', file, 'y', -90)
    #     print(csv_data)
    #     write_TS('./ts/', file, 'Rotate_' + file, csv_data)

    # # 实体模型基准点坐标转换
    # data = pd.read_csv('./读取数据/实体模型基准点坐标5.14.csv')
    # print(data)
    # for i in range(data.shape[0]):
    #     temp = list(data.loc[i, ['X', 'Y']])
    #     wgs = cgcs_to_wgs(temp)
    #     data.loc[i, 'lat'] = wgs[0]
    #     data.loc[i, 'lon'] = wgs[1]
    # data.to_csv('./读取数据/实体模型基准点坐标_含经纬度5.14.csv')
    # print(data)

    # 隧道相对坐标向绝对坐标转换
    ts_path = './源数据处理/隧道/'
    linename, long = 'LeftLine.ts', 1770.0563389045694  # 总长度1770.0563389045694
    # linename, long = 'RightLine.ts', 1749.8467681726147  # 总长度1749.8467681726147

    data = read_TS(ts_path, linename)
    i = 0
    res = []
    while i < long:
        x, y, z, unit_v = to_real_cor(data, i)
        lat, lon = cgcs_to_wgs([x, y])
        res.append([i,x,y,z,unit_v,lat,lon])
        print(i,x,y,z,unit_v,lat,lon)
        i += 1
    df = pd.DataFrame(res, columns=['dis', 'X', 'Y', 'Z', 'unit_v','lat','lon'])
    # df.to_csv('./地球物理属性数据/左线米级传感器坐标数据.csv', encoding='utf-8-sig')

    # results = []
    # i = 0
    # while i < 1770.0563389045694:
    #     x, y, z, r = to_real_cor(data, i)
    #     for j in range(13):
    #         results.append([i, x,y-6,z-3+j, r])
    #     for j in range(13):
    #         results.append([i, x,y  ,z-3+j, r])
    #     for j in range(13):
    #         results.append([i, x,y+6,z-3+j, r])
    #     i += 1
    #     print(i)
    # df = pd.DataFrame(results, columns=['id','X','Y','Z','r'])
    # df.to_csv('./源数据处理/隧道/左线米级点坐标及旋转角.csv', encoding='utf-8-sig')
    # # df.to_csv('./源数据处理/隧道/右线米级点坐标及旋转角.csv', encoding='utf-8-sig')

    # filepath = './源数据处理/视电阻率/'
    # filename = '视电阻率_810.csv'
    # resistanceData = pd.read_csv(filepath + filename, encoding='utf-8-sig')
    # for i in range(resistanceData.shape[0]):
    #     x, y, z = to_real_cor(data, float(resistanceData.loc[i, '距离']))
    #     resistanceData.loc[i, 'X'] = x
    #     resistanceData.loc[i, 'Y'] = y
    #     resistanceData.loc[i, 'Z'] = z + resistanceData.loc[i, 'rZ']
    # print(resistanceData)
    # resistanceData.to_csv(filepath + filename, encoding='utf-8-sig')




    # leftlinepoints = pd.read_csv('./源数据处理/隧道/左线米级点坐标及旋转角.csv', encoding='utf-8-sig')
    # rightlinepoints = pd.read_csv('./源数据处理/隧道/右线米级点坐标及旋转角.csv', encoding='utf-8-sig')

    # # 为点集添加视电阻率数据
    # filepath = './源数据处理/视电阻率/'
    # filename = '视电阻率_810.csv'
    # resistanceData = pd.read_csv(filepath + filename, encoding='utf-8-sig')
    # for i in range(resistanceData.shape[0]):
    #     x = resistanceData.loc[i, 'X']
    #     y = resistanceData.loc[i, 'Y']
    #     z = resistanceData.loc[i, 'Z']
    #     linepoints.loc[(linepoints['X'] == x) & (linepoints['Y'] == y) & (linepoints['Z'] == z), 'V'] = resistanceData.loc[i, '模拟值']
    # linepoints.to_csv('./源数据处理/隧道/左线米级点坐标及旋转角.csv', encoding='utf-8-sig')

    # # 点集坐标转换
    # for i in range(leftlinepoints.shape[0]):
    #     x = leftlinepoints.loc[i, 'X']
    #     y = leftlinepoints.loc[i, 'Y']
    #     lat,lon = cgcs_to_wgs([x,y])
    #     leftlinepoints.loc[i, 'lat'] = lat
    #     leftlinepoints.loc[i, 'lon'] = lon
    #     print('left', i)
    # leftlinepoints.to_csv('./源数据处理/隧道/左线米级点坐标及旋转角.csv', encoding='utf-8-sig')
    #
    # for i in range(rightlinepoints.shape[0]):
    #     x = rightlinepoints.loc[i, 'X']
    #     y = rightlinepoints.loc[i, 'Y']
    #     lat,lon = cgcs_to_wgs([x,y])
    #     rightlinepoints.loc[i, 'lat'] = lat
    #     rightlinepoints.loc[i, 'lon'] = lon
    #     print('right', i)
    # rightlinepoints.to_csv('./源数据处理/隧道/右线米级点坐标及旋转角.csv', encoding='utf-8-sig')

