import pandas as pd
from pyproj import Transformer
from pyproj import CRS
import json
import random


crs_CGCS2000 = CRS.from_wkt(
    'PROJCS["CGCS_2000_3_Degree_GK_CM_117E",GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",117.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')  # degree
crs_WGS84 = CRS.from_epsg(4326)
from_crs = crs_CGCS2000
to_crs = crs_WGS84
transformer = Transformer.from_crs(from_crs, to_crs)

color_dic = {
    "地表": [0, 255, 255, 255],
    "杂填土": [191, 255, 0, 1.0],
    "黏土": [255, 191, 0, 1.0],
    "泥灰岩": [0, 255, 255, 1.0],
    "石灰岩": [137, 137, 137, 1.0],
    "页岩": [255, 127, 191, 1.0],
    "溶洞": [255, 0, 0, 0.7],
    "溶洞无充填": [255, 0, 0, 0.7],
    "溶洞半充填": [255, 127, 0, 0.7],
    "溶洞全充填": [0, 255, 0, 0.7],
}


def generate_random_color():
    # 生成随机的RGB值
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    a = 1.0
    # 返回RGB值作为元组
    return r, g, b, a


def export_well_json(data, json_path):
    pre = ''
    jsondata = []
    x = data.loc[0, 'X']
    y = data.loc[0, 'Y']
    latitude, longtitude = transformer.transform(x, y)
    well = {'id': 0, 'name': data.loc[0, '孔号'], 'positions': [longtitude, latitude, data.loc[0, '孔口标高']], 'formations': []}
    pre = well['name']
    for i in range(data.shape[0]):
        if data.loc[i, '孔号'] != pre or i == data.shape[0] - 1:
            jsondata.append(well)
            print(well['positions'])
            x = data.loc[i, 'X']
            y = data.loc[i, 'Y']
            latitude, longtitude = transformer.transform(x, y)
            well = {'id': len(jsondata), 'name': data.loc[i, '孔号'], 'positions': [longtitude, latitude, data.loc[i, '孔口标高']], 'formations': []}
            pre = well['name']
        _type = data.loc[i, '地层类型']
        height = data.loc[i, '分层厚度']
        bottom = data.loc[i, '层底标高']
        well['formations'].append({
            'formation': _type,
            'position': [longtitude, latitude, round(bottom + height / 2, 2)],
            'length': height,
            'color': color_dic[_type]
        })
    print(len(jsondata))
    return jsondata


data_path = './源数据处理/钻孔/'
filename = 'Wells.csv'
data = pd.read_csv(data_path + filename)
json_path = './JSON/DrillingHole'
jsondata = export_well_json(data, json_path)
outname = 'Wells'
print(jsondata)
js_json_path = 'E:/SuiDaoProject/jscode/Vue3-Vite-Cesium_V1.8.8/public/js/json/DrillingHole'
with open(json_path + '/' + outname + '.json', 'w') as json_file:
    json.dump(jsondata, json_file, indent=2)
with open(js_json_path + '/' + outname + '.json', 'w') as json_file:
    json.dump(jsondata, json_file, indent=2)
