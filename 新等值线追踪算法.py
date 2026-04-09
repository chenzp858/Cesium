from ModelData import *
from 包围盒类 import *


class ContourPoints:
    def __init__(self, startid, v, trglmesh: TrglMesh, type='Z'):
        self.type = type
        self.v = v
        self.mesh = trglmesh
        # self.mesh.create_edge_data()
        # id是三角网中的edge编号，xyz是边上等值点的坐标
        self.edges = pd.DataFrame(columns=['id', 'X', 'Y', 'Z', 'point'])
        # 奇异点处理方法
        # print(self.type)
        for i in range(self.mesh.points.shape[0]):
            if self.type != 'Z':
                if self.mesh.points.loc[i, self.type] == v:
                    # self.mesh.edge_points.loc[i, self.type] += 0.0000001
                    self.mesh.points.loc[i, self.type] += 0.0000001
            else:
                if self.mesh.points.loc[i, 'Z'] == v:
                    # self.mesh.edge_points.loc[i, 'Z'] += 0.0000001
                    self.mesh.points.loc[i, 'Z'] += 0.0000001
        cutbox = BoundingBox(self.mesh)
        if self.type != 'Z':
            cutbox.column = self.type  # 指定包围盒z方向数据列名
        boxnode = cutbox.create_tree(cutbox.dataset, cutbox.dataset.shape[0], 4)
        self.find_point(boxnode)
        self.index = startid
        # self.edges.to_csv('./TrglMesh_Data/New_edge_test.csv', encoding='utf-8-sig')

    def pd_z_box(self, z, box):
        if self.type != 'Z':
            return box.min[3] < z < box.max[3]
        else:
            return box.min[2] < z < box.max[2]

    # 存储等值点数据
    def add_points(self, id_edges):
        if len(id_edges) != 2:
            return
        ids = []
        for point in id_edges:
            x, y = point[0], point[1]
            condition = (self.edges['X'] == x) & (self.edges['Y'] == y)
            if condition.any():
                index = self.edges[condition].index.tolist()[0]
                ids.append(index)
            else:
                ids.append(len(self.edges))
                if self.type != 'Z':
                    self.edges.loc[len(self.edges)] = [len(self.edges), x, y, point[2], []]
                else:
                    self.edges.loc[len(self.edges)] = [len(self.edges), x, y, self.v, []]
        self.edges.loc[ids[0], 'point'].append(ids[1])
        self.edges.loc[ids[1], 'point'].append(ids[0])
        return

    # 寻找等值线点
    def find_point(self, boxnode):
        if boxnode is None:
            return
        # 包围盒与等值面不相交
        if self.pd_z_box(self.v, boxnode.bounding_box) is False:
            return
        if boxnode.isleaf is True:
            # 三角形id数组
            ids = list(boxnode.bounding_box.dataset.index)
            # 单个三角形的处理
            for id in ids:
                id_edges = []  # 存储同一个三角形的交点
                if self.type != 'Z':
                    v = self.v
                    if self.mesh.dataset.loc[id, f'{self.type}_min'] > v or self.mesh.dataset.loc[
                        id, f'{self.type}_max'] < v:
                        continue
                    else:
                        x0, y0, z0, v0 = self.mesh.read_tri_v(id)[0]
                        x1, y1, z1, v1 = self.mesh.read_tri_v(id)[1]
                        x2, y2, z2, v2 = self.mesh.read_tri_v(id)[2]
                        if (v - v0) * (v - v1) < 0:
                            x = x0 + (x1 - x0) * (v - v0) / (v1 - v0)
                            y = y0 + (y1 - y0) * (v - v0) / (v1 - v0)
                            z = z0 + (z1 - z0) * (v - v0) / (v1 - v0)
                            id_edges.append([x, y, z])
                        if (v - v0) * (v - v2) < 0:
                            x = x0 + (x2 - x0) * (v - v0) / (v2 - v0)
                            y = y0 + (y2 - y0) * (v - v0) / (v2 - v0)
                            z = z0 + (z2 - z0) * (v - v0) / (v2 - v0)
                            id_edges.append([x, y, z])
                        if (v - v1) * (v - v2) < 0:
                            x = x1 + (x2 - x1) * (v - v1) / (v2 - v1)
                            y = y1 + (y2 - y1) * (v - v1) / (v2 - v1)
                            z = z1 + (z2 - z1) * (v - v1) / (v2 - v1)
                            id_edges.append([x, y, z])
                        self.add_points(id_edges)
                else:
                    z = self.v
                    if self.mesh.dataset.loc[id, 'z_min'] > z or self.mesh.dataset.loc[
                        id, 'z_max'] < z:
                        continue
                    else:
                        x0, y0, z0 = self.mesh.read_tri_v(id)[0]
                        x1, y1, z1 = self.mesh.read_tri_v(id)[1]
                        x2, y2, z2 = self.mesh.read_tri_v(id)[2]
                        if (z - z0) * (z - z1) < 0:
                            x = x0 + (x1 - x0) * (z - z0) / (z1 - z0)
                            y = y0 + (y1 - y0) * (z - z0) / (z1 - z0)
                            id_edges.append([x, y])
                        if (z - z0) * (z - z2) < 0:
                            x = x0 + (x2 - x0) * (z - z0) / (z2 - z0)
                            y = y0 + (y2 - y0) * (z - z0) / (z2 - z0)
                            id_edges.append([x, y])
                        if (z - z1) * (z - z2) < 0:
                            x = x1 + (x2 - x1) * (z - z1) / (z2 - z1)
                            y = y1 + (y2 - y1) * (z - z1) / (z2 - z1)
                            id_edges.append([x, y])
                        self.add_points(id_edges)
            return
        else:
            self.find_point(boxnode.b1)
            self.find_point(boxnode.b2)
            self.find_point(boxnode.b3)
            self.find_point(boxnode.b4)
            self.find_point(boxnode.b5)
            self.find_point(boxnode.b6)
            self.find_point(boxnode.b7)
            self.find_point(boxnode.b8)
        return

    # 判断封闭等值线
    def pd_closed_line(self):
        res = []
        for i in range(self.edges.shape[0]):
            edge_index = self.edges.iloc[i, 0]
            if len(self.edges.loc[edge_index, 'point']) == 1:
                res.append(edge_index)
        return res

    # 判断等值线是否在edges数据中，在，返回索引index，否则返回None
    def pd_point_in(self, point_index):
        condition = self.edges['id'] == point_index
        if condition.any():
            index = self.edges[condition].index.tolist()[0]
            return index
        else:
            return None

    # 等值线追踪
    def track_contourline(self):
        res = []
        closed = False
        while self.edges.shape[0] > 0:
            # 闭合等值线处理
            if len(self.pd_closed_line()) == 0:
                print('闭合等值线')
                closed = True
                startpoint = int(self.edges.iloc[0, 0])
            else:
                print('非闭合等值线')
                startpoint = int(self.pd_closed_line()[0])
            # 初始化起点坐标
            self.contourline = {'id': self.index, 'name': f'{self.type}{self.v}等值线', 'value': self.v,
                                'labelpositions': [], 'positions': []}
            self.contourline['positions'].extend(
                self.edges.loc[self.edges['id'] == startpoint, ['X', 'Y', 'Z']].values.tolist()[0])
            # 与当前点相连的点的索引
            print('起点edge', startpoint, closed)
            point = self.edges.loc[startpoint, 'point']
            # 非闭合等值线删除起点数据
            if closed is False:
                self.edges = self.edges.drop(self.edges[self.edges['id'] == startpoint].index)

            # 寻找下一个点
            for point_id in point:
                if self.pd_point_in(point_id) is not None:
                    curpoint = point_id
                    self.track_line(startpoint, curpoint, startpoint)
            self.get_labelpositions()
            res.append(self.contourline.copy())
        return res

    # 递归追踪
    def track_line(self, startpoint, curpoint, prepoint):
        # 将当前边的等值线点添加到结果数组中
        self.contourline['positions'].extend(
            self.edges.loc[self.edges['id'] == curpoint, ['X', 'Y', 'Z']].values.tolist()[0])
        # 与当前点相连的点的索引
        point = self.edges.loc[curpoint, 'point']
        self.edges = self.edges.drop(self.edges[self.edges['id'] == curpoint].index)
        # 等值线闭合或到达边界
        if startpoint == curpoint or len(point) == 1:
            print('终止标记', startpoint, curpoint, self.edges.shape[0])
            self.index += 1
            return
        for point_id in point:
            if self.pd_point_in(point_id) is None or point_id == prepoint:
                continue
            prepoint = curpoint
            curpoint = point_id
            self.track_line(startpoint, curpoint, prepoint)
        return

    # 设定默认标签位置
    def get_labelpositions(self):
        index = len(self.contourline['positions']) // 2
        while index % 3 != 0:
            index -= 1
        self.contourline['labelpositions'] = self.contourline['positions'][index: (index + 3)]
        return
# 117.00951535309501, 33.395744747755465, -380.0

if __name__ == "__main__":
    mesh_path = "AutoCAD模型输出/"
    mesh_filename = "Drawing4_15res"
    json_path = "E:/SuiDaoProject/jscode/Vue3-Vite-Cesium_V1.8.8/public/js/json/line/"
    data_dxf = Read_dxf(mesh_path, mesh_filename).read_polygons()
    meshdata = []
    for key in data_dxf.keys():
        points = data_dxf[key][0]
        trgls = data_dxf[key][1]
        mesh = TrglMesh(key, points, trgls)
        mesh.back_pv_csv()
        if key == "New_GCD":
            mesh.id = 0
    mesh.points = mesh.cgcs_to_wgs()
    cutbox = BoundingBox(mesh)

    # c = ContourPoints(0, 50, mesh)
    # data = c.track_contourline()
    # for i in data:
    #     print(i)
    # # print(data)

    z = 20
    step = 10
    id = 0
    data = []
    flag = 1
    while True:
        c = ContourPoints(id, z, mesh)
        print(z)
        item = c.track_contourline()
        if len(item) == 0 and flag == 1:
            z += step
            continue
        elif len(item) == 0 and flag == 0:
            break
        else:
            flag = 0
        data.extend(item)
        id = len(data)
        z += step
    print(data)
    with open(json_path + mesh.name + '等高线' + '.json', 'w') as json_file:
        json.dump(data, json_file, indent=2)
