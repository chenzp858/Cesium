import pandas as pd

from ModelData import TrglMesh
from 曲面求交运算 import *
from vedo import *
import time
# 多进程运行库
import multiprocessing as mp
import os
from ReadData import *


class BoxTreeNode:
    def __init__(self, b1=None,
                 b2=None,
                 b3=None,
                 b4=None,
                 b5=None,
                 b6=None,
                 b7=None,
                 b8=None, bounding_box=None, isleaf=False):
        self.b1 = b1  # 前左下
        self.b2 = b2  # 前右下
        self.b3 = b3  # 前左上
        self.b4 = b4  # 前右上
        self.b5 = b5  # 后左下
        self.b6 = b6  # 后右下
        self.b7 = b7  # 后左上
        self.b8 = b8  # 后右上
        self.bounding_box = bounding_box  # 当前节点的包围盒
        self.isleaf = isleaf


# 包围盒
class BoundingBox:
    def __init__(self, obj=None, dataset=None):
        self.column = 'z'
        self.min = None
        self.max = None
        self.dataset = dataset
        self.count = 0
        if obj is not None:
            self.name = obj.name
            self.min = np.array(obj.min)
            self.max = np.array(obj.max)
            obj.create_triangle_set()
            self.dataset = obj.dataset
            self.dif = self.max - self.min  # 求各轴长度
            self.mid = self.dif / 2 + self.min  # 求中心点
        elif dataset is not None:
            _min = []
            _max = []
            for j in dataset.columns:
                if j == 'id':
                    continue
                if j.split('_')[1] == 'min':
                    _min.append(dataset[j].min())
                else:
                    _max.append(dataset[j].max())
            self.min = np.array(_min)
            self.max = np.array(_max)
            self.dif = self.max - self.min  # 求各轴长度
            self.mid = self.dif / 2 + self.min  # 求中心点
        self.treenode = None

    # 构造树形结构
    def create_tree(self, df, shape, maxdepth, depth=0):
        depth += 1
        if df.empty:
            return None
        node = BoxTreeNode()
        node.bounding_box = BoundingBox(dataset=df)
        if depth >= maxdepth:
            if node.b1 is None and node.b2 is None and node.b3 is None and node.b4 is None and node.b5 is None and node.b6 is None and node.b7 is None and node.b8 is None:
                node.isleaf = True
                self.count += node.bounding_box.dataset.shape[0]
            return node
        if df.shape[0] <= 10:
            node.isleaf = True
            self.count += node.bounding_box.dataset.shape[0]
            return node
        axis = np.argmax(node.bounding_box.dif)  # 长轴索引，与包围盒范围相对应
        setb1 = df.loc[(df["x_max"] <= node.bounding_box.mid[0]) & (df["y_max"] <= node.bounding_box.mid[1]) & (
                    df[f"{self.column}_max"] <= node.bounding_box.mid[2]), :]
        setb2 = df.loc[(df["x_max"] > node.bounding_box.mid[0]) & (df["y_max"] <= node.bounding_box.mid[1]) & (
                    df[f"{self.column}_max"] <= node.bounding_box.mid[2]), :]
        setb3 = df.loc[(df["x_max"] <= node.bounding_box.mid[0]) & (df["y_max"] <= node.bounding_box.mid[1]) & (
                    df[f"{self.column}_max"] > node.bounding_box.mid[2]), :]
        setb4 = df.loc[(df["x_max"] > node.bounding_box.mid[0]) & (df["y_max"] <= node.bounding_box.mid[1]) & (
                    df[f"{self.column}_max"] > node.bounding_box.mid[2]), :]
        setb5 = df.loc[(df["x_max"] <= node.bounding_box.mid[0]) & (df["y_max"] > node.bounding_box.mid[1]) & (
                    df[f"{self.column}_max"] <= node.bounding_box.mid[2]), :]
        setb6 = df.loc[(df["x_max"] > node.bounding_box.mid[0]) & (df["y_max"] > node.bounding_box.mid[1]) & (
                    df[f"{self.column}_max"] <= node.bounding_box.mid[2]), :]
        setb7 = df.loc[(df["x_max"] <= node.bounding_box.mid[0]) & (df["y_max"] > node.bounding_box.mid[1]) & (
                    df[f"{self.column}_max"] > node.bounding_box.mid[2]), :]
        setb8 = df.loc[(df["x_max"] > node.bounding_box.mid[0]) & (df["y_max"] > node.bounding_box.mid[1]) & (
                    df[f"{self.column}_max"] > node.bounding_box.mid[2]), :]
        if setb1.shape == shape or setb2.shape == shape or setb3.shape == shape or setb4.shape == shape or setb5.shape == shape or setb6.shape == shape or setb7.shape == shape or setb8.shape == shape:
            return
        node.b1 = self.create_tree(setb1, setb1.shape, maxdepth, depth)
        node.b2 = self.create_tree(setb2, setb2.shape, maxdepth, depth)
        node.b3 = self.create_tree(setb3, setb3.shape, maxdepth, depth)
        node.b4 = self.create_tree(setb4, setb4.shape, maxdepth, depth)
        node.b5 = self.create_tree(setb5, setb5.shape, maxdepth, depth)
        node.b6 = self.create_tree(setb6, setb6.shape, maxdepth, depth)
        node.b7 = self.create_tree(setb7, setb7.shape, maxdepth, depth)
        node.b8 = self.create_tree(setb8, setb8.shape, maxdepth, depth)
        self.treenode = node
        return node

    # 读取树形结构
    def read_tree(self):
        pass


# 包围盒碰撞检测静态方法
def Box_Collide_Detection(box1, box2):
    for i in range(3):
        if box1.min[i] > box2.max[i] or box2.min[i] > box1.max[i]:
            return False
    return True


# 碰撞检测算法类
class Collide_Detection:
    def __init__(self, data1: TrglMesh, data2: TrglMesh, maxdepth1, maxdepth2):
        self.time_start = None
        self.time_end = None
        # 存储交线点计算结果
        self.edges = pd.DataFrame(columns=['id', 'X', 'Y', 'Z', 'point'])
        # 用于存放两个包围盒树中发生碰撞的叶节点中的三角形索引数据集
        self.res1 = []
        self.res2 = []
        self.datapath = "./data/"
        self.linepointspath = "./data/Line_Points/"
        # 读取两个三角网数据类
        self.data1 = data1
        self.data2 = data2
        # 指定两个包围盒树的最大递归层数
        self.maxdepth1 = maxdepth1
        self.maxdepth2 = maxdepth2
        # 建立模型包围盒
        self.box1 = BoundingBox(data1)
        self.box2 = BoundingBox(data2)
        # 创建包围盒树
        self.head1 = self.box1.create_tree(self.box1.dataset, self.box1.dataset.shape[0], self.maxdepth1)
        # print('*' * 20, 'box1')
        # print(self.box1.count)
        # print(self.box1.dataset.shape[0])
        self.head2 = self.box2.create_tree(self.box2.dataset, self.box2.dataset.shape[0], self.maxdepth2)
        # print('*' * 20, 'box2')
        # print(self.box2.count)
        # print(self.box2.dataset.shape[0])
        # print('包围盒树创建完成！')
        # return
        # 包围盒树碰撞检测（递归）
        self.time_start = time.time()
        self.Collide_Detection_Tree(self.head1, self.head2)
        self.index = 0

    # （不单独调用）AABB包围盒碰撞检测，递归得到相交三角形索引数据集res1和res2
    def Collide_Detection_Tree(self, n1: BoxTreeNode, n2: BoxTreeNode):
        # 某个节点包围盒数据为空
        if n1 is None or n2 is None:
            return
        if Box_Collide_Detection(n1.bounding_box, n2.bounding_box):
            if n1.isleaf is True:
                if n2.isleaf is True:
                    self.res1.append(n1.bounding_box.dataset.index)
                    self.res2.append(n2.bounding_box.dataset.index)
                    # print('递归到叶子节点，开始计算交线点', len(self.res1[0]), len(self.res2[0]))
                    self.Collide_Detection_Trgl()
                    return
                else:
                    self.Collide_Detection_Tree(n1, n2.b1)
                    self.Collide_Detection_Tree(n1, n2.b2)
                    self.Collide_Detection_Tree(n1, n2.b3)
                    self.Collide_Detection_Tree(n1, n2.b4)
                    self.Collide_Detection_Tree(n1, n2.b5)
                    self.Collide_Detection_Tree(n1, n2.b6)
                    self.Collide_Detection_Tree(n1, n2.b7)
                    self.Collide_Detection_Tree(n1, n2.b8)
            else:
                self.Collide_Detection_Tree(n1.b1, n2)
                self.Collide_Detection_Tree(n1.b2, n2)
                self.Collide_Detection_Tree(n1.b3, n2)
                self.Collide_Detection_Tree(n1.b4, n2)
                self.Collide_Detection_Tree(n1.b5, n2)
                self.Collide_Detection_Tree(n1.b6, n2)
                self.Collide_Detection_Tree(n1.b7, n2)
                self.Collide_Detection_Tree(n1.b8, n2)

    # 潜在三角形碰撞检测
    def Collide_Detection_Trgl(self):
        # 对包围盒中的几何元素求交，得到交点坐标数据集
        # 创建空数据集
        # df = pd.DataFrame(columns=["X", "Y", "Z"])
        for a in self.res1[0]:
            for b in self.res2[0]:
                res = cal_intersect_line(self.data1, self.data2, [a], [b])
                if res is not None:
                    # print('当前交线点计算结束',a,b, res)
                    self.add_points(res)
        self.res1.clear()
        self.res2.clear()

    # 存储计算结果
    def add_points(self, linepoints):
        if len(linepoints) != 2:
            return
        ids = []

        for point in linepoints:
            x, y, z = point[0], point[1], point[2]
            condition = (self.edges['X'] == x) & (self.edges['Y'] == y)
            if condition.any():
                index = self.edges[condition].index.tolist()[0]
                ids.append(index)
            else:
                ids.append(len(self.edges))
                self.edges.loc[len(self.edges)] = [len(self.edges), x, y, z, []]
        self.edges.loc[ids[0], 'point'].append(ids[1])
        self.edges.loc[ids[1], 'point'].append(ids[0])
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

    # 交线追踪
    def track_intersection_line(self):
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
            self.contourline = {'id': self.index, 'positions': []}
            self.contourline['positions'].append(
                self.edges.loc[self.edges['id'] == startpoint].values.tolist()[0])
            # self.edges.loc[self.edges['id'] == startpoint, ['X', 'Y', 'Z', 'point']].values.tolist()[0])
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
            res.append(self.contourline.copy())
        return res

    # 递归追踪
    def track_line(self, startpoint, curpoint, prepoint):
        # 将当前边的等值线点添加到结果数组中
        self.contourline['positions'].append(
            self.edges.loc[self.edges['id'] == curpoint].values.tolist()[0])
        # self.edges.loc[self.edges['id'] == curpoint, ['X', 'Y', 'Z', 'point']].values.tolist()[0])
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

    # 显示三角形碰撞检测的时间花费
    def run_time(self):
        time_c = self.time_end - self.time_start  # 运行所花时间
        print(f'{self.box1.name}和{self.box2.name}的相交检测时间花费：', time_c, 's')
        return time_c

    # 测试共需要计算多少对三角形
    def test_trgls(self):
        count = 0
        # print(len(res1))
        # print(res1)
        for i in range(len(self.res1)):
            for a in self.res1[i]:
                for b in self.res2[i]:
                    count += 1
        print(f'{self.box1.name}和{self.box2.name}的相交检测共{count}对三角形')


def task(i, backdata):
    for key in backdata.keys():
        # j = TrglMesh(key, backdata[key][0], backdata[key][1])
        ij = Collide_Detection(i, backdata[key], 4, 4)  # 后两位为最大递归深度
        # 测试需要检测的三角形对的数量
        ij.test_trgls()
        # 计算交线数据并输出
        ij.Collide_Detection_Trgl()
        ij.run_time()


if __name__ == "__main__":
    time_start = time.time()
    # S1, S2 = '',''
    # data_path = "E:/3DModel_YLH/formations/"
    # file1name = "surfaces1_2.ts"
    # file2name = "Back_surfaces11_03.ts"
    # data = read_TS(data_path, file1name)
    # backdata = read_TS(data_path, file2name)
    # S1 = TrglMesh('杂填土', data['杂填土'][0], data['杂填土'][1])
    # # S2 = TrglMesh('黏土', data['黏土'][0], data['黏土'][1])
    # for key in backdata.keys():
    #     print(key)
    # S2 = TrglMesh('∈1h泥灰岩2', backdata['∈1h泥灰岩2'][0], backdata['∈1h泥灰岩2'][1])

    # # 碰撞检测
    # data_path = "AutoCAD模型输出/"
    # # filename = "Drawing4_15res"
    # filename = "Drawing4_15_nocut"
    # data_dxf = Read_dxf(data_path, filename)
    # res = data_dxf.read_polygons()
    # backdata = {}
    # s1_name, s2_name = "New_Ztt", ''
    # S2 = '', ''
    # results = []
    # S1 = TrglMesh(s1_name, res[s1_name][0], res[s1_name][1])
    # box = BoundingBox(S1)
    # head = box.create_tree(box.dataset, box.dataset.shape[0], 3)
    # _min = head.b8.bounding_box.min
    # _max = head.b8.bounding_box.max
    # for _ in _min:
    #     print(_, end=' ')
    # print()
    # for _ in _max:
    #     print(_, end=' ')
    # for key in res.keys():
    #     if key == "New_Ztt" or key == 'New_Nt':
    #         continue
    #     else:
    #         # if key == 'New_C2z_shy38':
    #         print(key)
    #         s2_name = key
    #         S2 = TrglMesh(key, res[key][0], res[key][1])
    #         for j in range(7, 8):
    #             s = Collide_Detection(S1, S2, 7, j)
    #             # print(s.edges)
    #             s.time_end = time.time()
    #             print(j, end=' ')
    #             s_time = s.run_time()
    #         results.append([key, s_time, len(res[key][1])])
    #         print(s.edges)
    #         res_ = pd.DataFrame(s.track_intersection_line()[0]['positions'], columns=['id', 'X', 'Y', 'Z', 'point'])
    #         res_.to_csv(f'./data/Line_Track/{s1_name}_{s2_name}.csv', encoding='utf-8-sig')
    # s.edges.to_csv(f'./data/Line_Points_1227/{s1_name}_{s2_name}.csv', encoding='utf-8-sig')
    # print(results)
    # results = pd.DataFrame(results, columns=['地层名称', '时间花费', '三角形个数'])
    # results.to_csv('./data/运行结果.csv', encoding='utf-8-sig')

    # p1 = mp.Process(target=task, args=(mesh1, backdata))
    # # p2 = mp.Process(target=task, args=(mesh2, backdata))
    # p1.start()
    # # p2.start()
    # p1.join()
    # # p2.join()
    # time_end = time.time()
    # # 多进程结束后没有中断程序
    # time_c = time_end - time_start  # 运行所花时间
    # print(f'总时间花费：', time_c, 's')

    # 绘图查看交线
    plt = Plotter(axes=dict(xtitle="x", ytitle="y", ztitle="z", yzgrid=False), bg2="white", )

    # data_path = "E:/3DModel_YLH/formations/"
    # file1name = "surfaces1_2.ts"
    # file2name = "Back_surfaces11_03.ts"
    # # 读取点云数据，返回由TrglMesh实例对象构成的列表
    # data = read_TS(data_path, file1name)
    # backdata = read_TS(data_path, file2name)
    # mesh1 = data['杂填土']
    # # vedo可视化创建三角网的点数据序号从0开始
    # plt += Mesh([np.array(mesh1[0]), np.array(mesh1[1])]).color("green").backcolor("green")
    #
    # mesh2 = data['黏土']
    # plt += Mesh([np.array(mesh2[0]), np.array(mesh2[1])]).color("skyblue").backcolor("skyblue")
    #
    # for key in backdata.keys():
    #     mesh = backdata[key]
    #     plt += Mesh([np.array(mesh[0]), np.array(mesh[1])]).color("violet").backcolor("violet")

    data_path = "AutoCAD模型输出/"
    # filename = "backdata3_21"
    filename = "Drawing4_15res"
    data_dxf = Read_dxf(data_path, filename)
    res = data_dxf.read_polygons()
    mesh3 = TrglMesh('地表', res['New_GCD'][0], res['New_GCD'][1])
    mesh3.back_pv_csv()
    plt += Mesh([np.array(mesh3.points), np.array(mesh3.trgls)]).color("skyblue").backcolor("skyblue")
    filename = "Drawing4_15_nocut"
    data_dxf = Read_dxf(data_path, filename)
    res = data_dxf.read_polygons()

    for key in res.keys():
        if key == "New_Ztt":
            mesh1 = TrglMesh('杂填土', res['New_Ztt'][0], res['New_Ztt'][1])
            mesh1.back_pv_csv()
            plt += Mesh([np.array(mesh1.points), np.array(mesh1.trgls)]).color("green").backcolor("green")
        elif key == "New_Nt":
            # continue
            mesh2 = TrglMesh('黏土', res['New_Nt'][0], res['New_Nt'][1])
            mesh2.back_pv_csv()
            plt += Mesh([np.array(mesh2.points), np.array(mesh2.trgls)]).color("yellow").backcolor("yellow")
        elif key == "New_GCD":
            continue
            mesh3 = TrglMesh('地表', res['New_GCD'][0], res['New_GCD'][1])
            mesh3.back_pv_csv()
            plt += Mesh([np.array(mesh3.points), np.array(mesh3.trgls)]).color("skyblue").backcolor("skyblue")
        else:
            if key.split('_')[2].startswith('shy'):
                color = color_dic['石灰岩'][:3]
            elif key.split('_')[2].startswith('nhy'):
                color = color_dic['泥灰岩'][:3]
            elif key.split('_')[2].startswith('yy'):
                color = color_dic['页岩'][:3]
            mesh = TrglMesh(key, res[key][0], res[key][1])
            mesh.back_pv_csv()
            plt += Mesh([np.array(mesh.points), np.array(mesh.trgls)]).color(color).backcolor(color).alpha(1.0)
    #
    # # 添加交线
    # os.chdir("./data/Line_Track/")
    # file_chdir = os.getcwd()
    # csv_list = []
    # for root, dirs, files in os.walk(file_chdir):
    #     for file in files:
    #         # if file[0] == "杂":
    #         # if file[0] == "黏":
    #         # if os.path.splitext(file)[-1] == ".csv":
    #         #     print(os.path.splitext(file))
    #         line = pd.read_csv(file, encoding="utf-8-sig")
    #         line = np.array(line.loc[:, ["X", "Y", "Z"]])
    #         line = Line(line, lw=2, c="red")
    #         plt += line
    plt.show()
