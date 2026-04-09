from 备份代码.包围盒类 import *
import pyvista as pv


color_dic = {
    "地表": [0, 255, 255, 1.0],
    "杂填土": [191, 255, 0, 1.0],
    "黏土": [255, 191, 0, 1.0],
    "泥灰岩": [0, 255, 255, 1.0],
    "石灰岩": [137, 137, 137, 1.0],
    "页岩": [255, 127, 191, 1.0],
    "溶洞无充填": [255, 0, 0, 0.7],
    "溶洞半充填": [255, 127, 0, 0.7],
    "溶洞全充填": [0, 255, 0, 0.7],
}
# 射线求交算法
def triangle_intersect(v0, v1, v2, orig, end):
    dir = end - orig
    E1 = v1 - v0
    E2 = v2 - v0
    S = orig - v0
    S1 = np.cross(dir, E2)
    S2 = np.cross(S, E1)

    if np.dot(S1, E1) == 0:
        return '射线与三角形共面'
    coeff = 1.0 / np.dot(S1, E1)
    t = coeff * np.dot(S2, E2)
    b1 = coeff * np.dot(S1, S)
    b2 = coeff * np.dot(S2, dir)

    if 0 <= t <= 1 and b1 >= 0 and b2 >= 0 and (1 - b1 - b2) >= 0:
        intersection_point = orig + t * dir  # 交点坐标
        return intersection_point
    else:
        return False


# 计算两个三角形交线
def cal_intersect_line(trglmesh1: TrglMesh, trglmesh2: TrglMesh, tri_id1=None, tri_id2=None):
    if tri_id1 is None:
        tri_id1 = list(trglmesh1.trgls.index)
    if tri_id2 is None:
        tri_id2 = list(trglmesh2.trgls.index)
    linepoints = []
    for id1 in tri_id1:
        for id2 in tri_id2:
            v11 = trglmesh1.read_tri_v(id1)[0]
            v12 = trglmesh1.read_tri_v(id1)[1]
            v13 = trglmesh1.read_tri_v(id1)[2]
            v21 = trglmesh2.read_tri_v(id2)[0]
            v22 = trglmesh2.read_tri_v(id2)[1]
            v23 = trglmesh2.read_tri_v(id2)[2]
            res1 = triangle_intersect(v11, v12, v13, v21, v22)
            res2 = triangle_intersect(v11, v12, v13, v22, v23)
            res3 = triangle_intersect(v11, v12, v13, v23, v21)
            res4 = triangle_intersect(v21, v22, v23, v11, v12)
            res5 = triangle_intersect(v21, v22, v23, v12, v13)
            res6 = triangle_intersect(v21, v22, v23, v13, v11)
            temp = [res1, res2, res3, res4, res5, res6]
            for i in temp.copy():
                if i is not False:
                    linepoints.append(i)
    if linepoints:
        return linepoints
    else:
        return None


# 点和包围盒位置判断（点是否在包围盒内部）
def pd_point_box(point, box):
    for i in range(3):
        if point[i] > box.max[i] or point[i] < box.min[i]:
            return False
    return True


# 线段和包围盒位置判断（线段和包围盒是否有交叉）
def pd_line_box(startpoint, point, cutbox):
    for i in range(3):
        if (startpoint[i] < cutbox.min[i] and point[i] < cutbox.min[i]) or (
                startpoint[i] > cutbox.max[i] and point[i] > cutbox.max[i]):
            return False
    return True


# 建立包围盒树，排除不含point的数据集，并判断startpoint到point的线段是否与cutmesh有交点（有则异侧，无则同侧）
class Pd_inbox_point:
    def __init__(self, startpoint, point, boxnode, cutmesh):
        self.count = 0
        self.flag = False
        self.resPoint = None
        self.pd_inbox_point(startpoint, point, boxnode, cutmesh)

    def pd_inbox_point(self, startpoint, point, boxnode, cutmesh):
        if self.flag:
            return
        if boxnode is None:
            return
        if pd_line_box(startpoint, point, boxnode.bounding_box) is False:
            return
        # 是叶子节点
        if boxnode.isleaf is True:
            # 点在包围盒内部
            ids = list(boxnode.bounding_box.dataset.index)
            for id in ids:
                self.count += 1
                v11 = cutmesh.read_tri_v(id)[0]
                v12 = cutmesh.read_tri_v(id)[1]
                v13 = cutmesh.read_tri_v(id)[2]
                # print(v11, v12, v13, startpoint, point)
                res = triangle_intersect(v11, v12, v13, startpoint, point)
                if res is not False:
                    self.flag = True
                    self.resPoint = res
                    return
        else:
            self.pd_inbox_point(startpoint, point, boxnode.b1, cutmesh)
            self.pd_inbox_point(startpoint, point, boxnode.b2, cutmesh)
            self.pd_inbox_point(startpoint, point, boxnode.b3, cutmesh)
            self.pd_inbox_point(startpoint, point, boxnode.b4, cutmesh)
            self.pd_inbox_point(startpoint, point, boxnode.b5, cutmesh)
            self.pd_inbox_point(startpoint, point, boxnode.b6, cutmesh)
            self.pd_inbox_point(startpoint, point, boxnode.b7, cutmesh)
            self.pd_inbox_point(startpoint, point, boxnode.b8, cutmesh)
        return


# 添加交线点
def add_insert_points(cutmeshname, trglmeshname):
    point_path = "./data/Line_Points/"
    data = pd.read_csv(point_path + cutmeshname + '_' + trglmeshname + '.csv', encoding='utf-8-sig')
    data = np.array(data.loc[:, ['X', 'Y', 'Z']]) * 25.4  # 单位转换
    return data


# 交线两侧点云数据分割
def split_trgl(trglmesh, cutmesh, point_path, startpoint=None):  # startpoint:numpy数组
    trglscount = 0
    cutbox = BoundingBox(cutmesh)
    # 未输入起始点时寻找一个起点
    if startpoint is None:
        # 定义起点为模型XY中间位置，Z轴最低点
        startpoint = [(trglmesh.max[0] - trglmesh.min[0]) / 2 + trglmesh.min[0],
                      (trglmesh.max[1] - trglmesh.min[1]) / 2 + trglmesh.min[1], trglmesh.min[2]]
        print(startpoint)
    part1 = []
    part2 = []
    # 遍历点的位置状态
    for i in range(trglmesh.points.shape[0]):
        point = np.array(trglmesh.points.loc[i, :])
        # 点不在包围盒内部

        if pd_point_box(point, cutbox) is False:
            # 线段和包围盒不交叉，同侧
            if pd_line_box(startpoint, point, cutbox) is False:
                part1.append(point)
            # 异侧
            else:
                part2.append(point)
        # 点在包围盒内部
        else:
            boxnode = cutbox.create_tree(cutbox.dataset, cutbox.dataset.shape[0], 4)
            # 若线段与切割面没有交叉
            Pd = Pd_inbox_point(startpoint, point, boxnode, cutmesh)
            if Pd.flag is False:
                part1.append(point)
            else:
                part2.append(point)
            trglscount += Pd.count
    for i in add_insert_points(cutmesh.name, trglmesh.name):
        part1.append(i)
        part2.append(i)
    part1 = pd.DataFrame(part1, columns=['X', 'Y', 'Z'])
    part1.index.name = 'id'
    part1.to_csv(point_path + cutmesh.name + '_' + trglmesh.name + '_part1.csv', encoding='utf-8-sig')
    part2 = pd.DataFrame(part2, columns=['X', 'Y', 'Z'])
    part2.index.name = 'id'
    part2.to_csv(point_path + cutmesh.name + '_' + trglmesh.name + '_part2.csv', encoding='utf-8-sig')
    return part1, part2, trglscount


if __name__ == "__main__":
    # res = triangle_intersect(np.array([0,0,0]), np.array([0,1,0]), np.array([1,0,0]), np.array([0.5,0.5,0.1]), np.array([0.5,0,0.2]))
    # print(res)
    # 绘图
    plt = Plotter(axes=dict(xtitle="x", ytitle="y", ztitle="z", yzgrid=False), bg="white", bg2="white", )
    point_path = "data/Point_Parts/"
    data_path = "AutoCAD模型输出/"
    filename = "backdata3_21"
    data_dxf = Read_dxf(data_path, filename)
    res = data_dxf.read_polygons()
    for key in res.keys():
        if key == "New_Ztt":
            mesh1 = TrglMesh('杂填土', res['New_Ztt'][0], res['New_Ztt'][1])
            mesh1.back_pv_csv()
            # plt += Mesh([np.array(mesh1.points), np.array(mesh1.trgls)]).color("green").backcolor("green")
        elif key == "New_Nt":
            mesh2 = TrglMesh('黏土', res['New_Nt'][0], res['New_Nt'][1])
            mesh2.back_pv_csv()
            # plt += Mesh([np.array(mesh2.points), np.array(mesh2.trgls)]).color("yellow").backcolor("yellow")
        # elif key == "New_GCD":
        #     mesh3 = TrglMesh('地表', res['New_GCD'][0], res['New_GCD'][1])
        # else:
        #     mesh = TrglMesh(key, res[key][0], res[key][1])
        #     mesh.back_pv_csv()
        #     plt += Mesh([np.array(mesh.points), np.array(mesh.trgls)]).color("violet").backcolor("violet").lw(1.0)
    # # 第一次分割——杂填土
    # for key in res.keys():
    #     if key != "New_Ztt" and key != "New_Nt":
    #         mesh = TrglMesh(key, res[key][0], res[key][1])
    #         mesh.back_pv_csv()
    #         print(f'{key}开始曲面分割')
    #         time_start = time.time()
    #         part1, part2, count = split_trgl(mesh, mesh1, point_path)
    #         time_end = time.time()
    #         time_c = time_end - time_start
    #         print(f'{mesh1.name}和{key}总时间花费： {time_c} s，共涉及{count}个三角形')


    # S1 = ''
    # S2 = ''
    # point_path = "data/Point_Parts/"
    # data_path = "AutoCAD模型输出/"
    # filename = "Drawing4_15res"
    # data_dxf = Read_dxf(data_path, filename)
    # res = data_dxf.read_polygons()
    # for key in res.keys():
    #     if key == "New_Ztt":
    #         S1 = TrglMesh('杂填土', res['New_Ztt'][0], res['New_Ztt'][1])
    #     elif key == 'New_C1h_nhy2':
    #         S2 = TrglMesh(key, res[key][0], res[key][1])
    # s = Collide_Detection(S1, S2, 4, 4)
    # s.Collide_Detection_Trgl()


    # 第二次分割——黏土
    # pyvista绘图方案
    # p = pv.Plotter()
    # for file in res.keys():
    #     if file != "New_Ztt" and file != "New_Nt" and file != "New_GCD":
    #         meshname = file
    #         points = pd.read_csv(point_path + '杂填土'+'_'+meshname+'_part1.csv',encoding='utf-8-sig')
    #         points.set_index('id', inplace=True)
    #         points2 = pd.read_csv(point_path + '杂填土'+'_'+meshname+'_part2.csv',encoding='utf-8-sig')
    #         points2.set_index('id', inplace=True)
    #         pv_mesh = pv.PolyData(np.array(points.loc[:, ['X', 'Y', 'Z']])).delaunay_2d()
    #         pv_mesh2 = pv.PolyData(np.array(points2.loc[:, ['X', 'Y', 'Z']])).delaunay_2d()
    #         p.add_mesh(pv_mesh, color='violet', show_edges=False)
    #         p.add_mesh(pv_mesh2, color='green', show_edges=False)
    #         # edges = pv_mesh.extract_feature_edges()
    #         # print(edges.points)
    #         # print(edges)
    #         # p.add_mesh(edges, line_width=10)
    #         # p.add_points(edges.points, point_size=10)
    #
    #
    #         # points = pd.read_csv(point_path + '黏土' + '_' + meshname + '_part1.csv', encoding='utf-8-sig')
    #         # points.set_index('id', inplace=True)
    #         # pv_mesh = pv.PolyData(np.array(points.loc[:, ['X', 'Y', 'Z']])).delaunay_2d()
    #         # p.add_mesh(pv_mesh, color='violet', show_edges=False)
    #
    #
    #
    #         # pv_points = pv_mesh.points
    #         # pv_trgls = pv_mesh.faces
    #         # res = []
    #         # for i in range(0, len(pv_trgls), 4):
    #         #     res.append(pv_trgls[i + 1:i + 4])
    #         # res = pd.DataFrame(res, columns=["t1", "t2", "t3"]).astype(int)
    #         # trgls = res
    #         # mesh = TrglMesh(meshname, points, trgls)
    #         # mesh.back_pv_csv()
    #         # print(mesh2.name)
    #         # print(f'{mesh.name}开始曲面分割')
    #         # time_start = time.time()
    #         # part1, part2, count = split_trgl(mesh, mesh2, point_path)
    #         # time_end = time.time()
    #         # time_c = time_end - time_start
    #         # print(f'{mesh2.name}和{mesh.name}总时间花费： {time_c} s，共涉及{count}个三角形')
    #
    # m1 = mesh1.triangulate()
    # p.add_mesh(m1, color='orange')
    # # m2 = mesh2.triangulate()
    # # p.add_mesh(m2, color='green')
    # # m3 = mesh3.triangulate()
    # # p.add_mesh(m3, color='blue')
    # # e1 = m1.extract_feature_edges()
    # # e2 = m2.extract_feature_edges()
    # # p.add_mesh(e1, line_width=10)
    # # p.add_mesh(e2, line_width=10)
    # p.show()

    # vedo绘图方案
    # plt = Plotter(axes=dict(xtitle="x", ytitle="y", ztitle="z", yzgrid=False), bg="white", bg2="white", )
    for file in res.keys():
        if file != "New_Ztt" and file != "New_Nt" and file != "New_GCD":
            meshname = file
            # if meshname != 'New_C1m_nhy12':
            #     continue
            if meshname.split('_')[2].startswith('shy'):
                color = color_dic['石灰岩'][:3]
            elif meshname.split('_')[2].startswith('nhy'):
                color = color_dic['泥灰岩'][:3]
            elif meshname.split('_')[2].startswith('yy'):
                color = color_dic['页岩'][:3]
            points = pd.read_csv(point_path + '杂填土' + '_' + meshname + '_part1.csv', encoding='utf-8-sig')
            points.set_index('id', inplace=True)
            mesh = pv.PolyData(np.array(points.loc[:, ['X', 'Y', 'Z']])).delaunay_2d()
            pv_trgls = mesh.faces
            trgl = []
            for i in range(0, len(pv_trgls), 4):
                trgl.append(pv_trgls[i + 1:i + 4])
            trgl = pd.DataFrame(trgl, columns=["t1", "t2", "t3"]).astype(int)
            # plt += Mesh([np.array(points), np.array(trgl)]).color("violet").backcolor("violet").lw(1.0)
            plt += Mesh([np.array(points), np.array(trgl)]).color(color).backcolor(color)

            points = pd.read_csv(point_path + '杂填土' + '_' + meshname + '_part2.csv', encoding='utf-8-sig')
            points.set_index('id', inplace=True)
            mesh = pv.PolyData(np.array(points.loc[:, ['X', 'Y', 'Z']])).delaunay_2d()
            pv_trgls = mesh.faces
            trgl = []
            for i in range(0, len(pv_trgls), 4):
                trgl.append(pv_trgls[i + 1:i + 4])
            trgl = pd.DataFrame(trgl, columns=["t1", "t2", "t3"]).astype(int)
            # plt += Mesh([np.array(points), np.array(trgl)]).color("yellow").backcolor("yellow").alpha(0.65).lw(1.0)
            # plt += Mesh([np.array(points), np.array(trgl)]).color(color).backcolor(color).alpha(0.65)

    plt += Mesh([np.array(mesh1.points), np.array(mesh1.trgls)]).color("green").backcolor("green")
    plt += Mesh([np.array(mesh2.points), np.array(mesh2.trgls)]).color("yellow").backcolor("yellow")
    # plt.show()

    # 添加交线
    os.chdir("./data/Line_Points/")
    file_chdir = os.getcwd()
    csv_list = []
    for root, dirs, files in os.walk(file_chdir):
        for file in files:
            if file[0] == "杂":
                # if file[0] == "黏":
                # if os.path.splitext(file)[-1] == ".csv":
                #     print(os.path.splitext(file))
                # if file != '杂填土_New_C1m_nhy12.csv':
                #     continue
                line = pd.read_csv(file, encoding="utf-8-sig")
                line = np.array(line.loc[:, ["X", "Y", "Z"]]) * 25.4
                line = Line(line, lw=3, c="red")
                plt += line
    plt.show()

    # data_path = "E:/3DModel_YLH/formations/"
    # file1name = "surfaces1_2.ts"
    # file2name = "Back_surfaces11_03.ts"
    # data = read_TS(data_path, file1name)
    # backdata = read_TS(data_path, file2name)
    # print(data[0].name, data[0].trgls.index)  # 杂填土，共9510个三角形
    # print(backdata[0].name, backdata[0].trgls.index)  # ∈1h泥灰岩2，共1494个三角形
    # res = cal_intersect_line(data[0], backdata[0])
    # print(len(res))
    # print(res)
    # df = pd.DataFrame(res)
    # df.to_csv("testpoints.csv", encoding="utf-8-sig")

    # # 相交测试
    # data1 = np.array([[0, 0.5, 0], [1, 0, 0], [0, 1, 10]])
    # data2 = np.array([[0.5, 0.5, -1], [0.2, -0.8, 0.5], [0.8, 0.8, 1]])
    #
    # linepoints = []
    # v11 = data1[0]
    # v12 = data1[1]
    # v13 = data1[2]
    # v21 = data2[0]
    # v22 = data2[1]
    # v23 = data2[2]
    # res1 = triangle_intersect(v11, v12, v13, v21, v22)
    # res2 = triangle_intersect(v11, v12, v13, v22, v23)
    # res3 = triangle_intersect(v11, v12, v13, v23, v21)
    # res4 = triangle_intersect(v21, v22, v23, v11, v12)
    # res5 = triangle_intersect(v21, v22, v23, v12, v13)
    # res6 = triangle_intersect(v21, v22, v23, v13, v11)
    # temp = [res1, res2, res3, res4, res5, res6]
    # for i in temp:
    #     if i is not False:
    #         linepoints.append(i)
    # print(linepoints)
    # for point in linepoints:
    #     x, y, z = point[0], point[1], point[2]
    #     print(x, y, z)
    #
    # # plt = Plotter(axes=dict(xtitle="x", ytitle="y", ztitle="z", yzgrid=False), bg2="lb", )
    # # mesh1 = Mesh([data1, [[0, 1, 2]]])
    # # mesh2 = Mesh([data2, [[0, 1, 2]]])
    # # intersectline = Line(linepoints[0], linepoints[1]).color("green").opacity(1).lw(2)
    # # mesh1.color("red").backcolor("red").opacity(0.5)
    # # mesh2.color("blue").backcolor("blue").opacity(0.5)
    # # plt += mesh1
    # # plt += mesh2
    # # plt += intersectline
    # # plt.show()
