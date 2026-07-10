# 地形生成器：基于输入场景 XML 生成包含多种地形的 MuJoCo 场景文件
# 支持立方体、几何体、楼梯、悬浮楼梯、杂乱地面、Perlin 噪声高程图及图像高程图等地形

import xml.etree.ElementTree as xml_et
import numpy as np
import cv2
import noise

# 机器人名称
ROBOT = "legbot"
# 输入场景文件路径
INPUT_SCENE_PATH = "./scene.xml"
# 输出地形场景文件路径
OUTPUT_SCENE_PATH = "../legbot/xmls/scene_terrain.xml"


# 将 ZYX 欧拉角转换为四元数
def euler_to_quat(roll, pitch, yaw):
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)

    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


# 将 ZYX 欧拉角转换为旋转矩阵
def euler_to_rot(roll, pitch, yaw):
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )

    rot_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


# 二维旋转
def rot2d(x, y, yaw):
    nx = x * np.cos(yaw) - y * np.sin(yaw)
    ny = x * np.sin(yaw) + y * np.cos(yaw)
    return nx, ny


# 三维旋转
def rot3d(pos, euler):
    R = euler_to_rot(euler[0], euler[1], euler[2])
    return R @ pos


# 将向量转换为空格分隔的字符串
def list_to_str(vec):
    return " ".join(str(s) for s in vec)


class TerrainGenerator:

    def __init__(self) -> None:
        # 解析输入场景 XML
        self.scene = xml_et.parse(INPUT_SCENE_PATH)
        self.root = self.scene.getroot()
        # 定位世界体节点与资源节点
        self.worldbody = self.root.find("worldbody")
        self.asset = self.root.find("asset")

    # 添加立方体几何体
    def AddBox(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0],
               size=[0.1, 0.1, 0.1]):
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = "box"
        # MuJoCo 中立方体尺寸使用半边长
        geo.attrib["size"] = list_to_str(0.5 * np.array(size))
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    # 添加通用几何体
    def AddGeometry(self,
                    position=[1.0, 0.0, 0.0],
                    euler=[0.0, 0.0, 0.0],
                    size=[0.1, 0.1],
                    geo_type="box"):
        # 支持的几何体类型：plane、sphere、capsule、ellipsoid、cylinder、box
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = geo_type
        # 立方体类型在 MuJoCo 中使用半边长
        geo.attrib["size"] = list_to_str(0.5 * np.array(size))
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    # 添加连续楼梯
    def AddStairs(self,
                  init_pos=[1.0, 0.0, 0.0],
                  yaw=0.0,
                  width=0.2,
                  height=0.15,
                  length=1.5,
                  stair_nums=10):

        local_pos = [0.0, 0.0, -0.5 * height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]],
                        [0.0, 0.0, yaw], [width, length, height])

    # 添加悬浮楼梯
    def AddSuspendStairs(self,
                         init_pos=[1.0, 0.0, 0.0],
                         yaw=1.0,
                         width=0.2,
                         height=0.15,
                         length=1.5,
                         gap=0.1,
                         stair_nums=10):

        local_pos = [0.0, 0.0, -0.5 * height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]],
                        [0.0, 0.0, yaw],
                        [width, length, abs(height - gap)])

    # 添加杂乱地面
    def AddRoughGround(self,
                       init_pos=[1.0, 0.0, 0.0],
                       euler=[0.0, -0.0, 0.0],
                       nums=[10, 10],
                       box_size=[0.5, 0.5, 0.5],
                       box_euler=[0.0, 0.0, 0.0],
                       separation=[0.2, 0.2],
                       box_size_rand=[0.05, 0.05, 0.05],
                       box_euler_rand=[0.2, 0.2, 0.2],
                       separation_rand=[0.05, 0.05]):

        local_pos = [0.0, 0.0, -0.5 * box_size[2]]
        new_separation = np.array(separation) + np.array(
            separation_rand) * np.random.uniform(-1.0, 1.0, 2)
        for i in range(nums[0]):
            local_pos[0] += new_separation[0]
            local_pos[1] = 0.0
            for j in range(nums[1]):
                # 随机扰动尺寸、姿态与间距
                new_box_size = np.array(box_size) + np.array(
                    box_size_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_box_euler = np.array(box_euler) + np.array(
                    box_euler_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_separation = np.array(separation) + np.array(
                    separation_rand) * np.random.uniform(-1.0, 1.0, 2)

                local_pos[1] += new_separation[1]
                pos = rot3d(local_pos, euler) + np.array(init_pos)
                self.AddBox(pos, new_box_euler, new_box_size)

    # 基于 Perlin 噪声生成高程图地形
    def AddPerlinHeighField(
            self,
            position=[1.0, 0.0, 0.0],  # 地形中心位置
            euler=[0.0, -0.0, 0.0],  # 地形姿态
            size=[1.0, 1.0],  # 地形长宽
            height_scale=0.2,  # 最大高度
            negative_height=0.2,  # z 轴负向高度
            image_width=128,  # 高程图像素宽度
            img_height=128,  # 高程图像素高度
            smooth=100.0,  # 噪声平滑尺度
            perlin_octaves=6,  # Perlin 噪声倍频数
            perlin_persistence=0.5,
            perlin_lacunarity=2.0,
            output_hfield_image="height_field.png"):

        # 根据 Perlin 噪声生成灰度高程图
        terrain_image = np.zeros((img_height, image_width), dtype=np.uint8)
        for y in range(image_width):
            for x in range(image_width):
                noise_value = noise.pnoise2(x / smooth,
                                            y / smooth,
                                            octaves=perlin_octaves,
                                            persistence=perlin_persistence,
                                            lacunarity=perlin_lacunarity)
                # 将噪声值从 [-1, 1] 映射到 [0, 255]
                terrain_image[y, x] = int((noise_value + 1) / 2 * 255)

        cv2.imwrite("../legbot/xmls/" + output_hfield_image,
                    terrain_image)

        # 注册高程图资源
        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "perlin_hfield"
        # 尺寸属性前两项为半边长，后两项为正向与负向最大高度
        hfield.attrib["size"] = list_to_str(
            [size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = "../" + output_hfield_image

        # 在世界中放置高程图地形
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "perlin_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    # 基于图像生成高程图地形
    def AddHeighFieldFromImage(
            self,
            position=[1.0, 0.0, 0.0],  # 地形中心位置
            euler=[0.0, -0.0, 0.0],  # 地形姿态
            size=[2.0, 1.6],  # 地形长宽
            height_scale=0.02,  # 最大高度
            negative_height=0.1,  # z 轴负向高度
            input_img=None,
            output_hfield_image="height_field.png",
            image_scale=[1.0, 1.0],  # 图像缩放比例
            invert_gray=False):

        input_image = cv2.imread(input_img)  # 读取输入图像

        width = int(input_image.shape[1] * image_scale[0])
        height = int(input_image.shape[0] * image_scale[1])
        resized_image = cv2.resize(input_image, (width, height),
                                   interpolation=cv2.INTER_AREA)
        terrain_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        if invert_gray:
            terrain_image = 255 - position
        cv2.imwrite("../legbot/xmls/" + output_hfield_image,
                    terrain_image)

        # 注册高程图资源
        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "image_hfield"
        hfield.attrib["size"] = list_to_str(
            [size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = "../" + output_hfield_image

        # 在世界中放置高程图地形
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "image_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    # 保存生成的场景文件
    def Save(self):
        self.scene.write(OUTPUT_SCENE_PATH)


if __name__ == "__main__":
    tg = TerrainGenerator()

    # 立方体障碍物
    tg.AddBox(position=[1.5, 0.0, 0.1], euler=[0, 0, 0.0], size=[1, 1.5, 0.2])

    # 圆柱体几何体障碍物
    tg.AddGeometry(position=[1.5, 0.0, 0.25], euler=[0, 0, 0.0], size=[1.0, 0.5, 0.5], geo_type="cylinder")

    # 斜坡
    tg.AddBox(position=[2.0, 2.0, 0.5],
              euler=[0.0, -0.5, 0.0],
              size=[3, 1.5, 0.1])

    # 楼梯
    tg.AddStairs(init_pos=[1.0, 4.0, 0.0], yaw=0.0)

    # 悬浮楼梯
    tg.AddSuspendStairs(init_pos=[1.0, 6.0, 0.0], yaw=0.0)

    # 杂乱地面
    tg.AddRoughGround(init_pos=[-2.5, 5.0, 0.0],
                      euler=[0, 0, 0.0],
                      nums=[10, 8])

    # Perlin 噪声高程图
    tg.AddPerlinHeighField(position=[-1.5, 4.0, 0.0], size=[2.0, 1.5])

    # 基于图像的高程图
    tg.AddHeighFieldFromImage(position=[-1.5, 2.0, 0.0],
                              euler=[0, 0, -1.57],
                              size=[2.0, 2.0],
                              input_img="./unitree_robot.jpeg",
                              image_scale=[1.0, 1.0],
                              output_hfield_image="unitree_hfield.png")

    tg.Save()
