import os
import mujoco
from mujoco.usd import exporter

# ================= 配置区 =================
# 1. 你的模型文件路径 (.xml)
XML_PATH = 'vbot.xml' 

# 2. 导出设置
OUTPUT_DIR = 'mujoco_usd_export'      # 导出的文件夹名称
DURATION = 5                          # 仿真持续时间 (秒)
FRAMERATE = 60                        # 视频帧率 (FPS)
CAMERA_NAME = None                    # 指定摄像机名称，若为 None 则使用默认
# ==========================================

def run_usd_export():
    # 检查模型文件是否存在
    if not os.path.exists(XML_PATH):
        print(f"错误: 找不到模型文件 {XML_PATH}")
        return

    print("正在加载模型...")
    # 加载模型和数据
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 初始化 USD 导出器
    # 你可以在这里添加 camera_names=['cam_1'] 等参数
    exp = exporter.USDExporter(
        model=model,
        output_directory=OUTPUT_DIR,
        verbose=True
    )

    print(f"开始仿真导出 (持续时间: {DURATION}s, 帧率: {FRAMERATE}fps)...")

    # 运行物理仿真循环
    while data.time < DURATION:
        # 进行一次物理步进
        mujoco.mj_step(model, data)

        # 检查是否需要更新 USD 帧
        # 逻辑：当导出器的帧数进度落后于仿真时间进度时，保存当前位姿
        if exp.frame_count < data.time * FRAMERATE:
            exp.update_scene(data=data)

    # 导出最终的 USD 文件
    print("\n正在保存 USD 文件及其素材...")
    exp.save_scene(filetype="usd")
    
    print(f"导出完成！文件保存在目录: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    run_usd_export()
