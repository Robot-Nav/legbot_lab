#!/usr/bin/env python3
"""
电机编码器零位移动脚本
======================
将腿关节运动到"电机编码器原始值=0"的物理位置（MIT协议电机输出轴零位）。
轮毂保持阻尼不动。支持选择单条腿或全部腿。

⚠️  警告：此脚本用于机器人初始装配/未安装腿连杆时使用！
    当前offset配置下，大腿需转162°、小腿需转165°，若腿已组装会猛烈撞击机械结构！

控制参数：Kp=30, Kd=1.0，速度限制0.2rad/s(≈11°/s)，线性插值平滑运动。
实时显示各关节当前角度、目标角度和偏差（角度制）。
Ctrl+C中止。

用法：
  python move_to_zero.py                 # 交互式选择腿
  python move_to_zero.py --leg fl        # 只动左前腿
  python move_to_zero.py --leg all       # 动所有腿（默认）
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "controller"))
from motor_client import (
    MotorClient,
    MOTOR_COUNT,
    MOTOR_LABELS,
    WHEEL_INDICES,
    MotorSafetyError,
)

# ---- 控制参数（安全保守值）----
KP = 30.0
KD = 1.0
MAX_SPEED = 0.2          # rad/s (~11°/s)，慢速防撞击
DAMPING_KD = 2.0         # 非控制腿保持阻尼Kd
WHEEL_KD = 1.0
HOLD_KP = 20.0           # 到位后保持刚度
HOLD_KD = 1.0
CONTROL_HZ = 100
PERIOD = 1.0 / CONTROL_HZ
PRINT_INTERVAL = 0.3     # 秒，刷新间隔

# ---- 与can_service_wheel.cpp保持一致的offset参数 ----
kHipOffsetDeg = 25.0
kThighOffsetDeg = 162.0
kCalfRatio = 1.4615
kCalfOffsetDeg = 165.0 * kCalfRatio  # = 241.1475° (电机侧)
kDeg2Rad = np.pi / 180.0

LEG_INDICES = [i for i in range(MOTOR_COUNT) if i not in WHEEL_INDICES]

# 腿定义：每条腿3个关节(hip, thigh, calf)
LEG_NAMES = {
    "fl": "左前腿(Front Left)",
    "fr": "右前腿(Front Right)",
    "rl": "左后腿(Rear Left)",
    "rr": "右后腿(Rear Right)",
}
LEG_JOINTS = {
    "fl": [0, 1, 2],
    "fr": [4, 5, 6],
    "rl": [8, 9, 10],
    "rr": [12, 13, 14],
}

# ANSI
CLEAR = "\033[2J\033[H" if os.name != "nt" else ""
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
GRAY = "\033[90m"
BOLD = "\033[1m"
NC = "\033[0m"


def build_target_positions() -> np.ndarray:
    """计算电机编码器=0时各关节应发送的关节角指令(rad)"""
    target = np.zeros(MOTOR_COUNT, dtype=np.float32)
    for i in range(MOTOR_COUNT):
        if i in WHEEL_INDICES:
            target[i] = 0.0
            continue
        leg = i // 4
        jtype = i % 4
        right = (leg % 2) != 0
        front = leg < 2

        if jtype == 0:  # hip
            sign = 1.0 if ((front and not right) or (not front and right)) else -1.0
            offset_deg = sign * kHipOffsetDeg
            target[i] = -offset_deg * kDeg2Rad
        elif jtype == 1:  # thigh
            sign = -1.0 if right else 1.0
            offset_deg = sign * kThighOffsetDeg
            target[i] = -offset_deg * kDeg2Rad
        elif jtype == 2:  # calf
            sign = -1.0 if right else 1.0
            offset_deg = sign * kCalfOffsetDeg
            target[i] = (offset_deg / kCalfRatio) * kDeg2Rad
    return target


def select_leg_interactive() -> str:
    """交互式选择要动的腿"""
    print()
    print("  请选择要移动到编码器零位的腿：")
    print()
    for i, (key, name) in enumerate(LEG_NAMES.items(), 1):
        joints = LEG_JOINTS[key]
        joint_names = " / ".join(MOTOR_LABELS[j] for j in joints)
        print(f"    {i}. {BOLD}{name}{NC}  ({key})  [{joint_names}]")
    print(f"    5. {BOLD}全部腿{NC}  (all)")
    print()
    while True:
        try:
            raw = input("  输入编号(1-5)或缩写(fl/fr/rl/rr/all): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return ""
        mapping = {"1": "fl", "2": "fr", "3": "rl", "4": "rr", "5": "all"}
        if raw in mapping:
            return mapping[raw]
        if raw in LEG_NAMES or raw == "all":
            return raw
        print(f"  无效输入，请输入 1-5 或 fl/fr/rl/rr/all")


def active_indices(leg_key: str) -> list[int]:
    """返回要位置控制的关节索引（不包括轮毂）"""
    if leg_key == "all":
        return list(LEG_INDICES)
    return list(LEG_JOINTS[leg_key])


def print_status(target_deg: np.ndarray, cur_deg: np.ndarray,
                 progress: float, phase: str, temps: np.ndarray,
                 active_idx: list[int], leg_key: str):
    """实时打印各关节角度和偏差表"""
    active_set = set(active_idx)
    leg_display = LEG_NAMES.get(leg_key, "全部腿") if leg_key != "all" else "全部腿"

    lines = []
    lines.append("=" * 76)
    lines.append(f"  电机编码器零位移动  |  {BOLD}{leg_display}{NC}  |  阶段: {phase}  |  {progress*100:5.1f}%")
    lines.append("=" * 76)
    lines.append(f"  {'关节':<22}{'当前(°)':>10}{'目标(°)':>10}{'偏差(°)':>10}{'温度':>8}  状态")
    lines.append("  " + "-" * 68)

    max_abs_err = 0.0
    for i in LEG_INDICES:
        is_active = i in active_set
        name = MOTOR_LABELS[i]
        c = cur_deg[i]
        t = target_deg[i]
        e = c - t
        abs_e = abs(e)
        if is_active:
            max_abs_err = max(max_abs_err, abs_e)
        temp = temps[i]

        name_pad = f"{name:<22}"
        if not is_active:
            name_s = f"{GRAY}{name_pad}{NC}"
            c_s = f"{GRAY}{c:>+10.2f}{NC}"
            t_s = f"{GRAY}{'—':>10}{NC}"
            e_s = f"{GRAY}{'阻尼':>10}{NC}"
            t_disp = f"{GRAY}{temp:>6.1f}°{NC}"
            status = f"{GRAY}阻尼{NC}"
        else:
            name_s = f"{BOLD}{name_pad}{NC}"
            c_s = f"{c:>+10.2f}"
            t_s = f"{t:>+10.2f}"
            if abs_e < 2.0:
                e_s = f"{GREEN}{e:>+10.2f}{NC}"
            elif abs_e < 10.0:
                e_s = f"{YELLOW}{e:>+10.2f}{NC}"
            else:
                e_s = f"{RED}{e:>+10.2f}{NC}"
            if temp > 70:
                t_disp = f"{RED}{temp:>6.1f}°{NC}"
            elif temp > 55:
                t_disp = f"{YELLOW}{temp:>6.1f}°{NC}"
            else:
                t_disp = f"{temp:>6.1f}°"
            status = f"{BOLD}控制{NC}"

        lines.append(f"  {name_s}{c_s}{t_s}  {e_s}  {t_disp}  {status}")

    for i in WHEEL_INDICES:
        name = MOTOR_LABELS[i]
        c = cur_deg[i]
        temp = temps[i]
        name_pad = f"{name:<22}"
        t_disp = f"{GRAY}{temp:>6.1f}°{NC}" if temp <= 55 else f"{YELLOW}{temp:>6.1f}°{NC}"
        lines.append(f"  {GRAY}{name_pad}{NC}{GRAY}{c:>+10.2f}{NC}{GRAY}{'—':>10}{NC}  {GRAY}{'—':>10}{NC}  {t_disp}  {GRAY}轮毂阻尼{NC}")

    lines.append("  " + "-" * 68)
    if leg_key == "all":
        scope = "全部腿"
    else:
        scope = LEG_NAMES[leg_key]
    if max_abs_err < 2.0 and progress >= 1.0:
        lines.append(f"  {GREEN}{scope} 最大偏差: {max_abs_err:.2f}°  (已到位){NC}")
    else:
        lines.append(f"  {scope} 最大偏差: {RED}{max_abs_err:.2f}°{NC}  |  Kp={KP} Kd={KD}  速度={np.degrees(MAX_SPEED):.0f}°/s")
    lines.append("=" * 76)
    lines.append("  Ctrl+C 中止")

    print(CLEAR + "\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description="移动到电机编码器零位")
    parser.add_argument("--leg", type=str, default=None,
                        choices=["fl", "fr", "rl", "rr", "all"],
                        help="选择要移动的腿: fl/fr/rl/rr/all (默认交互式选择)")
    parser.add_argument("--yes", action="store_true", help="跳过YES确认（脚本调用用）")
    args = parser.parse_args()

    running = True

    def sigint(*a):
        nonlocal running
        print("\n[中断]")
        running = False
    signal.signal(signal.SIGINT, sigint)

    # 选择腿
    leg_key = args.leg
    if leg_key is None:
        print("=" * 64)
        print("  目标：电机编码器原始值=0 （MIT协议电机输出轴零位）")
        print("=" * 64)
        leg_key = select_leg_interactive()
        if not leg_key:
            print("已取消")
            return

    active_idx = active_indices(leg_key)
    active_set = set(active_idx)
    other_leg_idx = [i for i in LEG_INDICES if i not in active_set]

    TARGET = build_target_positions()
    TARGET_DEG = np.degrees(TARGET)

    # 打印目标角度表（仅显示选中腿）
    print()
    print("=" * 64)
    print(f"  目标：电机编码器零位  |  {BOLD}{LEG_NAMES.get(leg_key, '全部腿') if leg_key != 'all' else '全部腿'}{NC}")
    print("=" * 64)
    print(f"  {'关节':<22}{'目标(°)':>10}")
    print("  " + "-" * 34)
    for i in (active_idx if leg_key != "all" else LEG_INDICES):
        print(f"  {MOTOR_LABELS[i]:<22}{TARGET_DEG[i]:>+10.2f}")
    if leg_key != "all":
        print("  " + "-" * 34)
        print(f"  其他腿关节: 保持阻尼（Kp=0, Kd={DAMPING_KD}）")
        print(f"  轮毂: 保持阻尼（Kp=0, Kd={WHEEL_KD}）")
    print("=" * 64)
    print()
    print(f"  {RED}{BOLD}⚠⚠⚠  警  告  ⚠⚠⚠{NC}")
    if leg_key == "all":
        print("  大腿/小腿需转动约160°！")
        print("  仅在【未安装腿连杆】或【机器人完全拆解】时使用！")
    else:
        print(f"  选中腿的大腿/小腿需转动约160°！")
        print(f"  请确保选中腿无任何阻挡/连接，其他腿保持阻尼。")
    print("  若腿已组装，请勿运行！")
    print()
    print(f"  Kp={KP}, Kd={KD}, 速度限制={np.degrees(MAX_SPEED):.0f}°/s")
    print("  Ctrl+C 可随时中止并阻尼")
    print()
    if args.yes:
        ans = "YES"
    else:
        try:
            ans = input("  确认安全，输入 YES 继续: ").strip()
        except (EOFError, KeyboardInterrupt):
            ans = ""
    if ans != "YES":
        print("已取消")
        return

    print("\n等待电机服务就绪...")
    motor = MotorClient("127.0.0.1", 55100)
    fb = motor.wait_until_ready(10.0, damping_rate_hz=50)
    print(f"  rx={fb.rx_rate_hz:.0f}Hz tx={fb.tx_rate_hz:.0f}Hz")
    motor.update_motor_states()

    start_pos = motor.positions.copy()
    # delta只对选中关节计算
    delta = np.zeros(MOTOR_COUNT, dtype=np.float32)
    for i in active_idx:
        delta[i] = TARGET[i] - start_pos[i]
    max_delta = float(np.max(np.abs(delta[active_idx]))) if active_idx else 0.0
    max_deg = np.degrees(max_delta)
    print(f"  最大需转动: {max_deg:.1f}°")

    # ---- 构建控制增益数组 ----
    # 运动增益：选中关节Kp/Kd，其他Kp=0+阻尼Kd
    kp_arr = np.zeros(MOTOR_COUNT, dtype=np.float32)
    kd_arr = np.full(MOTOR_COUNT, DAMPING_KD, dtype=np.float32)
    for i in active_idx:
        kp_arr[i] = KP
        kd_arr[i] = KD
    kd_arr[list(WHEEL_INDICES)] = WHEEL_KD

    # 保持增益
    hold_kp = np.zeros(MOTOR_COUNT, dtype=np.float32)
    hold_kd = np.full(MOTOR_COUNT, DAMPING_KD, dtype=np.float32)
    for i in active_idx:
        hold_kp[i] = HOLD_KP
        hold_kd[i] = HOLD_KD
    hold_kd[list(WHEEL_INDICES)] = WHEEL_KD

    # 阻尼增益（退出用）
    damp_kp = np.zeros(MOTOR_COUNT, dtype=np.float32)
    damp_kd = np.full(MOTOR_COUNT, DAMPING_KD, dtype=np.float32)
    damp_kd[list(WHEEL_INDICES)] = WHEEL_KD

    vel = np.zeros(MOTOR_COUNT, dtype=np.float32)
    torq = np.zeros(MOTOR_COUNT, dtype=np.float32)

    duration = max(3.0, max_delta / MAX_SPEED) if active_idx else 3.0
    t0 = time.monotonic()
    last_print = 0.0

    print(f"  预计运动时长: {duration:.1f}s")
    time.sleep(1.5)

    # ---- 运动阶段 ----
    arrived = False
    # 非选中关节：cmd = 当前位置（保持不动，阻尼）
    # 选中关节：从start_pos线性插值到TARGET
    static_pos = start_pos.copy()
    static_pos[list(WHEEL_INDICES)] = 0.0

    while running:
        t = time.monotonic()
        alpha = min(1.0, (t - t0) / duration)

        # 构建cmd_pos
        cmd_pos = static_pos.copy()
        for i in active_idx:
            if not arrived:
                cmd_pos[i] = start_pos[i] + (TARGET[i] - start_pos[i]) * alpha
            else:
                cmd_pos[i] = TARGET[i]

        if not arrived:
            kp_use = kp_arr
            kd_use = kd_arr
            phase = "运动中"
        else:
            kp_use = hold_kp
            kd_use = hold_kd
            phase = f"保持中(Kp={HOLD_KP})"

        cmd_pos[list(WHEEL_INDICES)] = 0.0
        motor.send_command(cmd_pos, vel, torq, kp_use, kd_use)
        motor.update_motor_states()

        fb = motor.last_feedback
        if fb and fb.safety_latched:
            print(f"\n[错误] 电机服务锁存: {fb.latch_reasons}")
            running = False
            break

        cur_deg = np.degrees(motor.positions)
        temps = motor.temperatures

        # 检查到位（仅选中关节）
        if active_idx:
            err_rad = np.abs(motor.positions[active_idx] - TARGET[active_idx])
            max_err = float(np.max(err_rad))
        else:
            max_err = 0.0

        if not arrived and alpha >= 1.0 and max_err < np.radians(3.0):
            arrived = True
            phase = f"到位(Kp={HOLD_KP})"

        # 周期性刷新显示
        now = time.monotonic()
        if now - last_print >= PRINT_INTERVAL:
            print_status(TARGET_DEG, cur_deg, alpha if not arrived else 1.0,
                         phase, temps, active_idx, leg_key)
            last_print = now

        elapsed = time.monotonic() - t
        if PERIOD - elapsed > 0:
            time.sleep(PERIOD - elapsed)

    # ---- 退出：全阻尼 ----
    print("\n发送阻尼退出...")
    z = np.zeros(MOTOR_COUNT, dtype=np.float32)
    for _ in range(30):
        motor.send_command(z, z, z, damp_kp, damp_kd)
        time.sleep(0.01)
    motor.close()
    print("完成")


if __name__ == "__main__":
    try:
        main()
    except MotorSafetyError as e:
        print(f"[错误] 电机安全: {e}")
    except TimeoutError as e:
        print(f"[错误] 超时: {e}")
    except OSError as e:
        print(f"[错误] 连接失败: {e}")
        print("请确认电机服务已启动 (sudo ./move_to_zero.sh --no-build)")
