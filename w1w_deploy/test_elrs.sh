#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source venv/bin/activate
python3 -c "
import socket, struct, time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('127.0.0.1', 0))
sock.settimeout(0.5)

def send_heartbeat():
    # 持续发送心跳命令包（保持客户端注册）
    command = struct.pack('<IHHII', 0x454C5243, 1, 0, int(time.time()*1000) & 0xFFFFFFFF, 0)
    sock.sendto(command, ('127.0.0.1', 55201))

print('ELRS通道监视器启动，拨动摇杆/开关看数据变化，按Ctrl+C退出\n')
last_hb = 0

def norm(ch):
    # CRSF实际范围是172~1811，中心991.5
    return max(-1.0, min(1.0, (ch - 992) / 820.0 * 2 - 1))

try:
    while True:
        now = time.monotonic()
        if now - last_hb > 0.1:  # 每100ms发一次心跳
            send_heartbeat()
            last_hb = now
        
        try:
            data, _ = sock.recvfrom(256)
        except socket.timeout:
            continue
            
        if len(data) < 70: continue
        vals = struct.unpack('<IHHIIII HHHHHHHHHHHHHHHH BB fff', data)
        if vals[0] != 0x454C5246: continue
        
        ch = vals[7:23]
        age = vals[25]
        hz = vals[26]
        stale = (vals[6] & 1) != 0
        
        print('\033[2J\033[H', end='')
        print(f'=== W1W ELRS通道监视器 ===  采样率: {hz:.0f}Hz  延迟: {age:.1f}ms  {'✅信号正常' if not stale else '⚠️  信号丢失'}')
        print()
        print(f'通道    CH1(转向)  CH2(未用)  CH3(前进)  CH4(侧移)')
        print(f'原始值   {ch[0]:<10}{ch[1]:<10}{ch[2]:<10}{ch[3]}')
        print(f'归一化   {norm(ch[0]):<+10.3f}{norm(ch[1]):<+10.3f}{norm(ch[2]):<+10.3f}{norm(ch[3]):<+.3f}')
        print()
        print(f'通道    CH5(SA)   CH6(SB模式) CH7(速度) CH8(SH急停)')
        sw = lambda c: 'HIGH' if c > 1500 else 'LOW'
        mode_sb = 0 if ch[5] < 1300 else (1 if ch[5] < 1700 else 2)
        speed_sc = 0 if ch[6] < 1300 else (1 if ch[6] < 1700 else 2)
        mode_map = {0: '上:阻尼', 1: '中:站立', 2: '下:行走'}
        speed_map = {0: '低速', 1: '中速', 2: '高速'}
        print(f'开关值  {sw(ch[4]):<10}{mode_map[mode_sb]:<12}{speed_map[speed_sc]:<10}{'急停!' if ch[7]>1500 else '正常'}')
        print()
        print('正确启动顺序：SB上(阻尼) → SH下(解除急停) → SB中(站立) → SB下(行走)')
        time.sleep(0.03)
except KeyboardInterrupt:
    print('\n退出')
"

