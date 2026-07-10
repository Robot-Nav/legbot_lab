// 灵足 RS02 USB-CAN 串口帧 Web 编解码库：45 54 头、0D 0A 尾、type1/type2/type3/type4。

const HEADER = [0x45, 0x54];
const TAIL = [0x0d, 0x0a];
const STANDARD_FRAME = 0x01;  // 标准运控帧（主机下发）
const EXTENDED_FRAME = 0x02;  // 扩展反馈帧

// 各物理量 16 位无符号定点量化范围。
const RANGES = {
  q: [-12.5663706144, 12.5663706144],
  dq: [-44.0, 44.0],
  kp: [0.0, 500.0],
  kd: [0.0, 5.0],
  tau: [-17.0, 17.0],
};

// 将任意十六进制字符串解析为字节数组（支持 0x 前缀、空格、逗号等分隔符）。
export function parseHex(input) {
  const cleaned = input.replace(/0x/gi, " ").replace(/[^0-9a-fA-F]/g, " ").trim();
  if (!cleaned) return [];
  const parts = cleaned.split(/\s+/);
  return parts.map((part) => {
    if (part.length > 2) throw new Error(`字节过长: ${part}`);
    const value = Number.parseInt(part, 16);
    if (!Number.isFinite(value) || value < 0 || value > 0xff) throw new Error(`非法字节: ${part}`);
    return value;
  });
}

// 字节数组 -> "xx xx" 十六进制字符串。
export function toHex(bytes) {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join(" ");
}

// 浮点值按 [min, max] 16 位无符号定点量化。
export function floatToUint(value, min, max) {
  const clamped = Math.max(min, Math.min(max, Number(value)));
  const levels = 0xffff;
  return Math.round(((clamped - min) * levels) / (max - min)) & 0xffff;
}

// 16 位无符号定点 -> 浮点值。
export function uintToFloat(raw, min, max) {
  const value = Math.max(0, Math.min(0xffff, raw));
  return min + ((max - min) * value) / 0xffff;
}

// 大端 16 位读取。
function u16be(bytes, offset) {
  return ((bytes[offset] << 8) | bytes[offset + 1]) & 0xffff;
}

// 大端 16 位写入。
function pushU16BE(out, value) {
  out.push((value >> 8) & 0xff, value & 0xff);
}

// 从帧中提取电机 CAN ID：标准帧取 canId，扩展帧取 idField 低 8 位。
export function motorIdFromFrame(frame) {
  if (frame.frameType === STANDARD_FRAME) return frame.canId;
  if (frame.frameType === EXTENDED_FRAME) return frame.idField & 0xff;
  throw new Error(`不支持的 frame_type 0x${frame.frameType.toString(16)}`);
}

// 解析完整串口帧：校验头、尾、dlc，并按帧类型拆分字段。
export function parseSerialFrame(input) {
  const bytes = typeof input === "string" ? parseHex(input) : Array.from(input);
  if (bytes.length < 10) throw new Error("帧太短");
  if (bytes[0] !== HEADER[0] || bytes[1] !== HEADER[1]) throw new Error("帧头错误，应为 45 54");

  const channel = bytes[2];
  const frameType = bytes[3];
  const idField = u16be(bytes, 4);
  const canOrMaster = bytes[6];
  const dlc = bytes[7];
  if (dlc > 8) throw new Error(`DLC 过大: ${dlc}`);

  // 头(2) + channel(1) + frame_type(1) + id_field(2) + can/master(1) + dlc(1) + data(dlc) + 尾(2)
  const expectedLength = 2 + 1 + 1 + 2 + 1 + 1 + dlc + 2;
  if (bytes.length !== expectedLength) {
    throw new Error(`长度不匹配: 实际 ${bytes.length}, 期望 ${expectedLength}`);
  }
  if (bytes[expectedLength - 2] !== TAIL[0] || bytes[expectedLength - 1] !== TAIL[1]) {
    throw new Error("帧尾错误，应为 0d 0a");
  }

  const data = bytes.slice(8, 8 + dlc);
  const frame = {
    bytes,
    channel,
    frameType,
    idField,
    dlc,
    data,
  };
  // 标准帧：idField 为 tau 量化值，canOrMaster 为 CAN ID。
  if (frameType === STANDARD_FRAME) {
    frame.torqueRaw = idField;
    frame.canId = canOrMaster;
    // 扩展帧：idField 低 8 位为电机 ID，canOrMaster 为 master ID。
  } else if (frameType === EXTENDED_FRAME) {
    frame.motorId = idField & 0xff;
    frame.masterId = canOrMaster;
    // type3/type4 模式帧：idField 为 master ID，canOrMaster 为目标电机 ID。
  } else if (frameType === 0x03 || frameType === 0x04) {
    frame.masterId = idField;
    frame.motorId = canOrMaster;
  }
  return frame;
}

// 打包完整串口帧。
export function encodeFrame({ channel, frameType, idField, canOrMaster, data }) {
  const payload = Array.from(data || []);
  if (payload.length > 8) throw new Error("数据区长度必须 <= 8");
  const out = [...HEADER, channel & 0xff, frameType & 0xff];
  pushU16BE(out, idField & 0xffff);
  out.push(canOrMaster & 0xff, payload.length & 0xff, ...payload, ...TAIL);
  return out;
}

// 打包 type1 数据区：q, dq, kp, kd（每个 2 字节大端）。
export function encodeType1Data({ q, dq, kp, kd }) {
  const out = [];
  pushU16BE(out, floatToUint(q, ...RANGES.q));
  pushU16BE(out, floatToUint(dq, ...RANGES.dq));
  pushU16BE(out, floatToUint(kp, ...RANGES.kp));
  pushU16BE(out, floatToUint(kd, ...RANGES.kd));
  return out;
}

// 打包标准 type1 帧（主机下发）：tau 嵌入 idField，motorId 作为 CAN ID。
export function encodeStandardType1Frame({ channel, motorId, q, dq, kp, kd, tau }) {
  return encodeFrame({
    channel,
    frameType: STANDARD_FRAME,
    idField: floatToUint(tau, ...RANGES.tau),
    canOrMaster: motorId,
    data: encodeType1Data({ q, dq, kp, kd }),
  });
}

// 打包 type3 使能 / type4 失能/清故障帧。
export function encodeModeFrame({ channel, mode, masterId, motorId, clearFault = false }) {
  if (mode !== 3 && mode !== 4) throw new Error("模式帧仅支持通信类型 3 或 4");
  const data = new Array(8).fill(0);
  if (mode === 4 && clearFault) data[0] = 1;
  return encodeFrame({
    channel,
    frameType: mode,
    idField: masterId,
    canOrMaster: motorId,
    data,
  });
}

// 解析 type3/type4 模式帧。
export function decodeModeFrame(frame) {
  if (frame.frameType !== 3 && frame.frameType !== 4) {
    throw new Error("不是 type3/type4 模式帧");
  }
  if (frame.data.length !== 8) throw new Error("模式帧需要 8 字节数据区");
  return {
    channel: frame.channel,
    mode: frame.frameType,
    masterId: frame.idField,
    motorId: frame.motorId,
    clearFault: frame.frameType === 4 && frame.data[0] === 1,
    data: frame.data,
  };
}

// 解析 type1 运控帧。
export function decodeType1Command(frame) {
  if (frame.data.length !== 8) throw new Error("type1 指令需要 8 字节数据区");
  return {
    motorId: motorIdFromFrame(frame),
    q: uintToFloat(u16be(frame.data, 0), ...RANGES.q),
    dq: uintToFloat(u16be(frame.data, 2), ...RANGES.dq),
    kp: uintToFloat(u16be(frame.data, 4), ...RANGES.kp),
    kd: uintToFloat(u16be(frame.data, 6), ...RANGES.kd),
    tau: frame.frameType === STANDARD_FRAME ? uintToFloat(frame.idField, ...RANGES.tau) : 0,
  };
}

// 解析 type2 反馈帧。
export function decodeType2Feedback(frame) {
  if (frame.data.length !== 8) throw new Error("type2 反馈需要 8 字节数据区");
  return {
    motorId: motorIdFromFrame(frame),
    q: uintToFloat(u16be(frame.data, 0), ...RANGES.q),
    dq: uintToFloat(u16be(frame.data, 2), ...RANGES.dq),
    tau: uintToFloat(u16be(frame.data, 4), ...RANGES.tau),
    tempC: u16be(frame.data, 6) / 10.0,
  };
}

export const constants = {
  HEADER,
  TAIL,
  STANDARD_FRAME,
  EXTENDED_FRAME,
  RANGES,
};
