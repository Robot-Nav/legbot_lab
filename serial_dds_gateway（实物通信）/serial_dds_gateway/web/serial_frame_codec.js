const HEADER = [0x45, 0x54];
const TAIL = [0x0d, 0x0a];
const STANDARD_FRAME = 0x01;
const EXTENDED_FRAME = 0x02;

const RANGES = {
  q: [-12.5663706144, 12.5663706144],
  dq: [-44.0, 44.0],
  kp: [0.0, 500.0],
  kd: [0.0, 5.0],
  tau: [-17.0, 17.0],
};

export function parseHex(input) {
  const cleaned = input.replace(/0x/gi, " ").replace(/[^0-9a-fA-F]/g, " ").trim();
  if (!cleaned) return [];
  const parts = cleaned.split(/\s+/);
  return parts.map((part) => {
    if (part.length > 2) throw new Error(`Byte too long: ${part}`);
    const value = Number.parseInt(part, 16);
    if (!Number.isFinite(value) || value < 0 || value > 0xff) throw new Error(`Invalid byte: ${part}`);
    return value;
  });
}

export function toHex(bytes) {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join(" ");
}

export function floatToUint(value, min, max) {
  const clamped = Math.max(min, Math.min(max, Number(value)));
  const levels = 0xffff;
  return Math.round(((clamped - min) * levels) / (max - min)) & 0xffff;
}

export function uintToFloat(raw, min, max) {
  const value = Math.max(0, Math.min(0xffff, raw));
  return min + ((max - min) * value) / 0xffff;
}

function u16be(bytes, offset) {
  return ((bytes[offset] << 8) | bytes[offset + 1]) & 0xffff;
}

function pushU16BE(out, value) {
  out.push((value >> 8) & 0xff, value & 0xff);
}

export function motorIdFromFrame(frame) {
  if (frame.frameType === STANDARD_FRAME) return frame.canId;
  if (frame.frameType === EXTENDED_FRAME) return frame.idField & 0xff;
  throw new Error(`Unsupported frame_type 0x${frame.frameType.toString(16)}`);
}

export function parseSerialFrame(input) {
  const bytes = typeof input === "string" ? parseHex(input) : Array.from(input);
  if (bytes.length < 10) throw new Error("Frame too short");
  if (bytes[0] !== HEADER[0] || bytes[1] !== HEADER[1]) throw new Error("Bad header, expected 45 54");

  const channel = bytes[2];
  const frameType = bytes[3];
  const idField = u16be(bytes, 4);
  const canOrMaster = bytes[6];
  const dlc = bytes[7];
  if (dlc > 8) throw new Error(`DLC too large: ${dlc}`);

  const expectedLength = 2 + 1 + 1 + 2 + 1 + 1 + dlc + 2;
  if (bytes.length !== expectedLength) {
    throw new Error(`Length mismatch: got ${bytes.length}, expected ${expectedLength}`);
  }
  if (bytes[expectedLength - 2] !== TAIL[0] || bytes[expectedLength - 1] !== TAIL[1]) {
    throw new Error("Bad tail, expected 0d 0a");
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
  if (frameType === STANDARD_FRAME) {
    frame.torqueRaw = idField;
    frame.canId = canOrMaster;
  } else if (frameType === EXTENDED_FRAME) {
    frame.motorId = idField & 0xff;
    frame.masterId = canOrMaster;
  } else if (frameType === 0x03 || frameType === 0x04) {
    frame.masterId = idField;
    frame.motorId = canOrMaster;
  }
  return frame;
}

export function encodeFrame({ channel, frameType, idField, canOrMaster, data }) {
  const payload = Array.from(data || []);
  if (payload.length > 8) throw new Error("data length must be <= 8");
  const out = [...HEADER, channel & 0xff, frameType & 0xff];
  pushU16BE(out, idField & 0xffff);
  out.push(canOrMaster & 0xff, payload.length & 0xff, ...payload, ...TAIL);
  return out;
}

export function encodeType1Data({ q, dq, kp, kd }) {
  const out = [];
  pushU16BE(out, floatToUint(q, ...RANGES.q));
  pushU16BE(out, floatToUint(dq, ...RANGES.dq));
  pushU16BE(out, floatToUint(kp, ...RANGES.kp));
  pushU16BE(out, floatToUint(kd, ...RANGES.kd));
  return out;
}

export function encodeStandardType1Frame({ channel, motorId, q, dq, kp, kd, tau }) {
  return encodeFrame({
    channel,
    frameType: STANDARD_FRAME,
    idField: floatToUint(tau, ...RANGES.tau),
    canOrMaster: motorId,
    data: encodeType1Data({ q, dq, kp, kd }),
  });
}

export function encodeModeFrame({ channel, mode, masterId, motorId, clearFault = false }) {
  if (mode !== 3 && mode !== 4) throw new Error("mode frame supports only communication type 3 or 4");
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

export function decodeModeFrame(frame) {
  if (frame.frameType !== 3 && frame.frameType !== 4) {
    throw new Error("not a type3/type4 mode frame");
  }
  if (frame.data.length !== 8) throw new Error("mode frame requires 8 data bytes");
  return {
    channel: frame.channel,
    mode: frame.frameType,
    masterId: frame.idField,
    motorId: frame.motorId,
    clearFault: frame.frameType === 4 && frame.data[0] === 1,
    data: frame.data,
  };
}

export function decodeType1Command(frame) {
  if (frame.data.length !== 8) throw new Error("type1 command requires 8 data bytes");
  return {
    motorId: motorIdFromFrame(frame),
    q: uintToFloat(u16be(frame.data, 0), ...RANGES.q),
    dq: uintToFloat(u16be(frame.data, 2), ...RANGES.dq),
    kp: uintToFloat(u16be(frame.data, 4), ...RANGES.kp),
    kd: uintToFloat(u16be(frame.data, 6), ...RANGES.kd),
    tau: frame.frameType === STANDARD_FRAME ? uintToFloat(frame.idField, ...RANGES.tau) : 0,
  };
}

export function decodeType2Feedback(frame) {
  if (frame.data.length !== 8) throw new Error("type2 feedback requires 8 data bytes");
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
