import {
  decodeType1Command,
  decodeType2Feedback,
  decodeModeFrame,
  encodeModeFrame,
  encodeStandardType1Frame,
  parseSerialFrame,
  toHex,
} from "./serial_frame_codec.js";

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function close(actual, expected, eps, message) {
  assert(Math.abs(actual - expected) <= eps, `${message}: got ${actual}, expected ${expected}`);
}

const standardZero = "45 54 00 01 00 00 20 08 00 00 00 00 00 00 00 00 0d 0a";
const standard = parseSerialFrame(standardZero);
assert(standard.channel === 0, "standard channel");
assert(standard.frameType === 0x01, "standard frame type");
assert(standard.idField === 0x0000, "standard torque field");
assert(standard.canId === 0x20, "standard CAN ID");
assert(standard.data.length === 8, "standard DLC");

const standardCommand = decodeType1Command(standard);
assert(standardCommand.motorId === 0x20, "standard type1 motor ID");
close(standardCommand.tau, -17.0, 0.001, "standard torque");

const encodedStandard = encodeStandardType1Frame({
  channel: 0,
  motorId: 0x20,
  q: standardCommand.q,
  dq: standardCommand.dq,
  kp: standardCommand.kp,
  kd: standardCommand.kd,
  tau: standardCommand.tau,
});
assert(toHex(encodedStandard) === toHex(standard.bytes), "standard re-encode");

const extendedFeedback = "45 54 01 02 00 20 fd 08 a3 5b 7f ac 7f ff 01 22 0d 0a";
const extended = parseSerialFrame(extendedFeedback);
assert(extended.channel === 1, "extended channel");
assert(extended.frameType === 0x02, "extended frame type");
assert(extended.idField === 0x0020, "extended ID field");
assert(extended.masterId === 0xfd, "extended master ID");

const feedback = decodeType2Feedback(extended);
assert(feedback.motorId === 0x20, "feedback motor ID");
close(feedback.q, 3.4713, 0.001, "feedback q");
close(feedback.dq, -0.1121, 0.001, "feedback dq");
close(feedback.tau, -0.0003, 0.001, "feedback tau");
close(feedback.tempC, 29.0, 0.01, "feedback temp");

const enableHex = "45 54 00 03 00 fd 7f 08 00 00 00 00 00 00 00 00 0d 0a";
const enable = parseSerialFrame(enableHex);
const enableMode = decodeModeFrame(enable);
assert(enableMode.mode === 3, "enable mode");
assert(enableMode.masterId === 0xfd, "enable master ID");
assert(enableMode.motorId === 0x7f, "enable motor ID");
assert(enableMode.clearFault === false, "enable clear fault");
assert(toHex(encodeModeFrame({ channel: 0, mode: 3, masterId: 0xfd, motorId: 0x7f })) === toHex(enable.bytes),
  "enable re-encode");

const clearFaultHex = "45 54 00 04 00 fd 7f 08 01 00 00 00 00 00 00 00 0d 0a";
const clearFault = parseSerialFrame(clearFaultHex);
const clearFaultMode = decodeModeFrame(clearFault);
assert(clearFaultMode.mode === 4, "clear fault mode");
assert(clearFaultMode.clearFault === true, "clear fault flag");
assert(
  toHex(encodeModeFrame({ channel: 0, mode: 4, masterId: 0xfd, motorId: 0x7f, clearFault: true })) ===
    toHex(clearFault.bytes),
  "clear fault re-encode",
);

console.log("serial_frame_codec tests passed");
