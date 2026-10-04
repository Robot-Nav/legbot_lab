# W1W wheel-legged robot deployment

完整中文说明请先阅读：[部署与测试说明.md](部署与测试说明.md)。

This directory is a standalone deployment package for one KickPi controlling
12 RobStride joints and four W190 wheels.

Use [FIELD_TEST.md](FIELD_TEST.md) for the staged first-run procedure. The
installer and this document never require starting the controller immediately.

Create the transfer archive locally with:

```bash
python3 scripts/package_release.py --output ../w1w_runtime.tar.gz
```

## Motor layout

| Index | Device | Bus / ID |
|---:|---|---|
| 0, 1, 2 | front-left hip, thigh, calf | can0 / 4, 3, 2 |
| 3 | front-left wheel | can4 / 1 |
| 4, 5, 6 | front-right hip, thigh, calf | can1 / 4, 3, 2 |
| 7 | front-right wheel | can4 / 2 |
| 8, 9, 10 | rear-left hip, thigh, calf | can2 / 4, 3, 2 |
| 11 | rear-left wheel | can5 / 1 |
| 12, 13, 14 | rear-right hip, thigh, calf | can3 / 4, 3, 2 |
| 15 | rear-right wheel | can5 / 2 |

`can0` through `can3` use classic CAN at 1 Mbps. `can4` and `can5` use
CAN FD at 1 Mbps arbitration and 5 Mbps data rate. Wheel signs pass through
unchanged. Joint and wheel command loops run at approximately 700 Hz
(`1429 us`) in the current validated configuration.

## Safety behavior

- The motor service starts in damping: joint `Kd=2`, wheel `Kd=1`.
- It requires all 16 motors online before accepting active control.
- A 100 ms active-command timeout or feedback loss latches damping until an
  explicit motor-service restart.
- The policy controller also latches damping for stale IMU/ELRS data,
  emergency stop, over-temperature, invalid observations or invalid policy
  output. It never automatically resumes.
- `w1w-motor.service` and `w1w-controller.service` both use `Restart=no`.
- The web service is read-only and sends only the exact safe damping packet.
- Installing the package does not enable or start the policy controller.

## Install on KickPi

Runtime requirements are Python 3, NumPy, PyYAML and ONNX Runtime. The installer checks these before it
changes the active release.

On Ubuntu 24.04, install the policy environment prerequisites first:

```bash
sudo apt-get install python3-venv python3-pip python3-yaml
```

```bash
cd w1w_runtime
sudo W1W_RUN_USER=kickpi ./scripts/install.sh
sudo W1W_RUN_USER=kickpi /opt/w1w/current/scripts/setup_policy_env.sh
sudo w1wctl start-base
sudo w1wctl health
```

The installer deliberately leaves every service stopped and never enables the
policy controller. On a first install, safe base services are enabled only by
the explicit `w1wctl start-base` command. During an upgrade, existing base
enablement is preserved, but the installer still refuses to switch releases
while any W1W process is running. The safe base services are
CAN setup, motor damping, IMU, ELRS and web monitoring. Open
`http://192.168.0.218:8080/` to inspect all motor feedback before policy tests.

Policy control requires the robot to be securely supported:

```bash
sudo w1wctl controller-start I_HAVE_SUPPORTED_THE_ROBOT
```

Stopping an active controller causes the motor watchdog to latch. Rearming is
therefore explicit and also requires support confirmation:

```bash
sudo w1wctl controller-stop
sudo w1wctl motor-rearm I_HAVE_SUPPORTED_THE_ROBOT
```

Useful commands:

```bash
sudo w1wctl status
sudo w1wctl can-status
sudo w1wctl elrs-health
sudo w1wctl logs w1w-motor.service
sudo w1wctl logs w1w-controller.service
```

Configuration is preserved at `/etc/w1w/controller.yaml`. Releases are stored
under `/opt/w1w/releases`; `/opt/w1w/current` is switched atomically and
`/opt/w1w/previous` is retained for rollback.
