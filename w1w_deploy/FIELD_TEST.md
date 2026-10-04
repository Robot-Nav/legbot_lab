# W1W staged field test

Nothing in the installer starts motion control. Run each stage separately and
inspect the result before continuing.

## Stage 0: physical preparation

- Support the chassis so all four wheels and feet can move freely.
- Keep an independent power disconnect within reach.
- Start with ELRS CH6 in damping, CH8 released and CH9 speed lock enabled.
- Confirm all CAN wiring and both CAN FD termination arrangements.

## Stage 1: copy and install, without starting services

Stop the old manually launched motor process only when ready to switch. Then:

```bash
cd /path/to/w1w_runtime
sudo apt-get install python3-venv python3-pip python3-yaml
sudo W1W_RUN_USER=kickpi ./scripts/install.sh
```

Expected result: a release under `/opt/w1w/releases`, systemd units installed,
and no W1W process started. Check:

```bash
systemctl is-active w1w-motor.service w1w-controller.service
readlink -f /opt/w1w/current
```

Both services should report `inactive`.

## Stage 2: set up and validate policy dependencies offline

```bash
sudo W1W_RUN_USER=kickpi /opt/w1w/current/scripts/setup_policy_env.sh
```

Expected result: all unit tests pass. This loads no motor service and sends no
CAN command.

## Stage 3: start safe base services

```bash
sudo w1wctl start-base
sudo w1wctl status
sudo w1wctl motor-health
sudo w1wctl elrs-health
```

Expected motor state is `READY`, all 16 motor ages are below 200 ms, joint
faults are zero, CAN error counters do not grow, and the controller remains
inactive. `motor-health` sends only the exact safe damping packet.

Open `http://192.168.0.218:8080/` and verify the four-leg layout, positions,
velocities, torque, temperature, wheel voltage, feedback ages and raw W190
status.

Monitor in separate terminals:

```bash
sudo w1wctl logs w1w-motor.service
watch -n 1 'sudo w1wctl can-status'
```

Do not continue if status is `WAITING`/`LATCHED`, a motor is offline, or CAN
error counters keep increasing.

## Stage 4: IMU and ELRS validation

```bash
sudo w1wctl health
```

Expected: motor `READY`, IMU `healthy=True`, ELRS `healthy=True`, and fresh data.
Move the supported chassis by hand and confirm IMU values update before policy
control.

Use `w1wctl elrs-health` before starting the controller and verify all switch
directions. The expected mapping is CH6 damping/hold/RL, CH7 low/medium/high,
CH8 active-low emergency, and CH9 high speed lock. Do not continue until the
actual transmitter values match. Configure receiver failsafe as CH6=damping,
CH8=emergency and CH9=speed lock, then turn the transmitter off and verify the
safe values or stale state are reported.

## Stage 5: launch controller in damping only

Keep ELRS at CH6=damping, CH8 released and CH9=speed lock. Then:

```bash
sudo w1wctl controller-start I_HAVE_SUPPORTED_THE_ROBOT
sudo w1wctl logs w1w-controller.service
```

Expected log sequence ends at `DAMPING`. No standing or RL command is sent
until CH6 explicitly transitions from damping to hold.

## Stage 6: standing and RL, one transition at a time

1. Move CH6 from damping to hold and observe the crouch-to-stand move.
2. Hold with zero velocity command and inspect all feedback.
3. Enter RL only after the standing pose, signs and IMU orientation are correct.

Any ELRS/IMU timeout, emergency stop, joint fault, over-temperature, policy
NaN/Inf, controller delay over 80 ms, motor timeout or feedback loss permanently
latches damping for that run. Signal recovery never resumes motion.

After an active controller is stopped, the motor service is expected to latch
within 100 ms. To clear it, support the robot and explicitly run:

```bash
sudo w1wctl motor-rearm I_HAVE_SUPPORTED_THE_ROBOT
```
