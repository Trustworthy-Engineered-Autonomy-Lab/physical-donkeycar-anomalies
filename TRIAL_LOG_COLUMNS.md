# Trial Log File Columns

This repository writes trial results under `trials/<anomaly_type>/<session>/`.
The main files are:

- `runs/<run_id>/log_<run_id>.csv`: detailed per-frame telemetry for one run.
- `runs/summary.csv`: one summary row per run in the session.
- `data_<env>_<noise>_<name>/cte_values.csv`: older/simple CTE-only log written by the simulator wrapper.

## Detailed run log

The detailed run log is produced by `donkeycar/donkeycar/parts/run_logger.py`.
Each row represents one vehicle loop/frame.

| Column | Meaning |
| --- | --- |
| `frame_id` | Zero-based frame counter within this run. Increments once per logged row. |
| `uncorrupted_image_path` | Intended path for the clean camera image for this run, formatted as `imgs/normal/image_<run_id>.jpg`. This is a path label written by the logger. |
| `corrupted_image_path` | Intended path for the anomaly/noise camera image for this run, formatted as `imgs/noise/noise_image_<run_id>.jpg`. This is a path label written by the logger. |
| `timestamp_ms` | Wall-clock time when the row was logged, in Unix epoch milliseconds from the host machine. |
| `sim_time` | Simulator-provided time value from `info["sim_time"]`. Blank when the simulator does not provide it. |
| `run_sim_time` | Elapsed simulator time since the first non-blank `sim_time` in this run. If `sim_time` is blank, this is `0.0`. |
| `steering_cmd` | Raw pilot/model steering command from `pilot/angle`, before `DriveMode` applies steering gain or bias. Typically normalized around `-1.0` to `1.0`, with sign following DonkeyCar/simulator steering convention. |
| `steering_act` | Actual steering value sent to the drivetrain/simulator after drive-mode processing. This includes configured steering gain and steering bias/offset. |
| `control/pilot_angle` | Pilot/model steering command recorded on the `control/pilot_angle` channel. In the current pipeline this mirrors the model command used for control logging. |
| `control/pilot_throttle` | Pilot/model throttle command recorded on the `control/pilot_throttle` channel. |
| `throttle_cmd` | Raw pilot/model throttle command from `pilot/throttle`, before final drive-mode scaling. |
| `throttle_act` | Actual throttle value sent to the drivetrain/simulator after drive-mode processing. |
| `steering_delayed` | Steering value currently being applied inside the simulator wrapper after command latency handling. With no command latency this matches the latest steering action; with latency it is an older queued command. |
| `throttle_delayed` | Throttle value currently being applied inside the simulator wrapper after command latency handling. With no command latency this matches the latest throttle action; with latency it is an older queued command. |
| `pos_x` | Simulator car position X coordinate in the Unity world coordinate system. |
| `pos_z` | Simulator car position Z coordinate in the Unity world coordinate system. The logger uses X/Z movement to compute traveled distance. |
| `yaw_rate` | Y-axis gyro value from simulator telemetry, used here as yaw rate. |
| `speed` | Simulator-reported vehicle speed. |
| `cte` | Cross-track error: signed lateral distance from the track center/path. Larger absolute values mean the car is farther from the intended path. |
| `accel_x` | Simulator accelerometer X value. |
| `accel_z` | Simulator accelerometer Z value. |
| `yaw` | Car yaw angle from simulator orientation telemetry, in degrees. |
| `pitch` | Car pitch angle from simulator orientation telemetry, in degrees. |
| `roll` | Car roll angle from simulator orientation telemetry, in degrees. |
| `anomaly_param` | Set-like string naming the anomaly type(s) active for this run, for example `{cam_pitch}` or `{noise}`. `{}` means no anomaly type was recorded. |
| `anomaly_intensity` | Set-like string containing the intensity value(s) corresponding to `anomaly_param`, for example `{-15.0}`. `{}` means no matching intensity was recorded. |
| `crashed` | Crash marker for this row. Values are `0` for normal rows and `1` on the last row if the run outcome is `CRASH`. A crash is recorded after sustained simulator contact lasts at least the configured timeout. |

## Summary log

`runs/summary.csv` is appended once per run when the `RunLogger` shuts down.

| Column | Meaning |
| --- | --- |
| `run_id` | Trial loop index for the run. This also appears in the detailed log filename. |
| `start_pos` | Simulator start-position index selected for the run. The trial runner chooses from 0 to 303 and avoids reusing positions within the same session until all are exhausted. |
| `anomaly_param` | Set-like string naming the anomaly type(s) active for the run. |
| `outcome` | Run result. Currently `SAFE` unless the simulator wrapper marks the run as `CRASH` after sustained contact. |
| `total_distance` | Approximate path length traveled in the X/Z plane, computed by summing Euclidean distance between consecutive logged positions. Rounded to three decimals. |
| `time_to_failure` | Number of logged frames in the run. For a crashed run, this is the frame count before shutdown; for a safe run, it is the total frame count. Despite the name, it is a frame count, not seconds. |
| `cumulative_cte` | Sum of absolute `cte` over all logged frames. Rounded to three decimals. |
| `avg_cte` | Mean absolute `cte`, computed as `cumulative_cte / frame_count`. Rounded to five decimals. |

## CTE-only log

`data_<env>_<noise>_<name>/cte_values.csv` is written directly by the simulator wrapper thread.

| Column | Meaning |
| --- | --- |
| `Time Step` | Zero-based simulator wrapper step counter. It increments once per environment step. |
| `CTE` | Cross-track error at that step, copied from simulator telemetry. |

## Notes

- Position and orientation values come from Unity simulator telemetry. Unity uses X/Y/Z coordinates; this project logs X and Z for ground-plane motion.
- `hit`/`contact` are used internally to decide whether the run crashed, but the detailed CSV stores only the final `crashed` marker.
- The detailed image path columns are generated labels. They do not guarantee that a corresponding image file exists unless the image-saving code for that run also wrote those files.
