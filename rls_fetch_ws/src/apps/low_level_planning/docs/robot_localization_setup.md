# Robot Localization Setup for Fetch Robot

This document explains the robot_localization package integration for improving odometry accuracy through sensor fusion.

## Overview

The robot_localization package provides state estimation nodes that take in multiple sensor inputs and produces filtered state estimates. We use the Extended Kalman Filter (EKF) node to fuse wheel odometry and IMU data for better localization.

## Benefits

1. **Improved Accuracy**: Fuses multiple sensors to reduce drift and improve position estimates
2. **IMU Integration**: Uses IMU data (when available) to correct for wheel slip and improve orientation estimates
3. **Better Navigation**: Provides more reliable odometry for AMCL and move_base
4. **Reduced Drift**: EKF corrects for accumulated errors in wheel odometry

## Architecture

```
Raw Sensors          Robot Localization         Navigation Stack
============         ===================        ================

/odom        ─┐
              ├─> EKF Node ──> /odometry/filtered ──> AMCL ──> /map
/imu/data    ─┘                                              └─> move_base
```

## Configuration Files

### For Real Robot (with IMU)

- **Config**: `config/robot_localization_ekf.yaml`
- **Features**: Fuses wheel odometry + IMU data
- **Sensors**:
  - Odometry: x, y, yaw, linear velocities, angular velocity
  - IMU: angular velocity (yaw rate), linear acceleration

### For Gazebo Simulation (no IMU)

- **Config**: `config/robot_localization_ekf_gazebo.yaml`
- **Features**: Processes only wheel odometry with filtering
- **Sensors**:
  - Odometry: x, y, yaw, linear velocities, angular velocity

## Launch Files

### Gazebo Simulation

```bash
roslaunch low_level_planning fetch_control_gazebo.launch
```

- Uses `robot_localization_ekf_gazebo.yaml` (no IMU)
- Sets `use_sim_time: true`

### Real Robot

```bash
roslaunch low_level_planning fetch_control_drivers.launch
```

- Uses `robot_localization_ekf.yaml` (with IMU)
- Sets `use_sim_time: false`

## Topics

### Input Topics

- `/odom` - Raw wheel odometry from robot
- `/imu/data` - IMU data (real robot only)

### Output Topics

- `/odometry/filtered` - Filtered odometry estimate (used by navigation stack)
- `/accel/filtered` - Filtered acceleration estimate

## Key Parameters

### Sensor Configuration

```yaml
# Odometry configuration [x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw, ax, ay, az]
odom0_config: [true,  true,  false,
               false, false, true,
               true,  true,  false,
               false, false, true,
               false, false, false]

# IMU configuration (when available)
imu0_config: [false, false, false,
              false, false, false, 
              false, false, false,
              false, false, true,    # vyaw (angular velocity)
              true,  true,  false]   # ax, ay (linear acceleration)
```

### Process Noise Covariance

- Controls how much the filter trusts predictions vs measurements
- Lower values = trust process model more
- Higher values = trust measurements more

## Troubleshooting

### Common Issues

1. **No filtered odometry output**
   - Check that raw `/odom` topic is publishing
   - Verify EKF node is running: `rosnode list | grep ekf`
   - Check diagnostics: `rostopic echo /diagnostics`

2. **Poor performance**
   - Tune process noise covariance matrix
   - Check sensor data quality
   - Verify frame_id consistency

3. **IMU not working**
   - Verify `/imu/data` topic exists and publishes
   - Check IMU frame transforms
   - Ensure IMU data follows REP-103 (ENU frame)

### Debugging Commands

```bash
# Check EKF status
rostopic echo /diagnostics

# Monitor filtered odometry
rostopic echo /odometry/filtered

# Visualize in RViz
# Add Odometry displays for both /odom and /odometry/filtered

# Check TF tree
rosrun tf view_frames
```

## Tuning Guidelines

### For Better Performance

1. **Increase process noise** if the filter is too slow to respond to changes
2. **Decrease process noise** if the estimates are too noisy
3. **Adjust sensor timeout** based on sensor publishing rates
4. **Tune rejection thresholds** to filter out sensor outliers

### IMU Specific Tuning

- `imu0_remove_gravitational_acceleration: true` - Remove gravity from accelerometer readings
- `imu0_linear_acceleration_rejection_threshold` - Reject large acceleration spikes
- `imu0_angular_velocity_rejection_threshold` - Reject large angular velocity spikes

## Integration with Existing Navigation

The setup automatically integrates with your existing navigation stack:

1. **AMCL** now uses `/odometry/filtered` instead of `/odom`
2. **move_base** receives better odometry data
3. **Existing costmaps and planners** work unchanged
4. **Improved localization** leads to better path planning and execution

## Next Steps

1. **Test in simulation** first with Gazebo setup
2. **Deploy on real robot** with IMU integration
3. **Monitor performance** and tune parameters as needed
4. **Compare odometry quality** between raw and filtered estimates

For more advanced configuration options, refer to the [robot_localization documentation](http://docs.ros.org/en/melodic/api/robot_localization/html/index.html).
