# Examples

This directory contains example scripts demonstrating various capabilities of the Robot Living Studio digital twin. The examples cover navigation and manipulation tasks using the Fetch robot.

## Getting Started

Before running any examples, you need to start the digital twin environment:

```bash
roslaunch low_level_planning rls_env.launch
```

This will launch the GAZEBO simulation of the Robot Living Studio environment. The environment includes:

- Fetch robot
- Basic studio layout
- Interactable objects (e.g., milk box)
- Obstacles (e.g., chairs)

Users can load additional objects and obstacles as needed.

## Navigation Examples

### Basic Movement Control

To control the robot's movement using keyboard:

1. Open a new terminal
2. Run:

```bash
python examples/navigation/base_movement.py
```

### Autonomous Navigation

For autonomous navigation to target positions:

1. Launch the navigation stack in a new terminal:

```bash
roslaunch low_level_planning navigation.launch
```

2. Ensure the robot's initial pose in the simulation matches the RViz map
   - Manually adjust the pose estimation if needed
   - The robot will automatically refine its localization during navigation

3. Run the target position navigation example:

```bash
python examples/navigation/go_to_target_position.py
```

## Manipulation Examples

### Joint Space Control

To demonstrate motion planning using joint states:

```bash
python examples/manipulation/move_arm_joints.py
```

### Cartesian Space Control

For direct end-effector control in Cartesian space, we provide an example that replays a pre-recorded trajectory for opening a fridge:

```bash
python examples/manipulation/replay_open_fridge_trajectory.py
```

This example uses:

- Direct end-effector control through a PD controller
- Pre-recorded trajectory data from `trajectory/abs_ee_pose.npy`
- Cartesian space interface for more intuitive manipulation

## Notes

- Make sure the ROS environment is properly sourced before running examples
- Each example can be modified to understand different aspects of robot control
- The robot will automatically handle collision avoidance during both navigation and manipulation tasks
- The pre-recorded trajectories serve as templates that can be adapted for similar tasks