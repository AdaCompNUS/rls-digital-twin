#ifndef FETCH_WHOLE_BODY_CONTROLLER_H
#define FETCH_WHOLE_BODY_CONTROLLER_H

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/JointState.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <nav_msgs/Odometry.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <vector>
#include <string>
#include <memory>
#include <thread>

namespace fetch_drivers {

/**
 * @brief Controller class for whole body motion of the Fetch robot.
 * 
 * This class provides coordinated control of the base and arm using PID control.
 */
class WholeBodyController {
public:
    /**
     * @brief Construct a new Whole Body Controller object
     * 
     * @param nh ROS node handle
     */
    WholeBodyController(ros::NodeHandle& nh);

    /**
     * @brief Destroy the Whole Body Controller object
     */
    ~WholeBodyController();

    /**
     * @brief Reset all error terms for a new trajectory execution.
     */
    void resetErrors();

    /**
     * @brief Calculate base velocity commands using PID control.
     * 
     * @param currentPosition Current [x, y, theta] of the base
     * @param targetPosition Target [x, y, theta] for the base
     * @param distanceError Euclidean distance to the target position
     * @param angleToTarget Angle error to face the target position
     * @param finalAngleError Angle error for the final orientation
     * @param basePositionReached Whether the base has reached the target position
     * @return std::pair<double, double> (linear_vel, angular_vel) commands for the base
     */
    std::pair<double, double> calculateBaseVelocity(
        const std::vector<double>& currentPosition,
        const std::vector<double>& targetPosition,
        double distanceError,
        double angleToTarget,
        double finalAngleError,
        bool basePositionReached);

    /**
     * @brief Calculate joint velocity commands using PID control.
     * 
     * @param currentJointPositions Current joint positions [8-DOF]
     * @param targetJointPositions Target joint positions [8-DOF]
     * @param jointVelocities Output joint velocity commands
     * @param jointErrors Output joint errors
     * @return bool Whether all joints have reached their targets
     */
    bool calculateJointVelocities(
        const std::vector<double>& currentJointPositions,
        const std::vector<double>& targetJointPositions,
        std::vector<double>& jointVelocities,
        std::vector<double>& jointErrors);

    /**
     * @brief Send velocity commands to both the base and joints.
     * 
     * @param linearVel Linear velocity for the base
     * @param angularVel Angular velocity for the base
     * @param jointVelocities List of velocities for each joint
     */
    void sendVelocityCommands(
        double linearVel,
        double angularVel,
        const std::vector<double>& jointVelocities);

    /**
     * @brief Stop all robot motion by sending zero velocity commands.
     */
    void stopAllMotion();

    /**
     * @brief Follow a coordinated whole body trajectory using direct velocity control.
     * 
     * @param armPath List of arm joint configurations (8-DOF including torso)
     * @param baseConfigs List of base configurations [x, y, theta]
     * @param duration Total duration for the trajectory execution in seconds
     * @return bool True if execution succeeded, False otherwise
     */
    bool followWholeBodyTrajectory(
        const std::vector<std::vector<double>>& armPath,
        const std::vector<std::vector<double>>& baseConfigs,
        double duration = 10.0);

private:
    // ROS node handle
    ros::NodeHandle& nh_;

    // Publishers for base and joint velocity commands
    ros::Publisher cmdVelPub_;
    ros::Publisher jointVelPub_;

    // Subscriber for joint states
    ros::Subscriber jointStatesSub_;
    
    // Subscriber for odometry
    ros::Subscriber odomSub_;
    
    // Subscribers for trajectory messages
    ros::Subscriber armPathSub_;
    ros::Subscriber basePathSub_;

    // TF buffer and listener
    std::shared_ptr<tf2_ros::Buffer> tfBuffer_;
    std::shared_ptr<tf2_ros::TransformListener> tfListener_;

    // Latest joint states
    sensor_msgs::JointState currentJointState_;
    bool jointStatesReceived_;

    // Latest odom data
    nav_msgs::Odometry currentOdomData_;
    bool odomDataReceived_;
    
    // Stored trajectories
    std::vector<std::vector<double>> armPath_;
    std::vector<std::vector<double>> basePath_;
    bool armPathReceived_;
    bool basePathReceived_;
    ros::Time trajectoryStartTime_;
    double trajectoryDuration_;
    bool executingTrajectory_;

    // Control loop rate (Hz)
    double controlRate_;
    
    // Velocity limits
    double maxLinearVel_;
    double maxAngularVel_;
    double positionTolerance_;
    double angleTolerance_;
    
    // PID controller gains for base velocity control
    double kpLinear_;
    double kpAngular_;
    double kdLinear_;
    double kdAngular_;
    
    // PID controller gains for joint velocity control
    double kpJoint_;
    double kdJoint_;
    double kiJoint_;
    
    // For derivative term in base control
    double lastPosError_;
    double lastAngError_;
    
    // For integral and derivative terms in joint control
    std::vector<double> jointIntegralErrors_;
    std::vector<double> jointLastErrors_;
    
    // Maximum joint velocities
    std::vector<double> maxJointVel_;
    
    // Joint names for the arm and torso (8-DOF)
    std::vector<std::string> jointNames_;

    // Callbacks
    void jointStatesCallback(const sensor_msgs::JointState::ConstPtr& msg);
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void armPathCallback(const trajectory_msgs::JointTrajectory::ConstPtr& msg);
    void basePathCallback(const trajectory_msgs::JointTrajectory::ConstPtr& msg);
    
    // Execute trajectory if both arm and base paths are received
    void executeTrajectoryIfReady();

    // Helper method to get current joint positions
    bool getCurrentJointPositions(std::vector<double>& positions);

    // Helper method to normalize angles to [-pi, pi]
    double normalizeAngle(double angle);
};

} // namespace fetch_drivers

#endif // FETCH_WHOLE_BODY_CONTROLLER_H 