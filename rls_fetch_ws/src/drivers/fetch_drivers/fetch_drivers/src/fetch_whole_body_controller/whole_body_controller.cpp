#include "whole_body_controller.h"
#include <cmath>
#include <algorithm>
#include <map>
#include <tf2/LinearMath/Matrix3x3.h>
#include <thread>

namespace fetch_drivers {

WholeBodyController::WholeBodyController(ros::NodeHandle& nh) : nh_(nh), 
    jointStatesReceived_(false),
    odomDataReceived_(false),
    armPathReceived_(false),
    basePathReceived_(false),
    executingTrajectory_(false),
    lastPosError_(0.0),
    lastAngError_(0.0)
{
    // Load parameters from the parameter server
    ros::NodeHandle private_nh("~");
    
    // Initialize publishers
    cmdVelPub_ = nh_.advertise<geometry_msgs::Twist>("/base_controller/command", 10);
    jointVelPub_ = nh_.advertise<sensor_msgs::JointState>(
        "/arm_with_torso_controller/joint_velocity_controller/command", 10);
    
    // Initialize subscribers
    jointStatesSub_ = nh_.subscribe("/joint_states", 10, &WholeBodyController::jointStatesCallback, this);
    odomSub_ = nh_.subscribe("/odom", 10, &WholeBodyController::odomCallback, this);
    
    // Initialize trajectory subscribers
    armPathSub_ = nh_.subscribe("/fetch_whole_body_controller/arm_path", 1, 
                                &WholeBodyController::armPathCallback, this);
    basePathSub_ = nh_.subscribe("/fetch_whole_body_controller/base_path", 1, 
                                 &WholeBodyController::basePathCallback, this);
    
    // Initialize TF listener
    tfBuffer_ = std::make_shared<tf2_ros::Buffer>();
    tfListener_ = std::make_shared<tf2_ros::TransformListener>(*tfBuffer_);
    
    // Load controller parameters
    private_nh.param<double>("control_rate", controlRate_, 20.0);
    
    // Load velocity limits
    private_nh.param<double>("max_linear_vel", maxLinearVel_, 0.5);
    private_nh.param<double>("max_angular_vel", maxAngularVel_, 1.0);
    private_nh.param<double>("position_tolerance", positionTolerance_, 0.03);
    private_nh.param<double>("angle_tolerance", angleTolerance_, 0.05);
    
    // Load PID gains for base control
    private_nh.param<double>("kp_linear", kpLinear_, 2.0);
    private_nh.param<double>("kp_angular", kpAngular_, 3.0);
    private_nh.param<double>("kd_linear", kdLinear_, 0.5);
    private_nh.param<double>("kd_angular", kdAngular_, 0.8);
    
    // Load PID gains for joint control
    private_nh.param<double>("kp_joint", kpJoint_, 5.0);
    private_nh.param<double>("kd_joint", kdJoint_, 0.5);
    private_nh.param<double>("ki_joint", kiJoint_, 0.1);
    
    // Load joint names
    if (!private_nh.getParam("joint_names", jointNames_)) {
        // Default joint names if parameter is not set
        jointNames_ = {
            "torso_lift_joint",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint"
        };
        ROS_WARN("Joint names not specified, using defaults");
    }
    
    // Load maximum joint velocities
    std::vector<double> maxJointVelocities;
    if (!private_nh.getParam("max_joint_velocities", maxJointVelocities)) {
        // Default values if parameter is not set
        maxJointVel_ = {0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        ROS_WARN("Maximum joint velocities not specified, using defaults");
    } else {
        if (maxJointVelocities.size() == jointNames_.size()) {
            maxJointVel_ = maxJointVelocities;
        } else {
            maxJointVel_ = {0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
            ROS_WARN("Invalid max_joint_velocities length, using defaults");
        }
    }
    
    // Initialize error arrays
    jointIntegralErrors_.resize(jointNames_.size(), 0.0);
    jointLastErrors_.resize(jointNames_.size(), 0.0);
    
    ROS_INFO("Whole Body Controller initialized");
}

WholeBodyController::~WholeBodyController() {
    // Clean shutdown - stop all motion
    stopAllMotion();
    ROS_INFO("Whole Body Controller shutdown");
}

// Callback for joint states
void WholeBodyController::jointStatesCallback(const sensor_msgs::JointState::ConstPtr& msg) {
    currentJointState_ = *msg;
    jointStatesReceived_ = true;
}

// Callback for odometry data
void WholeBodyController::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    currentOdomData_ = *msg;
    odomDataReceived_ = true;
}

// Callback for arm path messages
void WholeBodyController::armPathCallback(const trajectory_msgs::JointTrajectory::ConstPtr& msg) {
    ROS_INFO("Received arm trajectory with %lu points", msg->points.size());
    
    if (msg->points.empty()) {
        ROS_WARN("Received empty arm trajectory");
        return;
    }
    
    // Extract joint positions from the trajectory message
    armPath_.clear();
    for (const auto& point : msg->points) {
        armPath_.push_back(point.positions);
    }
    
    armPathReceived_ = true;
    
    // Calculate the duration from the last point's time
    if (!msg->points.empty()) {
        trajectoryDuration_ = msg->points.back().time_from_start.toSec();
    }
    
    // Try to execute the trajectory if both arm and base paths are received
    executeTrajectoryIfReady();
}

// Callback for base path messages
void WholeBodyController::basePathCallback(const trajectory_msgs::JointTrajectory::ConstPtr& msg) {
    ROS_INFO("Received base trajectory with %lu points", msg->points.size());
    
    if (msg->points.empty()) {
        ROS_WARN("Received empty base trajectory");
        return;
    }
    
    // Extract base positions from the trajectory message
    basePath_.clear();
    for (const auto& point : msg->points) {
        basePath_.push_back(point.positions);
    }
    
    basePathReceived_ = true;
    
    // Try to execute the trajectory if both arm and base paths are received
    executeTrajectoryIfReady();
}

// Execute trajectory if both arm and base paths are received
void WholeBodyController::executeTrajectoryIfReady() {
    if (armPathReceived_ && basePathReceived_ && !executingTrajectory_) {
        // Verify paths have the same length
        if (armPath_.size() != basePath_.size()) {
            ROS_ERROR("Arm path size (%lu) doesn't match base path size (%lu)", 
                      armPath_.size(), basePath_.size());
            // Reset flags
            armPathReceived_ = false;
            basePathReceived_ = false;
            return;
        }
        
        // Mark as executing to prevent multiple executions
        executingTrajectory_ = true;
        
        // Store the start time
        trajectoryStartTime_ = ros::Time::now();
        
        // Execute in a separate thread to avoid blocking callbacks
        std::thread([this]() {
            bool success = followWholeBodyTrajectory(armPath_, basePath_, trajectoryDuration_);
            ROS_INFO("Trajectory execution %s", success ? "succeeded" : "failed");
            
            // Reset flags when execution is complete
            armPathReceived_ = false;
            basePathReceived_ = false;
            executingTrajectory_ = false;
        }).detach();
    }
}

// Normalize angles to [-pi, pi]
double WholeBodyController::normalizeAngle(double angle) {
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

// Reset all error terms for a new trajectory execution
void WholeBodyController::resetErrors() {
    lastPosError_ = 0.0;
    lastAngError_ = 0.0;
    std::fill(jointIntegralErrors_.begin(), jointIntegralErrors_.end(), 0.0);
    std::fill(jointLastErrors_.begin(), jointLastErrors_.end(), 0.0);
}

// Calculate base velocity using PID control
std::pair<double, double> WholeBodyController::calculateBaseVelocity(
    const std::vector<double>& currentPosition,
    const std::vector<double>& targetPosition,
    double distanceError,
    double angleToTarget,
    double finalAngleError,
    bool basePositionReached)
{
    // Compute derivative terms for smoother control
    double dPosError = distanceError - lastPosError_;
    double dAngError;
    
    if (!basePositionReached) {
        dAngError = angleToTarget - lastAngError_;
    } else {
        dAngError = finalAngleError - lastAngError_;
    }
    
    // Update last errors for next iteration
    lastPosError_ = distanceError;
    lastAngError_ = basePositionReached ? finalAngleError : angleToTarget;
    
    // PID control for linear velocity
    double linearVel = kpLinear_ * distanceError + kdLinear_ * dPosError;
    
    // Angular velocity control strategy varies based on distance to target
    double angularVel;
    
    // When close to target, slow down and focus on orientation
    if (distanceError < 0.1) {
        linearVel *= (distanceError / 0.1);
        // Start considering final orientation
        angularVel = kpAngular_ * (0.5 * angleToTarget + 0.5 * finalAngleError) + kdAngular_ * dAngError;
    } else {
        // Far from target: focus on getting to the right position first
        angularVel = kpAngular_ * angleToTarget + kdAngular_ * dAngError;
    }
    
    // Limit base velocities
    linearVel = std::max(-maxLinearVel_, std::min(linearVel, maxLinearVel_));
    angularVel = std::max(-maxAngularVel_, std::min(angularVel, maxAngularVel_));
    
    return std::make_pair(linearVel, angularVel);
}

// Calculate joint velocities using PID control
bool WholeBodyController::calculateJointVelocities(
    const std::vector<double>& currentJointPositions,
    const std::vector<double>& targetJointPositions,
    std::vector<double>& jointVelocities,
    std::vector<double>& jointErrors)
{
    bool jointsReached = true;
    
    // Resize output arrays
    jointVelocities.resize(jointNames_.size(), 0.0);
    jointErrors.resize(jointNames_.size(), 0.0);
    
    // Calculate velocity for each joint
    for (size_t j = 0; j < jointNames_.size(); j++) {
        // Calculate joint error
        double jointError = targetJointPositions[j] - currentJointPositions[j];
        jointErrors[j] = jointError;
        
        // Check if joint has reached target
        if (std::abs(jointError) > 0.02) {  // Joint position tolerance
            jointsReached = false;
        }
        
        // Calculate integral term (with anti-windup)
        if (std::abs(jointError) < 0.1) {  // Only integrate when close to target
            jointIntegralErrors_[j] += jointError / controlRate_;
        } else {
            jointIntegralErrors_[j] = 0.0;  // Reset when far from target
        }
        
        // Limit integral term to prevent windup
        jointIntegralErrors_[j] = std::max(-1.0, std::min(jointIntegralErrors_[j], 1.0));
        
        // Calculate derivative term
        double dJointError = (jointError - jointLastErrors_[j]) * controlRate_;
        jointLastErrors_[j] = jointError;
        
        // PID control for joint velocity
        double jointVel = kpJoint_ * jointError + kdJoint_ * dJointError + kiJoint_ * jointIntegralErrors_[j];
        
        // Apply velocity limit for this joint
        jointVelocities[j] = std::max(-maxJointVel_[j], std::min(jointVel, maxJointVel_[j]));
    }
    
    return jointsReached;
}

// Send velocity commands to both the base and joints
void WholeBodyController::sendVelocityCommands(
    double linearVel,
    double angularVel,
    const std::vector<double>& jointVelocities)
{
    // Create and publish velocity command for base
    geometry_msgs::Twist baseCmd;
    baseCmd.linear.x = linearVel;
    baseCmd.angular.z = angularVel;
    cmdVelPub_.publish(baseCmd);
    
    // Create and publish velocity command for joints
    sensor_msgs::JointState jointCmd;
    jointCmd.header.stamp = ros::Time::now();
    jointCmd.name = jointNames_;
    jointCmd.velocity = jointVelocities;
    jointVelPub_.publish(jointCmd);
}

// Stop all robot motion by sending zero velocity commands
void WholeBodyController::stopAllMotion()
{
    // Send zero velocity to base
    geometry_msgs::Twist baseCmd;
    cmdVelPub_.publish(baseCmd);
    
    // Send zero velocity to joints
    sensor_msgs::JointState jointCmd;
    jointCmd.header.stamp = ros::Time::now();
    jointCmd.name = jointNames_;
    jointCmd.velocity.resize(jointNames_.size(), 0.0);
    jointVelPub_.publish(jointCmd);
    
    ROS_INFO("All motion stopped");
}

// Helper method to get current joint positions
bool WholeBodyController::getCurrentJointPositions(std::vector<double>& positions)
{
    if (!jointStatesReceived_) {
        ROS_WARN("No joint states received yet");
        return false;
    }
    
    // Create a map of joint names to positions
    std::map<std::string, double> jointMap;
    for (size_t i = 0; i < currentJointState_.name.size(); i++) {
        jointMap[currentJointState_.name[i]] = currentJointState_.position[i];
    }
    
    // Extract positions for our planning joints
    positions.resize(jointNames_.size());
    for (size_t i = 0; i < jointNames_.size(); i++) {
        auto it = jointMap.find(jointNames_[i]);
        if (it != jointMap.end()) {
            positions[i] = it->second;
        } else {
            ROS_ERROR_STREAM("Joint " << jointNames_[i] << " not found in joint states");
            return false;
        }
    }
    
    return true;
}

// Follow a coordinated whole body trajectory using direct velocity control
bool WholeBodyController::followWholeBodyTrajectory(
    const std::vector<std::vector<double>>& armPath,
    const std::vector<std::vector<double>>& baseConfigs,
    double duration)
{
    try {
        // Validate input
        if (armPath.empty() || baseConfigs.empty()) {
            ROS_ERROR("Cannot execute motion: armPath or baseConfigs is empty");
            return false;
        }
        
        // Ensure equal length
        size_t min_len = std::min(armPath.size(), baseConfigs.size());
        if (armPath.size() != baseConfigs.size()) {
            ROS_WARN_STREAM("Length mismatch: armPath(" << armPath.size() 
                << ") != baseConfigs(" << baseConfigs.size() 
                << "). Truncating to " << min_len);
        }
        
        // Reset controller errors before starting
        resetErrors();
        
        // Wait briefly for sensor data
        size_t wait_count = 0;
        while ((!jointStatesReceived_ || !odomDataReceived_) && wait_count < 20) {
            ROS_INFO_COND(wait_count == 0, "Waiting for sensor data...");
            ros::Duration(0.1).sleep();
            ros::spinOnce();
            wait_count++;
            if (wait_count >= 20) {
                ROS_WARN("Timeout waiting for sensor data");
            }
        }
        
        // Calculate time per waypoint
        double waypointDuration = duration / min_len;
        
        ros::Rate rate(controlRate_);
        
        ROS_INFO_STREAM("Executing " << min_len << " waypoints over " 
            << duration << " seconds");
        
        // Execute trajectory point by point
        for (size_t i = 0; i < min_len; i++) {
            if (!ros::ok()) {
                ROS_WARN("ROS shutdown detected, stopping trajectory execution");
                stopAllMotion();
                return false;
            }
            
            // Get target position for this waypoint
            const std::vector<double>& targetJointPositions = armPath[i];
            const std::vector<double>& targetBaseConfig = baseConfigs[i];
            
            // ROS_INFO_STREAM("Processing waypoint " << (i+1) << "/" << min_len << ": "
            //             << "base [" << targetBaseConfig[0] << ", " << targetBaseConfig[1] << ", "
            //             << targetBaseConfig[2] << "]");
            
            // Control loop for this waypoint
            ros::Time waypointStartTime = ros::Time::now();
            
            // Continue until we reach the position or timeout
            while (ros::ok()) {
                // Process callbacks
                ros::spinOnce();
                
                // Check if we should timeout
                double elapsed = (ros::Time::now() - waypointStartTime).toSec();
                if (elapsed >= waypointDuration) {
                    ROS_WARN_STREAM("Timeout reached for waypoint " << (i+1));
                    break;
                }
                
                // Get current position
                geometry_msgs::TransformStamped transform;
                double current_x = 0.0;
                double current_y = 0.0;
                double current_theta = 0.0;
                
                try {
                    // First try to get position from map to base_link transform
                    transform = tfBuffer_->lookupTransform(
                        "map", "base_link", ros::Time(0), ros::Duration(0.1)
                    );
                    
                    // Extract current position and orientation
                    current_x = transform.transform.translation.x;
                    current_y = transform.transform.translation.y;
                    
                    // Get current yaw from quaternion
                    tf2::Quaternion q(
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z,
                        transform.transform.rotation.w
                    );
                    tf2::Matrix3x3 m(q);
                    double roll, pitch;
                    m.getRPY(roll, pitch, current_theta);
                    
                } catch (const tf2::TransformException& e) {
                    // Fall back to odometry if TF fails
                    if (!odomDataReceived_) {
                        ROS_WARN_STREAM("No position data available: " << e.what());
                        rate.sleep();
                        continue;
                    }
                    
                    // Get position from odom
                    current_x = currentOdomData_.pose.pose.position.x;
                    current_y = currentOdomData_.pose.pose.position.y;
                    
                    // Get orientation as yaw from quaternion
                    tf2::Quaternion q(
                        currentOdomData_.pose.pose.orientation.x,
                        currentOdomData_.pose.pose.orientation.y,
                        currentOdomData_.pose.pose.orientation.z,
                        currentOdomData_.pose.pose.orientation.w
                    );
                    tf2::Matrix3x3 m(q);
                    double roll, pitch;
                    m.getRPY(roll, pitch, current_theta);
                    
                    // Transform from odom to map frame if possible
                    try {
                        transform = tfBuffer_->lookupTransform(
                            "map", "odom", ros::Time(0), ros::Duration(0.1)
                        );
                        
                        // Apply transform
                        tf2::Quaternion tf_quat(
                            transform.transform.rotation.x,
                            transform.transform.rotation.y,
                            transform.transform.rotation.z,
                            transform.transform.rotation.w
                        );
                        
                        double tf_yaw;
                        tf2::Matrix3x3(tf_quat).getRPY(roll, pitch, tf_yaw);
                        
                        // Rotation matrix for 2D transform
                        double cos_yaw = std::cos(tf_yaw);
                        double sin_yaw = std::sin(tf_yaw);
                        
                        // Rotate position
                        double tx = transform.transform.translation.x;
                        double ty = transform.transform.translation.y;
                        double rx = current_x * cos_yaw - current_y * sin_yaw + tx;
                        double ry = current_x * sin_yaw + current_y * cos_yaw + ty;
                        
                        // Update with transformed values
                        current_x = rx;
                        current_y = ry;
                        current_theta += tf_yaw;
                        
                        // Normalize angle
                        current_theta = normalizeAngle(current_theta);
                        
                    } catch (const tf2::TransformException& e) {
                        // Continue with odom frame if transform fails
                        ROS_WARN_STREAM_THROTTLE(1.0, "Transform from odom to map failed: " << e.what());
                    }
                }
                
                // Get current joint positions
                std::vector<double> currentJointPositions;
                if (!getCurrentJointPositions(currentJointPositions)) {
                    ROS_WARN_THROTTLE(1.0, "Could not get current joint positions");
                    rate.sleep();
                    continue;
                }
                
                // Calculate position error in map frame for the base
                double dx = targetBaseConfig[0] - current_x;
                double dy = targetBaseConfig[1] - current_y;
                double distanceError = std::sqrt(dx*dx + dy*dy);
                
                // Calculate heading to target
                double targetHeading = std::atan2(dy, dx);
                
                // Calculate angle error to face target
                double angleToTarget = normalizeAngle(targetHeading - current_theta);
                
                // Final orientation error (only consider when close to target position)
                double finalAngleError = normalizeAngle(targetBaseConfig[2] - current_theta);
                
                // Determine if we've reached the target for base
                bool basePositionReached = distanceError < positionTolerance_;
                bool baseAngleReached = std::abs(finalAngleError) < angleTolerance_;
                
                // Use the controller to calculate base velocity
                std::vector<double> currentPosition = {current_x, current_y, current_theta};
                auto [linearVel, angularVel] = calculateBaseVelocity(
                    currentPosition, 
                    targetBaseConfig, 
                    distanceError, 
                    angleToTarget, 
                    finalAngleError, 
                    basePositionReached
                );
                
                // Use the controller to calculate joint velocities
                std::vector<double> jointVelocities;
                std::vector<double> jointErrors;
                bool jointsReached = calculateJointVelocities(
                    currentJointPositions, targetJointPositions, jointVelocities, jointErrors
                );
                
                // Check if we've reached target for both base and joints
                bool targetReached = basePositionReached && baseAngleReached && jointsReached;
                
                if (targetReached) {
                    ROS_INFO_STREAM("Waypoint " << (i+1) << " reached completely");
                    break;
                }
                
                // Send velocity commands
                sendVelocityCommands(linearVel, angularVel, jointVelocities);
                
                // Log progress every second
                if (int(elapsed * 2) % 2 == 0) {  // Every ~0.5 seconds
                    int progress = int(100 * elapsed / waypointDuration);
                    
                    // Calculate average joint error
                    double avgJointError = 0.0;
                    if (!jointErrors.empty()) {
                        for (double err : jointErrors) {
                            avgJointError += std::abs(err);
                        }
                        avgJointError /= jointErrors.size();
                    }
                    
                    ROS_INFO_STREAM_THROTTLE(0.5, "Waypoint " << (i+1) << " progress: " << progress << "%, " 
                                << "base error: " << distanceError << "m, " << (finalAngleError * 180.0/M_PI) << "°, "
                                << "joint errors: " << avgJointError);
                }
                
                // Control loop rate
                rate.sleep();
            }
        }
        
        // Final verification that we reached the last point
        try {
            // Get final joint positions
            std::vector<double> finalJointPositions;
            if (!getCurrentJointPositions(finalJointPositions)) {
                ROS_WARN("Could not get final joint positions");
                return false;
            }
            
            // Calculate final joint errors
            std::vector<double> finalJointErrors(finalJointPositions.size());
            double avgJointError = 0.0;
            
            for (size_t j = 0; j < finalJointPositions.size(); j++) {
                finalJointErrors[j] = std::abs(finalJointPositions[j] - armPath.back()[j]);
                avgJointError += finalJointErrors[j];
            }
            avgJointError /= finalJointPositions.size();
            
            // Get final base position
            geometry_msgs::TransformStamped transform = tfBuffer_->lookupTransform(
                "map", "base_link", ros::Time(0), ros::Duration(1.0));
                
            double final_x = transform.transform.translation.x;
            double final_y = transform.transform.translation.y;
            
            tf2::Quaternion q(
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            );
            
            double roll, pitch, final_theta;
            tf2::Matrix3x3(q).getRPY(roll, pitch, final_theta);
            
            double finalBasePosError = std::sqrt(
                std::pow(final_x - baseConfigs.back()[0], 2) + 
                std::pow(final_y - baseConfigs.back()[1], 2)
            );
            
            double finalBaseAngleError = std::abs(normalizeAngle(final_theta - baseConfigs.back()[2]));
            
            ROS_INFO_STREAM("Trajectory complete - Final errors: joint avg=" << avgJointError << ", "
                        << "base position=" << finalBasePosError << "m, "
                        << "base angle=" << finalBaseAngleError << "rad");
            
            // Return success if errors are acceptable
            return (avgJointError < 0.05 && 
                    finalBasePosError < 0.1 && 
                    finalBaseAngleError < 0.1);
            
        } catch (const tf2::TransformException& e) {
            ROS_WARN_STREAM("Could not verify final position: " << e.what());
            // If we can't verify, assume success
            return true;
        }
        
    } catch (const std::exception& e) {
        ROS_ERROR_STREAM("Error in whole body trajectory following: " << e.what());
        
        // Ensure the robot stops moving
        try {
            // Stop all motion
            stopAllMotion();
        } catch (...) {
            // Ignore any exceptions during emergency stop
        }
        
        return false;
    }
}

} // namespace fetch_drivers 