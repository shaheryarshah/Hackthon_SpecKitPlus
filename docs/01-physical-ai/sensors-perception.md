# Sensors and Physical Perception

In Physical AI systems, sensors serve as the primary interface between the robot and its environment. The ability to accurately perceive the physical world is fundamental to making informed decisions and executing appropriate actions. This section will explore the different types of sensors used in robotics and how they enable physical perception.

## Learning Outcomes

After completing this section, you should be able to:
- Identify and classify different types of robotic sensors
- Understand how different sensors provide physical perception capabilities
- Evaluate the trade-offs between different sensor modalities
- Integrate multiple sensor inputs for robust perception
- Apply sensor models to physical environments

## Core Concepts

### Types of Sensors

Robotic sensors can be broadly classified into two categories:

**Proprioceptive Sensors**: These sensors provide information about the robot's own state:
- Encoders: Measure joint angles and wheel rotations
- Inertial Measurement Units (IMUs): Measure acceleration and angular velocity
- Force/torque sensors: Measure forces exerted by the robot or on the robot
- Temperature sensors: Monitor internal system temperatures

**Exteroceptive Sensors**: These sensors provide information about the external environment:
- Cameras: Provide visual information
- LIDAR: Measure distances using laser light
- Sonar: Measure distances using sound waves
- GPS: Provide global positioning information
- Tactile sensors: Detect contact and pressure

### Sensor Characteristics

When selecting and using sensors for Physical AI systems, consider these important characteristics:

- **Accuracy**: How close measurements are to the true values
- **Precision**: How consistent repeated measurements are
- **Resolution**: The smallest detectable change in the measured quantity
- **Range**: The minimum and maximum values the sensor can measure
- **Bandwidth**: The frequency at which the sensor can provide measurements
- **Reliability**: The likelihood that the sensor will provide correct measurements

## Equations and Models

### Sensor Noise Model

A basic sensor noise model can be expressed as:

```
z = h(x) + n
```

Where:
- `z` is the sensor measurement
- `h(x)` is the true value derived from the system state `x`
- `n` is the sensor noise, often modeled as Gaussian with mean 0 and variance σ²

### Sensor Fusion

When combining multiple sensor readings, we often use probabilistic approaches. The weighted fusion of two sensors can be expressed as:

```
z_fused = (σ₂² * z₁ + σ₁² * z₂) / (σ₁² + σ₂²)
```

Where:
- `z₁`, `z₂` are measurements from sensors 1 and 2
- `σ₁²`, `σ₂²` are the variances of sensors 1 and 2 respectively

## Code Example: Sensor Fusion

Here's an example of fusing data from multiple sensors to improve perception accuracy:

```python
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, MagneticField
from geometry_msgs.msg import PoseWithCovarianceStamped

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        # Subscribe to various sensors
        self.lidar_subscription = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, 10)
        self.imu_subscription = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.magnetometer_subscription = self.create_subscription(
            MagneticField, 'magnetic_field', self.magnetometer_callback, 10)
        
        # Publish fused pose estimates
        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, 'fused_pose', 10)
        
        # Initialize sensor data storage
        self.lidar_data = None
        self.imu_data = None
        self.magnetometer_data = None
        
        # Timer for fusion process
        self.timer = self.create_timer(0.05, self.fusion_callback)
        
    def lidar_callback(self, msg):
        """Process LIDAR data for position estimation"""
        self.lidar_data = msg
    
    def imu_callback(self, msg):
        """Process IMU data for orientation estimation"""
        self.imu_data = msg
        
    def magnetometer_callback(self, msg):
        """Process magnetometer data for compass heading"""
        self.magnetometer_data = msg
    
    def fusion_callback(self):
        """Fusion of sensor data to estimate pose"""
        if not all([self.lidar_data, self.imu_data, self.magnetometer_data]):
            return
            
        # Create pose message
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        
        # Estimate position using LIDAR data (simplified)
        # In practice, use more sophisticated methods like particle filters
        # or Kalman filters for fusion
        pose_msg.pose.pose.position.x = self.estimate_x_position()
        pose_msg.pose.pose.position.y = self.estimate_y_position()
        
        # Estimate orientation using IMU and magnetometer data
        orient = self.fuse_orientation()
        pose_msg.pose.pose.orientation = orient
        
        # Set covariance matrix (uncertainty estimates)
        self.set_covariance_matrix(pose_msg)
        
        # Publish fused estimate
        self.pose_publisher.publish(pose_msg)
    
    def estimate_x_position(self):
        """Simplified position estimation based on LIDAR landmarks"""
        # This is a placeholder - in practice use landmark matching,
        # SLAM, or other techniques
        return 0.0
    
    def estimate_y_position(self):
        """Simplified position estimation based on LIDAR landmarks"""
        # This is a placeholder
        return 0.0
    
    def fuse_orientation(self):
        """Fuse IMU and magnetometer data for orientation"""
        # Simplified fusion - in practice use sensor fusion filters
        import math
        
        # Extract orientation from IMU
        imu_q = self.imu_data.orientation
        # Magnetometer provides heading reference
        # This is a simplified approach
        
        # Return some orientation estimate
        return imu_q
    
    def set_covariance_matrix(self, pose_msg):
        """Set the uncertainty estimates in the covariance matrix"""
        # Placeholder covariance values
        pose_msg.pose.covariance = [
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0,  # Position x
            0.0, 0.1, 0.0, 0.0, 0.0, 0.0,  # Position y
            0.0, 0.0, 0.1, 0.0, 0.0, 0.0,  # Position z
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0,  # Orientation x
            0.0, 0.0, 0.0, 0.0, 0.1, 0.0,  # Orientation y
            0.0, 0.0, 0.0, 0.0, 0.0, 0.1   # Orientation z
        ]

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Simulation Demonstration

This code can be tested in a Gazebo simulation with a robot equipped with the necessary sensors. The SensorFusionNode demonstrates how multiple sensor inputs can be combined to create a more accurate estimate of the robot's pose in the environment.

## Hands-On Lab: Multi-Sensor Perception System

In this lab, you'll implement a sensor fusion system that combines data from multiple sensors:

1. Set up a simulated robot with multiple sensors
2. Implement a basic sensor fusion algorithm
3. Compare the fused results with individual sensor readings
4. Evaluate the improvement in perception accuracy

### Required Equipment:
- ROS 2 Humble environment
- Gazebo simulation environment
- A simulated robot with multiple sensors (e.g., TurtleBot3 with LIDAR and IMU)

### Instructions:
1. Launch a robot simulation with multiple sensors
2. Create the SensorFusionNode as shown in the code example
3. Visualize the results in Rviz
4. Test the robot in different environments (empty, cluttered, etc.)
5. Analyze how sensor fusion improves perception compared to single sensors

## Common Pitfalls & Debugging Notes

- **Sensor Calibration**: Uncalibrated sensors can introduce systematic errors
- **Timing Issues**: Sensors may have different update rates; synchronization is crucial
- **Noise Characteristics**: Different sensors have different noise properties; model these correctly in your fusion algorithm
- **Environmental Conditions**: Sensors may behave differently under various environmental conditions (lighting, temperature, etc.)

## Summary & Key Terms

**Key Terms:**
- **Proprioceptive Sensors**: Sensors that measure the robot's own state
- **Exteroceptive Sensors**: Sensors that measure external environment properties
- **Sensor Fusion**: Combining data from multiple sensors to improve accuracy
- **Sensor Noise**: Random variations in sensor measurements
- **Covariance**: A measure of uncertainty in sensor measurements
- **Kalman Filter**: A common algorithm for sensor fusion
- **SLAM**: Simultaneous Localization and Mapping

## Further Reading & Citations

1. Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics. MIT Press.
2. Siciliano, B., & Khatib, O. (Eds.). (2016). Springer Handbook of Robotics. Springer.
3. Cox, D., & Rehg, J. (2011). Modeling, Recognition, and Decoding of Temporal Structures in Human Motion. Computer Vision and Pattern Recognition.
4. Luinge, H., & Veltink, P. (2005). Measuring orientation of human body segments using miniature gyroscopes and accelerometers. Medical & Biological Engineering & Computing, 43(2), 273-282.

## Assessment Questions

1. Compare and contrast proprioceptive and exteroceptive sensors. Provide two examples of each and explain their roles in Physical AI systems.
2. Explain why sensor fusion is important in robotics. Describe one situation where sensor fusion would be more reliable than using a single sensor.
3. Describe the differences between accuracy and precision in the context of robotic sensors.
4. What are the main challenges in implementing sensor fusion for Physical AI systems?
5. How might environmental conditions (e.g., lighting, temperature) affect the performance of different sensor types?

---
**Previous**: [Introduction to Physical AI & Embodied Intelligence](./intro.md)  
**Next**: [ROS 2 Basics - Introduction](../02-ros2-basics/intro.md)