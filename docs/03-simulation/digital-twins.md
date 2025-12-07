# Digital Twins and Sensor Simulation

Digital twins in robotics represent virtual replicas of physical systems that mirror their real-world counterparts' behaviors, characteristics, and states in real-time. Combined with accurate sensor simulation, digital twins provide an invaluable platform for testing, validating, and training robotic systems before deployment in the physical world.

## Learning Outcomes

After completing this section, you should be able to:
- Define and implement digital twin concepts for robotic systems
- Create accurate sensor simulation models in both Gazebo and Unity
- Understand the benefits and limitations of digital twin technologies
- Evaluate simulation-to-reality transfer approaches
- Design validation strategies for digital twin systems
- Identify the fidelity requirements for different robotic applications

## Core Concepts

### Digital Twin Architecture

A digital twin architecture for robotics typically consists of:
- **Physical System**: The real-world robot and environment
- **Virtual Model**: Digital replica of the physical system
- **Data Connection**: Real-time or batch data flow between physical and virtual systems
- **Simulation Engine**: Physics and sensor simulation capabilities
- **Analytics Layer**: For monitoring, prediction, and optimization

### Sensor Simulation Fidelity

Achieving realistic sensor simulation requires modeling:
- **Sensor Noise**: Realistic noise models based on physical sensor properties
- **Environmental Conditions**: Effects of lighting, weather, dust, etc.
- **Motion Artifacts**: Effects of robot motion on sensor data
- **Cross-Sensor Effects**: Interactions between different sensor modalities

### Reality Gap

The "reality gap" refers to the differences between simulation and real-world performance. Key components include:
- **System Modeling Errors**: Inaccuracies in physical modeling
- **Sensor Simulation Errors**: Differences in sensor behavior
- **Environmental Differences**: Simplifications in the simulated environment
- **Unmodeled Dynamics**: Physical effects not captured in simulation

## Equations and Models

### Digital Twin State Synchronization

The state of the digital twin (x_sim) should approximate the real system state (x_real):

```
x_sim(t) ≈ x_real(t)
```

With synchronization achieved through:

```
dx_sim/dt = f_sim(u, x_sim, t)
dx_real/dt = f_real(u, x_real, t)
```

Where `f_sim` and `f_real` are the system dynamics models and `u` is the control input.

### Sensor Simulation Model with Noise

For realistic sensor simulation:

```
z_sim = h(x_sim) + n_system + n_environment
```

Where:
- `z_sim` is the simulated sensor reading
- `h(x_sim)` is the noiseless sensor reading based on state
- `n_system` is system-specific sensor noise
- `n_environment` is environment-dependent noise

## Code Example: Sensor Simulation with Noise

Here's an example of implementing realistic sensor simulation with noise:

```python
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2


class SensorSimulatorNode(Node):
    def __init__(self):
        super().__init__('sensor_simulator')
        
        # Publishers for simulated sensors
        self.lidar_publisher = self.create_publisher(LaserScan, 'scan', 10)
        self.camera_publisher = self.create_publisher(Image, 'camera/image_raw', 10)
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Simulation parameters
        self.lidar_angle_min = -np.pi/2
        self.lidar_angle_max = np.pi/2
        self.lidar_angle_increment = np.pi / 180  # 1 degree
        self.lidar_range_min = 0.1
        self.lidar_range_max = 30.0
        
        # Camera parameters
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fov = 90  # degrees
        
        # Timer for sensor simulation
        self.timer = self.create_timer(0.1, self.publish_sensors)
        
        # Simulate robot moving in a world with obstacles
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        
        # Simulated obstacles in the environment
        self.obstacles = [
            {'x': 5.0, 'y': 2.0, 'radius': 0.5},  # Obstacle 1
            {'x': 3.0, 'y': -1.0, 'radius': 0.8},  # Obstacle 2
            {'x': 7.0, 'y': 0.0, 'radius': 1.0},   # Obstacle 3
        ]
    
    def simulate_lidar_scan(self):
        """Simulate LIDAR scan with realistic noise"""
        num_readings = int((self.lidar_angle_max - self.lidar_angle_min) / self.lidar_angle_increment) + 1
        ranges = []
        
        for i in range(num_readings):
            # Calculate angle of this ray in robot frame
            angle = self.lidar_angle_min + i * self.lidar_angle_increment
            
            # Transform to global frame
            global_angle = self.robot_theta + angle
            
            # Calculate ray endpoint
            max_range = self.lidar_range_max
            ray_x = self.robot_x + max_range * np.cos(global_angle)
            ray_y = self.robot_y + max_range * np.sin(global_angle)
            
            # Check for intersections with obstacles
            min_distance = self.lidar_range_max
            
            for obstacle in self.obstacles:
                # Calculate distance from ray start to obstacle center
                dx = obstacle['x'] - self.robot_x
                dy = obstacle['y'] - self.robot_y
                
                # Calculate projection of obstacle center onto ray
                ray_length = np.sqrt(dx*dx + dy*dy)
                ray_angle_to_obstacle = np.arctan2(dy, dx) - global_angle
                
                # Correct for angle wrapping
                ray_angle_to_obstacle = (ray_angle_to_obstacle + np.pi) % (2*np.pi) - np.pi
                
                # If obstacle is roughly in direction of ray
                if abs(ray_angle_to_obstacle) < 0.1:  # 0.1 radian = ~5.7 degrees
                    # Calculate closest point on ray to obstacle center
                    distance_to_obstacle = ray_length * np.cos(ray_angle_to_obstacle)
                    
                    # Perpendicular distance from ray to obstacle center
                    perpendicular_distance = ray_length * np.abs(np.sin(ray_angle_to_obstacle))
                    
                    # If ray passes close enough to obstacle
                    if perpendicular_distance <= obstacle['radius']:
                        # Calculate where ray intersects obstacle
                        intersection_distance = distance_to_obstacle - np.sqrt(obstacle['radius']**2 - perpendicular_distance**2)
                        
                        if 0 < intersection_distance < min_distance:
                            min_distance = intersection_distance
            
            # Add realistic sensor noise
            noise_std = 0.02  # 2cm standard deviation
            noisy_distance = min_distance + np.random.normal(0, noise_std)
            
            # Apply sensor range limits
            if noisy_distance < self.lidar_range_min:
                ranges.append(float('inf'))  # Out of range low
            elif noisy_distance > self.lidar_range_max:
                ranges.append(float('inf'))  # Out of range high
            else:
                ranges.append(noisy_distance)
        
        return ranges
    
    def simulate_camera_image(self):
        """Simulate camera image with realistic rendering"""
        # Create a simulated image using OpenCV
        image = np.ones((self.camera_height, self.camera_width, 3), dtype=np.uint8) * 150  # Gray background
        
        # Simulate obstacles as colored circles
        for obstacle in self.obstacles:
            # Transform obstacle position to robot frame
            dx = obstacle['x'] - self.robot_x
            dy = obstacle['y'] - self.robot_y
            
            # Convert to image coordinates (simplified pinhole model)
            # For a 90-degree FOV centered on robot's heading
            angle_to_obstacle = np.arctan2(dy, dx) - self.robot_theta
            
            # Only render if obstacle is in front of robot
            if abs(angle_to_obstacle) <= np.pi/2:  # 90 degree FOV
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Calculate image position (simplified)
                if distance > 0:
                    pixel_x = int(self.camera_width/2 + (angle_to_obstacle / (np.pi/2)) * self.camera_width/2)
                    pixel_y = int(self.camera_height/2)
                    
                    # Calculate approximate size based on distance
                    size = max(5, int(50 / distance))  # Closer objects appear larger
                    
                    if 0 <= pixel_x < self.camera_width:
                        # Draw obstacle as a colored circle
                        color = (0, 255, 0) if 'radius' in obstacle and obstacle['radius'] < 0.7 else (0, 100, 255)
                        image = cv2.circle(image, (pixel_x, pixel_y), size, color, -1)
        
        # Add realistic image noise
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def publish_sensors(self):
        """Publish simulated sensor data with realistic noise"""
        # Update robot position (simple movement model)
        self.robot_x += 0.1 * np.cos(self.robot_theta)  # Move forward slightly
        self.robot_theta += 0.05  # Rotate slowly
        
        # Create and publish LIDAR scan
        scan_msg = LaserScan()
        scan_msg.header = Header()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_frame'
        
        scan_msg.angle_min = self.lidar_angle_min
        scan_msg.angle_max = self.lidar_angle_max
        scan_msg.angle_increment = self.lidar_angle_increment
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1
        scan_msg.range_min = self.lidar_range_min
        scan_msg.range_max = self.lidar_range_max
        
        scan_msg.ranges = self.simulate_lidar_scan()
        self.lidar_publisher.publish(scan_msg)
        
        # Create and publish camera image
        image = self.simulate_camera_image()
        image_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        image_msg.header = Header()
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = 'camera_frame'
        self.camera_publisher.publish(image_msg)


def main(args=None):
    rclpy.init(args=args)
    sensor_simulator = SensorSimulatorNode()
    
    try:
        rclpy.spin(sensor_simulator)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_simulator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Simulation Demonstration

This node simulates both LIDAR and camera sensors with realistic noise models. The simulated robot navigates through an environment with obstacles, and the sensors detect these obstacles with appropriate noise and limitations, similar to real sensors.

## Hands-On Lab: Advanced Sensor Simulation

In this lab, you'll implement and validate advanced sensor simulation:

1. Create realistic sensor simulators with noise models
2. Validate the simulators against real sensor data
3. Compare different noise modeling approaches
4. Analyze the impact of sensor fidelity on robotic tasks

### Required Equipment:
- ROS 2 Humble environment
- Python development environment
- Basic understanding of probability and statistics
- (Optional) Real sensor data for comparison

### Instructions:
1. Create a new ROS 2 package: `ros2 pkg create --build-type ament_python advanced_sensor_simulator`
2. Add the SensorSimulatorNode code to your package
3. Create launch files to run the simulator with visualization tools
4. Add additional sensor types (e.g., IMU, GPS) with appropriate noise models
5. Implement parameterization of noise characteristics
6. Test the simulator with different environmental conditions
7. Visualize the sensor data in RViz
8. Document how different noise parameters affect robot perception
9. (Optional) Compare simulated data with real sensor data if available

## Common Pitfalls & Debugging Notes

- **Noise Modeling**: Ensure noise models are realistic but not overly complex
- **Computational Load**: Advanced sensor simulation can be computationally expensive
- **Parameter Tuning**: Carefully tune noise parameters based on real sensor specifications
- **Environmental Fidelity**: Balance environmental complexity with computational efficiency
- **Validation**: Validate sensor simulators against real sensor data when possible

## Summary & Key Terms

**Key Terms:**
- **Digital Twin**: Virtual replica of a physical system that mirrors its state and behavior
- **Sensor Simulation**: Modeling of real-world sensors with realistic noise and limitations
- **Reality Gap**: Differences between simulated and real-world robot performance
- **System Modeling Errors**: Inaccuracies in the physical model of the system
- **Sensor Fidelity**: Accuracy of the simulated sensor compared to the real sensor
- **Simulation Validation**: Process of verifying that simulation accurately represents reality
- **Perception Simulation**: Modeling of robot sensory inputs with realistic properties

## Further Reading & Citations

1. Rasheed, A., San, O., & Kvamsdal, T. (2020). Digital twin: Values, challenges and enablers from a modeling perspective. IEEE Access, 8, 21980-22012.
2. Batty, M. (2018). Digital twins. Environment and Planning B: Urban Analytics and City Science, 45(5), 817-820.
3. Kerschbaum, S., et al. (2021). Digital twin in manufacturing: A categorical literature review and classification. 
4. Khajavi, G. H., et al. (2019). Additive manufacturing in the spare parts supply chain. Computers & Industrial Engineering.

## Assessment Questions

1. Define digital twins and explain their role in robotics development.
2. What are the main components of a digital twin architecture for robotics?
3. Describe how sensor noise is modeled in realistic sensor simulation.
4. What is the "reality gap" and how can it be minimized?
5. Compare the computational requirements of high-fidelity vs. low-fidelity sensor simulation.

---
**Previous**: [Unity for Robotics Visualization](./unity.md)  
**Next**: [NVIDIA Isaac Platform - Introduction](../04-isaac-platform/intro.md)