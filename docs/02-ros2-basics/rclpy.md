# rclpy Python Client Library

The Robot Client Library for Python (rclpy) provides the Python API for ROS 2. It allows Python developers to create ROS 2 nodes, publish and subscribe to topics, make service calls, and interact with actions. Understanding rclpy is crucial for developing robotic applications in Python within the ROS 2 ecosystem.

## Learning Outcomes

After completing this section, you should be able to:
- Create ROS 2 nodes using rclpy
- Implement publishers and subscribers for message passing
- Create and use services for request-response communication
- Work with actions for long-running tasks
- Understand the lifecycle of a ROS 2 node using rclpy
- Use parameters and logging in ROS 2 nodes

## Core Concepts

### Node Structure

A typical rclpy node follows this structure:
- Initialize the ROS client library
- Create a node
- Create publishers, subscribers, services, or actions
- Spin the node to process callbacks
- Clean up resources at shutdown

### Message Passing

ROS 2 uses a middleware abstraction that allows different implementations of communication protocols. rclpy provides a Python interface to this middleware, enabling communication between nodes using topics, services, and actions.

### Callbacks and Execution

rclpy nodes can use different execution models:
- Single-threaded executor: Processes callbacks sequentially
- Multi-threaded executor: Processes callbacks in parallel
- Custom executors for specialized use cases

## Equations and Models

The node initialization and execution can be modeled as:

```
init() → create_node() → create_entities() → spin() → shutdown()
```

Where entities include publishers, subscribers, services, and actions.

## Code Example: Comprehensive rclpy Node

Here's an example of a comprehensive rclpy node implementing all communication patterns:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Standard message types
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from example_interfaces.srv import AddTwoInts
from example_interfaces.action import Fibonacci


class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Create a QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publisher for robot commands
        self.cmd_publisher = self.create_publisher(
            Twist, 'cmd_vel', 10)
        
        # Subscriber for sensor data
        self.scan_subscription = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, sensor_qos)
        
        # Subscriber for high-level commands
        self.cmd_subscription = self.create_subscription(
            String, 'robot_commands', self.command_callback, 10)
        
        # Service server for simple calculations
        self.calc_service = self.create_service(
            AddTwoInts, 'calculate_add', self.add_callback)
        
        # Action client for navigation
        self.nav_action_client = ActionClient(
            self, Fibonacci, 'navigate_to_goal')
        
        # Parameters with default values
        self.declare_parameter('linear_speed', 0.5)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('safety_distance', 0.5)
        
        # Get parameter values
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.safety_distance = self.get_parameter('safety_distance').value
        
        # Timer for periodic tasks
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Internal state
        self.obstacle_detected = False
        self.nav_goal_handle = None
        
        self.get_logger().info('Robot controller node initialized')
    
    def scan_callback(self, msg):
        """Process laser scan data"""
        # Find minimum distance
        min_distance = min([r for r in msg.ranges if r > 0.0], default=float('inf'))
        
        if min_distance < self.safety_distance:
            self.obstacle_detected = True
            self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m')
        else:
            self.obstacle_detected = False
    
    def command_callback(self, msg):
        """Process high-level commands"""
        command = msg.data.lower()
        
        if command == 'move_forward':
            self.move_robot(linear=0.2, angular=0.0)
        elif command == 'turn_left':
            self.move_robot(linear=0.0, angular=0.5)
        elif command == 'turn_right':
            self.move_robot(linear=0.0, angular=-0.5)
        elif command == 'stop':
            self.move_robot(linear=0.0, angular=0.0)
        elif command.startswith('navigate_to_'):
            # Extract goal information and send to action server
            self.send_navigation_goal(command)
    
    def add_callback(self, request, response):
        """Service callback for addition"""
        response.sum = request.a + request.b
        self.get_logger().info(f'Service call: {request.a} + {request.b} = {response.sum}')
        return response
    
    def move_robot(self, linear, angular):
        """Send movement command to robot"""
        if not self.obstacle_detected or (linear <= 0):  # Allow backward movement or stopping
            cmd = Twist()
            cmd.linear.x = linear
            cmd.angular.z = angular
            self.cmd_publisher.publish(cmd)
        else:
            self.get_logger().info('Movement blocked due to obstacle')
    
    def send_navigation_goal(self, command):
        """Send navigation goal to action server"""
        # Parse goal from command string
        try:
            # This is a simplified parsing - in practice would be more robust
            parts = command.split('_')
            if len(parts) >= 3:
                order = int(parts[-1])  # Get the last part as order
            else:
                order = 5  # Default order
        except ValueError:
            order = 5
        
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order
        
        # Wait for action server
        if not self.nav_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Navigation action server not available')
            return
        
        # Send goal
        self.nav_goal_handle = self.nav_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback_callback)
        
        self.nav_goal_handle.add_done_callback(self.navigation_goal_response_callback)
    
    def navigation_goal_response_callback(self, future):
        """Handle navigation goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Navigation goal rejected')
            return
        
        self.get_logger().info('Navigation goal accepted')
        self.nav_result_future = goal_handle.get_result_async()
        self.nav_result_future.add_done_callback(self.navigation_result_callback)
    
    def navigation_feedback_callback(self, feedback_msg):
        """Handle navigation feedback"""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Navigation feedback: sequence length {len(feedback.sequence)}')
    
    def navigation_result_callback(self, future):
        """Handle navigation result"""
        result = future.result().result
        self.get_logger().info(f'Navigation result: sequence {result.sequence}')
    
    def timer_callback(self):
        """Timer callback for periodic tasks"""
        # Log a periodic message
        self.get_logger().debug('Controller timer tick')
        
        # Update parameter values if they changed
        current_linear_speed = self.get_parameter('linear_speed').value
        if current_linear_speed != self.linear_speed:
            self.linear_speed = current_linear_speed
            self.get_logger().info(f'Linear speed updated to {self.linear_speed}')


def main(args=None):
    rclpy.init(args=args)
    
    # Create the node
    controller_node = RobotControllerNode()
    
    # Use multi-threaded executor to handle multiple callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(controller_node)
    
    try:
        # Spin the node
        executor.spin()
    except KeyboardInterrupt:
        controller_node.get_logger().info('Interrupted by user')
    finally:
        # Clean up
        controller_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Launch File for the Node

To run this node along with other components, you can create a launch file:

```python
# robot_controller_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_controller',
            executable='robot_controller',
            name='robot_controller',
            parameters=[
                {'linear_speed': 0.5},
                {'angular_speed': 0.5},
                {'safety_distance': 0.5}
            ],
            output='screen'
        )
    ])
```

## Simulation Demonstration

This comprehensive node demonstrates all the key features of rclpy. It can be run in a simulation environment with a robot model (like TurtleBot3) to perform navigation tasks while avoiding obstacles.

## Hands-On Lab: Building an Advanced rclpy Node

In this lab, you'll implement an advanced rclpy node that incorporates multiple communication patterns:

1. Create a node with publishers, subscribers, services, and actions
2. Implement parameter handling
3. Add logging and debugging capabilities
4. Test the node in simulation

### Required Equipment:
- ROS 2 Humble environment
- Python development environment
- (Optional) Gazebo simulation environment with a robot model

### Instructions:
1. Create a new ROS 2 package: `ros2 pkg create --build-type ament_python robot_controller`
2. Add the RobotControllerNode code to `robot_controller/robot_controller/robot_controller_node.py`
3. Add the launch file to `robot_controller/robot_controller/launch/robot_controller_launch.py`
4. Update the `setup.py` file to include the executable
5. Build the package: `colcon build --packages-select robot_controller`
6. Source the environment: `source install/setup.bash`
7. Run the node: `ros2 run robot_controller robot_controller_node`
8. Test the different functionalities:
   - Send commands via topic: `ros2 topic pub /robot_commands std_msgs/String "data: 'move_forward'"`
   - Call the service: `ros2 service call /calculate_add example_interfaces/srv/AddTwoInts "{a: 5, b: 7}"`
   - Change parameters: `ros2 param set /robot_controller linear_speed 0.7`
9. (Optional) Run with the robot simulation to see physical interaction

## Common Pitfalls & Debugging Notes

- **Thread Safety**: Be careful when accessing shared variables in callbacks; use locks if needed
- **Resource Management**: Always properly destroy nodes to free resources
- **Node Names**: Use unique node names to avoid conflicts in distributed systems
- **Parameter Declaration**: Declare parameters before getting them in the constructor
- **Exception Handling**: Wrap rclpy calls in try-catch blocks for robust error handling
- **Callback Groups**: Use appropriate callback groups when using timers and services together

## Summary & Key Terms

**Key Terms:**
- **rclpy**: Robot Client Library for Python, providing the Python API for ROS 2
- **Node**: A process that performs computation in ROS 2
- **QoS**: Quality of Service, settings that govern communication behavior
- **Callback**: Function executed when a message, service request, or action goal is received
- **Executor**: Manages the execution of callbacks in a ROS 2 node
- **Parameter**: Configuration value that can be set at runtime
- **Logging**: Recording information about node operation for debugging

## Further Reading & Citations

1. ROS 2 Documentation. (2023). "Tutorials: Writing a Simple Python Publisher and Subscriber." https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html
2. ROS 2 Documentation. (2023). "Tutorials: Writing a Simple Service and Client (Python)." https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Service-And-Client.html
3. ROS 2 Documentation. (2023). "Tutorials: Using Parameters in a Class (Python)." https://docs.ros.org/en/humble/Tutorials/Parameters/Using-Parameters-In-A-Class-Python.html
4. Kamran, M. A. (2021). ROS Robotics Projects. Packt Publishing.

## Assessment Questions

1. Explain the role of executors in rclpy and describe the differences between single-threaded and multi-threaded executors.
2. How do you declare and use parameters in an rclpy node? Why are parameters useful?
3. Describe the lifecycle of an rclpy node from initialization to shutdown.
4. What are QoS profiles and when would you use different QoS settings for different topics?
5. How do you implement service servers and clients using rclpy?

---
**Previous**: [URDF Modeling for Robotics](./urdf.md)  
**Next**: [Simulation Pipelines - Introduction](../03-simulation/intro.md)