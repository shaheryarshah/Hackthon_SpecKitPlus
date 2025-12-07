# ROS 2 Nodes, Topics, Services, and Actions

This section provides a deeper dive into the core communication patterns in ROS 2: nodes, topics, services, and actions. Understanding these concepts is essential for building modular, distributed robotic applications.

## Learning Outcomes

After completing this section, you should be able to:
- Design appropriate communication patterns for different robotic tasks
- Implement nodes that communicate via topics, services, and actions
- Understand when to use each communication pattern
- Configure Quality of Service (QoS) settings appropriately
- Debug communication issues in ROS 2 applications

## Core Concepts

### Nodes - The Foundation of ROS 2

A node is a fundamental component of ROS 2 that performs computation. Each node is typically responsible for one specific task and communicates with other nodes through topics, services, or actions. Nodes run in isolation but work together to achieve complex robotic behaviors.

### Topics - Asynchronous Communication

Topics enable asynchronous, many-to-many communication using a publish-subscribe pattern. Publishers send messages to topics without knowing which subscribers will receive them. Subscribers receive messages from topics without knowing which publishers sent them. This loose coupling allows for flexible system architecture.

### Services - Synchronous Request-Response

Services enable synchronous, one-to-one communication using a request-response pattern. A client sends a request to a service server and waits for a response. The communication is blocking, meaning the client waits for the complete response before continuing.

### Actions - Long-Running Tasks with Feedback

Actions are designed for tasks that take a significant amount of time to complete and may be preempted. Actions provide feedback during execution and send a result when complete. They support goals, which can be sent, canceled, or modified during execution.

## Equations and Models

### Communication Patterns

The communication patterns can be expressed as:

**Topics (Publish-Subscribe)**:
```
p_i → t_j → {s_k1, s_k2, ..., s_km}
```

**Services (Request-Response)**:
```
c_i → req → s_j → res → c_i
```

**Actions (Goal-Feedback-Result)**:
```
c_i → goal → s_j → {feedback}^n → result → c_i
```

### Quality of Service Model

QoS policies in ROS 2 can be modeled as:

```
QoS = {reliability, durability, history, depth, ...}
```

Where reliability can be `RELIABLE` or `BEST_EFFORT`, durability can be `TRANSIENT_LOCAL` or `VOLATILE`, etc.

## Code Example: Advanced Communication Patterns

Here's an example implementing all three communication patterns in ROS 2:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Import standard message types
from std_msgs.msg import String
from example_interfaces.srv import AddTwoInts
from example_interfaces.action import Fibonacci


class RobotCommunicationNode(Node):
    def __init__(self):
        super().__init__('robot_communication_node')
        
        # Topic publisher for robot status
        self.status_publisher = self.create_publisher(String, 'robot_status', 10)
        
        # Topic subscriber for commands
        self.command_subscription = self.create_subscription(
            String, 'robot_commands', self.command_callback, 10)
        
        # Service server for simple calculations
        self.service = self.create_service(
            AddTwoInts, 'add_two_ints', self.add_two_ints_callback)
        
        # Action client for navigation tasks
        self.action_client = ActionClient(
            self, Fibonacci, 'fibonacci_action')
        
        # Timer to periodically publish status
        self.timer = self.create_timer(1.0, self.publish_status)
        self.status_counter = 0
        
    def publish_status(self):
        """Publish robot status periodically"""
        msg = String()
        msg.data = f'Robot status update {self.status_counter}'
        self.status_publisher.publish(msg)
        self.status_counter += 1
        
    def command_callback(self, msg):
        """Handle incoming commands"""
        self.get_logger().info(f'Received command: {msg.data}')
        
        # Process command based on content
        if 'calculate_fibonacci' in msg.data:
            self.send_fibonacci_goal()
    
    def add_two_ints_callback(self, request, response):
        """Service callback for adding two integers"""
        result = request.a + request.b
        response.sum = result
        self.get_logger().info(f'{request.a} + {request.b} = {response.sum}')
        return response
    
    def send_fibonacci_goal(self):
        """Send a goal to the Fibonacci action server"""
        goal_msg = Fibonacci.Goal()
        goal_msg.order = 10
        
        # Wait for action server
        self.action_client.wait_for_server()
        
        # Send goal
        self._send_goal_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        """Handle response from action server"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback from action server"""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        """Handle result from action server"""
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        # Publish result as a status
        result_msg = String()
        result_msg.data = f'Fibonacci result: {result.sequence}'
        self.status_publisher.publish(result_msg)


def main(args=None):
    rclpy.init(args=args)
    
    # Create node
    comm_node = RobotCommunicationNode()
    
    # Use multi-threaded executor to handle multiple callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(comm_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        comm_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Simulation Demonstration

This node demonstrates all three communication patterns in a single application. It publishes status updates as topics, responds to calculation requests via services, and manages long-running Fibonacci calculations via actions. The code can be run in a simulation environment to see how these different communication patterns interact.

## Hands-On Lab: Implementing All Communication Patterns

In this lab, you'll implement a node that uses all three communication patterns:

1. Create a node with publishers and subscribers
2. Add a service server and client
3. Implement an action server and client
4. Test the communication patterns in simulation

### Required Equipment:
- ROS 2 Humble environment
- Python development environment
- (Optional) Gazebo simulation environment

### Instructions:
1. Create a new ROS 2 package for the lab
2. Implement the RobotCommunicationNode as shown in the code example
3. Create a separate service server if needed: `ros2 run example_interfaces add_two_ints_server`
4. Create a separate action server for Fibonacci: `ros2 run example_interfaces fibonacci_action_server`
5. Run the communication node: `ros2 run [your_package] robot_communication_node`
6. Test the different communication patterns:
   - Send a command via topic: `ros2 topic pub /robot_commands std_msgs/String "data: 'calculate_fibonacci'"`
   - Send a service request: `ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 2, b: 3}"`
   - Observe the status publications: `ros2 topic echo /robot_status std_msgs/String`
7. Analyze how each communication pattern serves different purposes

## Common Pitfalls & Debugging Notes

- **Quality of Service Mismatch**: Publishers and subscribers must have compatible QoS settings to communicate
- **Node Discovery**: In distributed systems, ensure nodes can discover each other across the network
- **Threading Issues**: Use appropriate callback groups (ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup) for multi-threaded scenarios
- **Action Preemption**: Consider how to handle goal preemption in action servers
- **Message Compatibility**: Ensure all nodes use compatible message definitions

## Summary & Key Terms

**Key Terms:**
- **Publish-Subscribe**: Asynchronous communication pattern where publishers send messages to topics and subscribers receive them
- **Request-Response**: Synchronous communication pattern where clients send requests to services and receive responses
- **Action**: Asynchronous communication pattern for long-running tasks with feedback and cancelation
- **Quality of Service (QoS)**: Settings that govern ROS 2 communication behavior
- **Callback Groups**: Mechanisms to control how callbacks are executed in multi-threaded scenarios
- **Action Server**: The node that executes action goals
- **Action Client**: The node that sends action goals

## Further Reading & Citations

1. ROS 2 Documentation. (2023). "Tutorials: Understanding Quality of Service." https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Quality-Of-Service.html
2. ROS 2 Documentation. (2023). "Concepts: About Communications." https://docs.ros.org/en/humble/Concepts/About-Topics.html
3. Kamran, M. A. (2021). ROS Robotics Projects. Packt Publishing.
4. Jung, J. (2019). ROS Robot Programming. CreateSpace.

## Assessment Questions

1. Explain the differences between topics, services, and actions in ROS 2. Provide an example of when you would use each communication pattern.
2. What are Quality of Service (QoS) settings in ROS 2? Why are they important?
3. Describe the role of callback groups in ROS 2 multi-threaded applications.
4. How does the action communication pattern differ from the service pattern in terms of execution and feedback?
5. What are some common debugging strategies for ROS 2 communication issues?

---
**Previous**: [Introduction to ROS 2](./intro.md)  
**Next**: [URDF Modeling for Robotics](./urdf.md)