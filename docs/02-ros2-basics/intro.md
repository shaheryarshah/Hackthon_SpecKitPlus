# Introduction to ROS 2 - The Robotic Nervous System

The Robot Operating System 2 (ROS 2) serves as the nervous system for robotic applications, providing a framework for writing robot software. Unlike a traditional operating system, ROS 2 is a collection of tools, libraries, and conventions that aim to simplify the development of complex robotic applications by providing hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more.

## Learning Outcomes

After completing this chapter, you should be able to:
- Explain the purpose and architecture of ROS 2
- Create and run basic ROS 2 nodes
- Understand the communication mechanisms: topics, services, and actions
- Implement publisher-subscriber communication patterns
- Use ROS 2 tools for introspection and debugging

## Core Concepts

### Nodes

A node is a process that performs computation. ROS 2 is designed with a distributed architecture where computation is decomposed into nodes that are distributed among different devices. Nodes can be thought of as mini-programs that work together to perform complex robotic tasks.

### Topics and Message Passing

Topics enable asynchronous communication between nodes through a publish-subscribe model. A node publishes messages to a topic, and any number of other nodes can subscribe to that topic to receive the messages. This decouples the publisher and subscriber both in time and space, allowing for flexible and modular robotic applications.

### Services

Services enable synchronous request-response communication between nodes. A client sends a request to a service server, which then processes the request and returns a response. This is useful for operations that require a definite result before proceeding.

### Actions

Actions are designed for long-running tasks that may take a significant amount of time to complete. They provide feedback during execution and allow for goal preemption. Actions are ideal for navigation tasks, trajectory execution, or any process where you need to know the progress of the task.

## Equations and Models

ROS 2 communication can be modeled as:

```
Publisher(p_i) → Topic(t_j) → Subscriber(s_k)
```

Where:
- `p_i` is the i-th publisher
- `t_j` is the j-th topic
- `s_k` is the k-th subscriber

The message passing is governed by the quality of service (QoS) policies which determine reliability, durability, and other communication characteristics.

## Code Example: Basic Publisher/Subscriber

Here's an example of a simple publisher and subscriber in ROS 2 using Python:

```python
# publisher_member_function.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    
    try:
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```python
# subscriber_member_function.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    
    try:
        rclpy.spin(minimal_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Simulation Demonstration

These nodes can be run in a ROS 2 environment (like with a simulated robot) to demonstrate the publish-subscribe pattern. The publisher will send messages to the topic at regular intervals, and the subscriber will receive and log these messages.

## Hands-On Lab: Creating a ROS 2 Publisher/Subscriber System

In this lab, you'll create and run your own publisher/subscriber system:

1. Create a new ROS 2 package
2. Implement the publisher and subscriber nodes
3. Run the nodes and observe the communication
4. Modify the message content and timing

### Required Equipment:
- ROS 2 Humble environment
- Python development environment

### Instructions:
1. Create a new ROS 2 package: `ros2 pkg create --build-type ament_python my_publisher_subscriber`
2. Create the publisher script in `my_publisher_subscriber/my_publisher_subscriber/publisher_member_function.py`
3. Create the subscriber script in `my_publisher_subscriber/my_publisher_subscriber/subscriber_member_function.py`
4. Create a setup.py file to make the scripts executable
5. Build the package: `colcon build --packages-select my_publisher_subscriber`
6. Source the environment: `source install/setup.bash`
7. Run the publisher in one terminal: `ros2 run my_publisher_subscriber publisher_member_function`
8. Run the subscriber in another terminal: `ros2 run my_publisher_subscriber subscriber_member_function`
9. Observe the communication between nodes
10. Modify the message content or timing and observe the changes

## Common Pitfalls & Debugging Notes

- **Node Names**: Ensure unique node names to avoid conflicts
- **Topic Names**: Verify that publisher and subscriber use the same topic names
- **Message Types**: Publisher and subscriber must use the same message type
- **Timing**: In distributed systems, timing can be unpredictable; design systems to be robust to timing variations
- **Resource Management**: Always properly clean up resources in the destroy_node() method

## Summary & Key Terms

**Key Terms:**
- **Node**: A process that performs computation in ROS 2
- **Topic**: A named bus over which nodes exchange messages
- **Publisher**: A node that sends messages to a topic
- **Subscriber**: A node that receives messages from a topic
- **Message**: A data structure that communicates information between nodes
- **Service**: A synchronous request-response communication pattern
- **Action**: An asynchronous communication pattern for long-running tasks
- **ROS 2**: The Robot Operating System version 2
- **rclpy**: Python client library for ROS 2

## Further Reading & Citations

1. ROS 2 Documentation. (2023). "Concepts: About ROS 2." https://docs.ros.org/en/humble/Concepts/About-ROS-2.html
2. Quigley, M., Conley, K., & Gerkey, B. (2009). ROS: an open-source Robot Operating System. ICRA Workshop on Open Source Software, 3(3.2), 5.
3. Macenski, S. (2022). Professional Robotics: ROS and ROS2 Development. Apress.
4. Saldanha, D. (2021). ROS Robotics Projects. Packt Publishing.

## Assessment Questions

1. Define a ROS 2 node and explain its role in the ROS 2 architecture.
2. Compare and contrast the publish-subscribe and request-response communication patterns in ROS 2.
3. What are actions in ROS 2, and when would you use them instead of services or topics?
4. Describe the Quality of Service (QoS) settings in ROS 2 and their importance.
5. Explain the difference between topics and services in terms of communication timing and use cases.

---
**Previous**: [Sensors and Physical Perception](../01-physical-ai/sensors-perception.md)  
**Next**: [ROS 2 Nodes, Topics, Services, and Actions](./nodes-topics.md)