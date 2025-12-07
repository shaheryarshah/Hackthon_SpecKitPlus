# Glossary of Robotics Terms

This glossary provides definitions for key terms used throughout the Physical AI & Humanoid Robotics textbook. Terms are organized alphabetically for easy reference.

## A

**Actuator**: A mechanical device that converts energy (electrical, hydraulic, or pneumatic) into motion. Common types include servomotors, stepper motors, and hydraulic cylinders.

**Affordance**: The possibility of action offered by an object or environment to an agent. In robotics, it refers to what actions are possible with or on an object.

**Artificial Intelligence (AI)**: The simulation of human intelligence processes by machines, especially computer systems. In robotics, AI enables perception, decision-making, and learning capabilities.

**Autonomous Robot**: A robot that can operate independently without human intervention, making decisions based on its sensors and programming.

## B

**Behavior-Based Robotics**: An approach to robotics that structures robot control around collections of task-oriented behaviors that operate in parallel.

**Bipedal Locomotion**: The act of walking on two legs, a key challenge in humanoid robotics involving balance and gait planning.

**Body Schema**: A representation of the body and its parts that allows a robot to understand its configuration and plan movements accordingly.

## C

**Center of Mass (CoM)**: The point where the total mass of a body may be assumed to be concentrated for the purpose of analyzing motion and balance.

**Collision Detection**: The computational problem of detecting the intersection of two or more objects, crucial for robot safety and navigation.

**Compliance Control**: A control approach that allows robots to adapt to environmental forces by adjusting their position or force output.

**Computer Vision**: A field of artificial intelligence that trains computers to interpret and understand visual information from the world.

**Control Loop**: A feedback mechanism where a system continuously measures its output, compares it to a desired reference, and adjusts its input accordingly.

**Covariance**: A measure of how much two random variables change together, used in robotics for uncertainty representation.

## D

**Deep Learning**: A subset of machine learning that uses neural networks with multiple layers to extract features and make decisions.

**Degrees of Freedom (DOF)**: The number of independent movements or parameters that define the configuration of a mechanical system.

**Dynamics**: The study of forces and torques and their effect on motion. Robot dynamics models how forces cause movement.

**Dynamic Movement Primitives (DMP)**: A mathematical framework for representing and generating movements, useful for learning and reproducing robot behaviors.

## E

**Embodied Artificial Intelligence**: AI systems that interact with the physical world through a body, emphasizing the role of physical interaction in intelligence.

**Embodied Cognition**: The theory that cognitive processes are influenced by the body's interactions with the environment.

**End Effector**: The device at the end of a robot arm that interacts with the environment, such as a gripper or tool.

**Epuck**: A small mobile robot platform commonly used in education and research.

**Ethics in Robotics**: The study of moral issues in the design, manufacture, use, and treatment of robots.

## F

**Forward Kinematics**: The process of determining the position and orientation of the end effector given the joint angles of a robot.

**Forward Model**: A predictive model that estimates the sensory consequences of motor commands, used for control and learning.

**Friction Compensation**: Techniques to account for and counteract friction forces in robot control systems.

## G

**Gazebo**: A 3D simulation environment for robotics that provides accurate physics simulation and rendering capabilities.

**Gaussian Noise**: Statistical noise having a probability density function equal to that of the normal distribution, commonly used to model sensor noise.

**Gripper**: A device used by robots to grasp and hold objects, analogous to a human hand.

**Ground Truth**: The actual state of the world, often used as a reference for evaluating robot perception and localization.

## H

**Haptic Feedback**: The use of touch and motion feedback to interact with virtual or remote environments.

**Hardware-in-the-Loop (HIL)**: A testing method that involves physical components in a simulated environment.

**Human-Robot Interaction (HRI)**: The study of interactions between humans and robots, including communication, collaboration, and trust.

**Humanoid Robot**: A robot with a physical structure resembling the human body, typically featuring a head, torso, two arms, and two legs.

## I

**Inertial Measurement Unit (IMU)**: An electronic device that measures and reports a body's specific force, angular rate, and sometimes the magnetic field surrounding the body.

**Inverse Kinematics**: The process of determining the joint angles required to achieve a desired end-effector position and orientation.

**Inverse Dynamics**: The computation of forces and torques required to achieve a desired motion, based on the robot's dynamics model.

**Isaac Sim**: NVIDIA's robotics simulation application built on the Omniverse platform, designed for photorealistic simulation.

**Isaac ROS**: A collection of hardware-accelerated perception and navigation packages for ROS 2, optimized for NVIDIA GPUs.

## J

**Jacobian Matrix**: A matrix that contains first-order partial derivatives of a vector-valued function, used in robotics to relate joint velocities to end-effector velocities.

**Joint Space**: The space defined by the robot's joint angles, as opposed to Cartesian space.

## K

**Kinematics**: The branch of mechanics concerned with the motion of objects without reference to the forces that cause the motion.

**Kinodynamic Planning**: Motion planning that considers both kinematic and dynamic constraints of a robot.

## L

**LIDAR (Light Detection and Ranging)**: A remote sensing method that uses light in the form of a pulsed laser to measure distances.

**Linear Inverted Pendulum Model (LIPM)**: A simplified model for bipedal walking that represents the robot's center of mass as a point mass on a massless rod.

**Localization**: The process of determining the robot's position and orientation in a known or unknown environment.

**Low-level Control**: Control systems that manage the direct actuation of robot joints or motors, typically running at high frequencies.

## M

**Manipulation**: The ability of a robot to purposefully control objects in its environment using its end effectors.

**Mapping**: The process of creating a representation of the environment, typically for navigation or interaction purposes.

**Mobile Robot**: A robot that is capable of locomotion through its environment.

**Motion Planning**: The process of determining a sequence of movements to achieve a goal while avoiding obstacles.

## N

**Navigation**: The process of planning and executing robot motion to move from one location to another.

**Neural Network**: A computing system inspired by the human brain, used in robotics for perception, control, and learning.

**Non-holonomic Constraint**: A constraint that restricts the direction of motion, common in wheeled robots that cannot move sideways.

## O

**Obstacle Avoidance**: The capability of a robot to detect and avoid obstacles in its environment.

**Occupancy Grid**: A probabilistic model of the environment represented as a grid of cells indicating the probability of occupancy.

**Odometry**: The use of data from motion sensors to estimate change in position over time.

**Open Source Robotics Foundation (OSRF)**: The organization that develops and maintains Gazebo and other open-source robotics software.

## P

**Path Planning**: The process of finding a sequence of positions that connect an initial state to a goal state.

**Perception**: The ability of a robot to interpret and understand its environment through sensors.

**Physical Artificial Intelligence**: AI systems that interact with the physical world, emphasizing embodied interaction and real-world tasks.

**PID Controller**: A control loop feedback mechanism that calculates an error value as the difference between desired and measured values.

**Point Cloud**: A set of data points in space, typically representing the external surface of an object, used in 3D perception.

**Probabilistic Robotics**: An approach to robotics that explicitly represents and handles uncertainty in robot perception and action.

## R

**Range Sensor**: A sensor that measures distances to objects, such as LIDAR, sonar, or structured light systems.

**Reachability Analysis**: The process of determining which positions a robot's end effector can reach.

**Real-Time Control**: Control systems that must respond to sensor inputs and update actuator commands within strict timing constraints.

**Reinforcement Learning**: A type of machine learning where agents learn to make decisions by performing actions and receiving rewards or penalties.

**Robot Operating System (ROS)**: A flexible framework for writing robot software that provides services for hardware abstraction, device drivers, and message passing.

**Robotics Middleware**: Software infrastructure that provides services to robotic applications, such as communication, resource management, and device abstraction.

**ROS 2**: The second generation of the Robot Operating System, designed for production robotics applications.

## S

**Sensor Fusion**: The process of combining data from multiple sensors to improve the accuracy and reliability of information.

**Sim-to-Real Transfer**: The application of policies trained in simulation to real-world robotic systems.

**SLAM (Simultaneous Localization and Mapping)**: The computational problem of constructing or updating a map of an unknown environment while simultaneously keeping track of an agent's location.

**State Estimation**: The process of estimating the internal state of a system from sensor measurements and control inputs.

**Stereo Vision**: A method of determining distance by comparing images from two cameras, mimicking human binocular vision.

**System Identification**: The process of developing mathematical models of dynamic systems from measured data.

## T

**Task Planning**: The process of decomposing high-level goals into sequences of actions that achieve the goals.

**Trajectory**: A path through space with timing information, specifying how to move from one configuration to another.

**Trajectory Optimization**: The process of finding an optimal path or trajectory for a robot to follow.

**Tactile Sensor**: A sensor that measures information obtained by touch, important for robotic manipulation.

## U

**Unity**: A 3D development platform that can be used for robotics visualization and simulation.

**Universal Robot (UR)**: A series of collaborative robots that can work alongside humans.

**URDF (Unified Robot Description Format)**: An XML format for representing robot models, including kinematic and dynamic properties.

## V

**Velodyne**: A company that produces LIDAR sensors widely used in robotics and autonomous vehicles.

**Vision System**: A robot system that uses cameras and computer vision algorithms for perception and navigation.

**Vision-Language-Action (VLA)**: An integrated approach combining visual perception, language understanding, and robotic action execution.

**Virtual Reality (VR)**: A simulated experience that can be similar to or completely different from the real world, sometimes used in robot teleoperation.

## W

**Wheel Odometry**: Odometry based on measuring the rotation of wheels to estimate motion.

**Whole-Body Control**: A control approach that simultaneously considers all the robot's degrees of freedom for coordination and balance.

**Workspace**: The volume of space that a robot manipulator can reach with its end effector.

## Y

**Yaw**: The rotation of a robot around its vertical axis, one of the three primary orientations (roll, pitch, yaw).

## Z

**Zero Moment Point (ZMP)**: A critical concept in bipedal locomotion representing the point where the net moment of the ground reaction forces is zero.