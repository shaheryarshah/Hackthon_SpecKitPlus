# URDF Modeling for Robotics

The Unified Robot Description Format (URDF) is an XML-based format used in ROS to describe robot models. This includes aspects of the robot such as kinematic and dynamic descriptions, visual and collision properties, and sensor mounting positions. Understanding URDF is essential for simulating robots, planning motions, and controlling robot hardware.

## Learning Outcomes

After completing this section, you should be able to:
- Understand the structure and components of URDF files
- Create basic URDF models for simple robots
- Define joints, links, and materials in URDF
- Use Xacro to simplify complex URDF models
- Validate and visualize URDF models in RViz

## Core Concepts

### Links

A link is a rigid body with mass, visual representation, and collision properties. In URDF, a link contains:

- **Inertial properties**: Mass, center of mass, and moment of inertia
- **Visual properties**: How the link appears in simulation and visualization
- **Collision properties**: How the link interacts with other objects in physics simulation

### Joints

Joints connect links and define how they can move relative to each other. The joint types in URDF include:

- **Fixed**: No movement allowed between the connected links
- **Continuous**: Continuous rotation around the joint axis (like a wheel)
- **Revolute**: Limited rotation around the joint axis (like an elbow)
- **Prismatic**: Linear sliding motion along the joint axis
- **Floating**: 6 degrees of freedom (rarely used)
- **Planar**: Motion constrained to a plane (rarely used)

### Transmissions

Transmissions define the relationship between an actuator (like a motor) and a joint, including the mechanical reduction and other physical properties.

## Equations and Models

The kinematic chain of a robot can be described using the Denavit-Hartenberg (DH) parameters:

```
T(i-1,i) = Rz(θi) * Tz(d_i) * Tx(a_i) * Rx(αi)
```

Where:
- `T(i-1,i)` is the transformation matrix between link i-1 and link i
- `θi`, `d_i`, `a_i`, and `αi` are the DH parameters
- R and T represent rotation and translation matrices respectively

For a simple revolute joint, the transformation includes the joint angle as a variable parameter.

## Code Example: Simple URDF Model

Here's an example of a simple differential drive robot in URDF:

```xml
<?xml version="1.0" ?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="1.0" />
      <origin xyz="0 0 0" />
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01" />
    </inertial>
    
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0" />
      <geometry>
        <box size="0.5 0.3 0.2" />
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1" />
      </material>
    </visual>
    
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0" />
      <geometry>
        <box size="0.5 0.3 0.2" />
      </geometry>
    </collision>
  </link>

  <!-- Left wheel -->
  <link name="left_wheel">
    <inertial>
      <mass value="0.2" />
      <origin xyz="0 0 0" />
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001" />
    </inertial>
    
    <visual>
      <origin xyz="0 0 0" rpy="1.5707963267948966 0 0" />
      <geometry>
        <cylinder radius="0.1" length="0.05" />
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1" />
      </material>
    </visual>
    
    <collision>
      <origin xyz="0 0 0" rpy="1.5707963267948966 0 0" />
      <geometry>
        <cylinder radius="0.1" length="0.05" />
      </geometry>
    </collision>
  </link>

  <!-- Right wheel -->
  <link name="right_wheel">
    <inertial>
      <mass value="0.2" />
      <origin xyz="0 0 0" />
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001" />
    </inertial>
    
    <visual>
      <origin xyz="0 0 0" rpy="1.5707963267948966 0 0" />
      <geometry>
        <cylinder radius="0.1" length="0.05" />
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1" />
      </material>
    </visual>
    
    <collision>
      <origin xyz="0 0 0" rpy="1.5707963267948966 0 0" />
      <geometry>
        <cylinder radius="0.1" length="0.05" />
      </geometry>
    </collision>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link" />
    <child link="left_wheel" />
    <origin xyz="0 0.15 0" rpy="0 0 0" />
    <axis xyz="0 1 0" />
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link" />
    <child link="right_wheel" />
    <origin xyz="0 -0.15 0" rpy="0 0 0" />
    <axis xyz="0 1 0" />
  </joint>

  <!-- Gazebo plugin for differential drive -->
  <gazebo>
    <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.3</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
    </plugin>
  </gazebo>

</robot>
```

## Using Xacro for Complex Models

Xacro is an XML macro language that allows you to create more complex and maintainable URDF models. Here's an example using Xacro:

```xml
<?xml version="1.0" ?>
<robot name="xacro_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Define properties -->
  <xacro:property name="base_width" value="0.5" />
  <xacro:property name="base_length" value="0.3" />
  <xacro:property name="base_height" value="0.2" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />
  <xacro:property name="wheel_yoffset" value="0.15" />

  <!-- Base macro -->
  <xacro:macro name="base">
    <link name="base_link">
      <inertial>
        <mass value="1.0" />
        <origin xyz="0 0 0" />
        <inertia 
          ixx="0.01" ixy="0.0" ixz="0.0" 
          iyy="0.01" iyz="0.0" izz="0.01" />
      </inertial>
      
      <visual>
        <origin xyz="0 0 ${base_height/2}" rpy="0 0 0" />
        <geometry>
          <box size="${base_width} ${base_length} ${base_height}" />
        </geometry>
        <material name="blue">
          <color rgba="0 0 1 1" />
        </material>
      </visual>
      
      <collision>
        <origin xyz="0 0 ${base_height/2}" rpy="0 0 0" />
        <geometry>
          <box size="${base_width} ${base_length} ${base_height}" />
        </geometry>
      </collision>
    </link>
  </xacro:macro>

  <!-- Wheel macro -->
  <xacro:macro name="wheel" params="prefix x_reflect y_reflect">
    <link name="${prefix}_wheel">
      <inertial>
        <mass value="0.2" />
        <origin xyz="0 0 0" />
        <inertia 
          ixx="0.001" ixy="0.0" ixz="0.0" 
          iyy="0.001" iyz="0.0" izz="0.001" />
      </inertial>
      
      <visual>
        <origin xyz="0 0 0" rpy="${pi/2 * x_reflect} 0 0" />
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}" />
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1" />
        </material>
      </visual>
      
      <collision>
        <origin xyz="0 0 0" rpy="${pi/2 * x_reflect} 0 0" />
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}" />
        </geometry>
      </collision>
    </link>
  </xacro:macro>

  <!-- Wheel joint macro -->
  <xacro:macro name="wheel_joint" params="prefix y_offset">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="base_link" />
      <child link="${prefix}_wheel" />
      <origin xyz="0 ${y_offset} 0" rpy="0 0 0" />
      <axis xyz="0 1 0" />
    </joint>
  </xacro:macro>

  <!-- Instantiate the robot -->
  <xacro:base />
  <xacro:wheel prefix="left" x_reflect="1" y_reflect="1" />
  <xacro:wheel prefix="right" x_reflect="1" y_reflect="-1" />
  <xacro:wheel_joint prefix="left" y_offset="${wheel_yoffset}" />
  <xacro:wheel_joint prefix="right" y_offset="${-wheel_yoffset}" />

</robot>
```

## Simulation Demonstration

This URDF model can be loaded into simulation environments like Gazebo to test robot behavior. The model includes both visual properties for rendering and collision properties for physics simulation. The Gazebo plugin enables differential drive control through ROS 2 topics.

## Hands-On Lab: Building Your Own URDF Robot

In this lab, you'll create and visualize your own simple robot model:

1. Create a URDF file for a simple robot
2. Visualize the robot in RViz
3. Load the robot into Gazebo simulation
4. Test the robot's kinematic properties

### Required Equipment:
- ROS 2 Humble environment
- RViz2 visualization tool
- Gazebo simulation environment (optional)

### Instructions:
1. Create a new package for your robot model: `ros2 pkg create --build-type ament_cmake my_robot_description`
2. Create a URDF file in `my_robot_description/urdf/` directory
3. Add the URDF content as shown in the examples above
4. Create a launch file to load the robot model in RViz
5. Launch RViz and visualize your robot: `ros2 launch my_robot_description view_robot.launch.py`
6. Check that all links and joints are displayed correctly
7. (Optional) Test in Gazebo by creating a world file and spawning your robot

## Common Pitfalls & Debugging Notes

- **Units**: URDF expects SI units (meters, kilograms, seconds)
- **Mass Values**: Make sure mass values are positive and realistic
- **Inertia Values**: Ensure inertia tensors are physically valid
- **Joint Limits**: Define reasonable joint limits to avoid simulation issues
- **Meshes**: When using mesh files, ensure the file paths are correct and the meshes are in proper units
- **Xacro Processing**: Remember to process Xacro files to URDF: `xacro input.xacro > output.urdf`

## Summary & Key Terms

**Key Terms:**
- **URDF**: Unified Robot Description Format, an XML-based robot description format
- **Link**: A rigid body in a robot model with mass and geometric properties
- **Joint**: A connection between two links that defines how they can move relative to each other
- **Xacro**: An XML macro language used to simplify and maintain URDF files
- **Inertial Properties**: Mass, center of mass, and moment of inertia of a link
- **Collision Properties**: How a link interacts with other objects in physics simulation
- **Visual Properties**: How a link appears in simulation and visualization
- **DH Parameters**: Denavit-Hartenberg parameters for kinematic modeling

## Further Reading & Citations

1. Smart, W. D., & Pasteur, C. (2017). Robot Modeling with URDF. In ROS Robot Programming (pp. 57-84).
2. ROS Documentation. (2023). "URDF Tutorials." http://wiki.ros.org/urdf/Tutorials
3. ROS Documentation. (2023). "Xacro." http://wiki.ros.org/xacro
4. Corke, P. (2017). Robotics, Vision and Control: Fundamental Algorithms in MATLAB (2nd ed.). Springer.

## Assessment Questions

1. Explain the difference between visual and collision properties in URDF. Why might they be different?
2. List the different joint types available in URDF and provide an example of where each would be used.
3. Describe the purpose of the inertial properties in a URDF link.
4. What are the advantages of using Xacro over plain URDF for complex robot models?
5. How would you validate that your URDF model is correct before using it in simulation?

---
**Previous**: [ROS 2 Nodes, Topics, Services, and Actions](./nodes-topics.md)  
**Next**: [rclpy Python Client Library](./rclpy.md)