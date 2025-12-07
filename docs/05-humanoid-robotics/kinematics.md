# Humanoid Kinematics and Dynamics

Humanoid kinematics and dynamics form the mathematical foundation for understanding and controlling the motion of humanoid robots. Kinematics deals with the geometric relationships between joints and end-effectors without considering forces, while dynamics includes the effects of forces, torques, masses, and accelerations. This section covers both forward and inverse kinematics as well as dynamic modeling approaches for humanoid robots.

## Learning Outcomes

After completing this section, you should be able to:
- Calculate forward and inverse kinematics for humanoid robot arms and legs
- Understand the Denavit-Hartenberg (DH) parameters for kinematic modeling
- Apply the Jacobian matrix to relate joint velocities to end-effector velocities
- Analyze the dynamics of humanoid robots using Lagrange equations
- Implement numerical methods for inverse kinematics solutions
- Evaluate the stability and workspace of humanoid robot mechanisms

## Core Concepts

### Forward Kinematics
Forward kinematics calculates the position and orientation of the end-effector given the joint angles. For a humanoid robot, this is essential for knowing where each part of the robot is in space.

### Inverse Kinematics
Inverse kinematics determines the required joint angles to achieve a desired end-effector position and orientation. This is crucial for motion planning and task execution.

### Workspace Analysis
Understanding the reachable workspace of humanoid limbs is important for task planning and determining if a robot can perform a required action.

### Jacobian Matrix
The Jacobian matrix relates joint space velocities to Cartesian space velocities. It's crucial for motion control, singularity analysis, and force control.

### Dynamics Modeling
Dynamics modeling involves understanding how forces and torques affect the motion of the robot. For humanoid robots, this is essential for balance control and dynamic movements.

## Equations and Models

### Forward Kinematics with Homogeneous Transformations

For an n-DOF robot, the forward kinematics is calculated as:

```
T_0^n = A_1(θ_1) * A_2(θ_2) * ... * A_n(θ_n)
```

Where `T_0^n` is the transformation matrix from the base frame to the end-effector frame, and `A_i(θ_i)` is the transformation matrix for joint i.

### Denavit-Hartenberg Transformation

The DH transformation matrix for joint i is:

```
     ⎡ cos(θ_i)   -sin(θ_i)*cos(α_i)   sin(θ_i)*sin(α_i)   a_i*cos(θ_i) ⎤
A_i = ⎢ sin(θ_i)    cos(θ_i)*cos(α_i)  -cos(θ_i)*sin(α_i)   a_i*sin(θ_i) ⎥
      ⎢    0           sin(α_i)            cos(α_i)            d_i     ⎥
      ⎣    0              0                  0                1       ⎦
```

Where θ_i, d_i, a_i, and α_i are the DH parameters.

### Inverse Kinematics Optimization

The inverse kinematics problem can be formulated as:

```
min_θ ||f(θ) - x_d||²
```

Where `f(θ)` is the forward kinematics function, `x_d` is the desired end-effector pose, and `θ` is the vector of joint angles.

### Jacobian Matrix

The geometric Jacobian is defined as:

```
J(θ) = ∂f(θ)/∂θ
```

For a 6-DOF manipulator, this results in a 6×n matrix (n = number of joints):

```
J = [J_v ]  (6×n)
    [J_ω ]
```

Where `J_v` is the linear velocity Jacobian and `J_ω` is the angular velocity Jacobian.

### Lagrange-Euler Dynamics

The dynamic equation for a robotic manipulator is:

```
M(q)q̈ + C(q, q̇)q̇ + G(q) = τ
```

Where:
- `M(q)` is the inertia matrix
- `C(q, q̇)` contains centrifugal and Coriolis terms
- `G(q)` is the gravity vector
- `τ` is the vector of joint torques
- `q`, `q̇`, `q̈` are joint position, velocity, and acceleration vectors

## Code Example: Advanced Humanoid Kinematics and Dynamics

Here's an example implementation of advanced kinematics and dynamics for a humanoid robot:

```python
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from scipy.linalg import block_diag


class HumanoidKinematicsDynamics:
    """
    Advanced kinematics and dynamics for a humanoid robot
    """
    def __init__(self, robot_config):
        """
        Initialize the humanoid model with robot configuration
        
        :param robot_config: Dictionary containing link lengths, joint limits, and DH parameters
        """
        self.config = robot_config
        self.num_joints = len(robot_config['dh_params'])
        
        # Extract DH parameters
        self.dh_params = robot_config['dh_params']  # List of [a, alpha, d, theta_offset]
        self.joint_limits = robot_config.get('joint_limits', [(-np.pi, np.pi)] * self.num_joints)
        
        # Link properties for dynamics
        self.link_masses = robot_config.get('link_masses', [1.0] * self.num_joints)
        self.link_com = robot_config.get('link_com', [[0, 0, 0]] * self.num_joints)  # Center of mass in link frame
        self.link_inertia = robot_config.get('link_inertia', [np.eye(3)] * self.num_joints)  # Inertia tensor in COM frame
        
    def dh_transform(self, a, alpha, d, theta):
        """
        Calculate the Denavit-Hartenberg transformation matrix
        """
        return np.array([
            [math.cos(theta), -math.sin(theta)*math.cos(alpha),  math.sin(theta)*math.sin(alpha), a*math.cos(theta)],
            [math.sin(theta),  math.cos(theta)*math.cos(alpha), -math.cos(theta)*math.sin(alpha), a*math.sin(theta)],
            [0,               math.sin(alpha),                   math.cos(alpha),               d],
            [0,               0,                                 0,                             1]
        ])
    
    def forward_kinematics(self, joint_angles, link_index=None):
        """
        Calculate forward kinematics for the entire chain or up to a specific link
        
        :param joint_angles: List of joint angles
        :param link_index: Index of the link to calculate FK for (None for end-effector)
        :return: 4x4 transformation matrix
        """
        if link_index is None:
            link_index = len(self.dh_params)
        
        T = np.eye(4)  # Identity transformation
        
        for i in range(link_index):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset
            
            A_i = self.dh_transform(a, alpha, d, theta)
            T = T @ A_i
        
        return T
    
    def get_end_effector_pose(self, joint_angles):
        """
        Get the position and orientation of the end-effector
        """
        T = self.forward_kinematics(joint_angles)
        position = T[:3, 3]
        orientation = R.from_matrix(T[:3, :3]).as_quat()
        return position, orientation
    
    def jacobian(self, joint_angles, point_index=6):
        """
        Calculate the geometric Jacobian matrix for a 6-DOF robot
        
        :param joint_angles: List of joint angles
        :param point_index: Index of the point to calculate Jacobian for (default: end-effector)
        :return: 6xN Jacobian matrix (linear and angular velocities)
        """
        num_joints = len(joint_angles)
        J = np.zeros((6, num_joints))
        
        # Get end-effector transform
        T_end = self.forward_kinematics(joint_angles)
        p_end = T_end[:3, 3]  # End-effector position
        
        # Calculate Jacobian columns for each joint
        for i in range(num_joints):
            # Transform of joint i
            T_i = self.forward_kinematics(joint_angles, i)
            p_i = T_i[:3, 3]  # Position of joint i
            
            # Z-axis of joint i in base frame
            z_i = T_i[:3, 2]
            
            # Calculate Jacobian column
            J[:3, i] = np.cross(z_i, (p_end - p_i))  # Linear velocity part
            J[3:, i] = z_i  # Angular velocity part
        
        return J
    
    def inverse_kinematics(self, target_pos, target_rot, current_angles, max_iter=100, tolerance=1e-4):
        """
        Solve inverse kinematics using iterative method (Jacobian transpose/pseudoinverse)
        
        :param target_pos: Target position [x, y, z]
        :param target_rot: Target rotation as quaternion [x, y, z, w]
        :param current_angles: Initial joint configuration
        :param max_iter: Maximum iterations
        :param tolerance: Position tolerance
        :return: Joint angles that reach the target
        """
        # Convert target rotation to matrix
        target_rot_matrix = R.from_quat(target_rot).as_matrix()
        
        # Create target transformation matrix
        target_transform = np.eye(4)
        target_transform[:3, 3] = target_pos
        target_transform[:3, :3] = target_rot_matrix
        
        q = np.array(current_angles, dtype=float)
        
        for iteration in range(max_iter):
            # Calculate current end-effector pose
            current_transform = self.forward_kinematics(q)
            
            # Calculate pose error
            pos_error = target_pos - current_transform[:3, 3]
            
            # For rotation error, use simple approach with angle-axis representation
            R_current = current_transform[:3, :3]
            R_target = target_transform[:3, :3]
            R_error = R_target @ R_current.T
            angle, axis = self.rotation_matrix_to_angle_axis(R_error)
            rot_error = angle * axis
            
            # Combine position and rotation errors
            error = np.concatenate([pos_error, rot_error])
            
            # Check if within tolerance
            if np.linalg.norm(pos_error) < tolerance:
                break
            
            # Calculate Jacobian
            J = self.jacobian(q)
            
            # Use damped least squares for better stability
            damping = 0.01
            JJT = J @ J.T
            I = np.eye(6)
            J_pinv = J.T @ np.linalg.inv(JJT + damping**2 * I)
            
            # Update joint angles
            dq = J_pinv @ error
            q = q + 0.1 * dq  # Use step size to prevent overshooting
            
            # Apply joint limits
            for i in range(len(q)):
                q[i] = np.clip(q[i], self.joint_limits[i][0], self.joint_limits[i][1])
        
        return q.tolist()
    
    def rotation_matrix_to_angle_axis(self, R_matrix):
        """
        Convert rotation matrix to angle-axis representation
        """
        # Calculate angle
        angle = math.acos(np.clip((np.trace(R_matrix) - 1) / 2, -1, 1))
        
        # Calculate axis (avoid division by zero)
        if angle < 1e-6:
            return 0.0, np.array([1, 0, 0])
        
        # Calculate axis
        axis = np.array([
            R_matrix[2, 1] - R_matrix[1, 2],
            R_matrix[0, 2] - R_matrix[2, 0],
            R_matrix[1, 0] - R_matrix[0, 1]
        ])
        axis = axis / (2 * math.sin(angle))
        axis = axis / np.linalg.norm(axis)  # Normalize
        
        return angle, axis
    
    def compute_inertia_matrix(self, joint_angles):
        """
        Compute the joint space inertia matrix M(q) using the Composite Rigid Body Algorithm
        Simplified implementation for demonstration
        """
        n = len(joint_angles)
        M = np.zeros((n, n))
        
        # For a simplified implementation, we'll use an approximation
        # In a real implementation, this would use the composite rigid body algorithm
        for i in range(n):
            # Approximate diagonal terms with link masses and positions
            T_i = self.forward_kinematics(joint_angles, i+1)
            p_i = T_i[:3, 3]  # Position of link i
            
            # Simplified inertia contribution (not physically accurate but demonstrates concept)
            M[i, i] = self.link_masses[i] * np.linalg.norm(p_i)**2
            
            # Add terms for off-diagonal elements based on geometric coupling
            for j in range(i):
                T_j = self.forward_kinematics(joint_angles, j+1)
                p_j = T_j[:3, 3]
                
                # Simplified coupling term
                M[i, j] = M[j, i] = 0.1 * self.link_masses[i] * self.link_masses[j] * np.dot(p_i, p_j) / (1 + abs(i-j))
        
        return M
    
    def compute_coriolis_gravity(self, joint_angles, joint_velocities):
        """
        Compute Coriolis/centrifugal and gravity terms
        Simplified implementation for demonstration
        """
        n = len(joint_angles)
        C = np.zeros(n)  # Coriolis/centrifugal terms
        G = np.zeros(n)  # Gravity terms
        
        # Simplified gravity term calculation
        for i in range(n):
            # Gravity effect based on joint angle (simplified)
            G[i] = self.link_masses[i] * 9.81 * math.sin(joint_angles[i])
        
        # Simplified Coriolis term
        for i in range(n):
            # Centrifugal effect based on velocity squared
            C[i] = 0.1 * joint_velocities[i] * abs(joint_velocities[i])
            
            # Coriolis coupling with other joints
            for j in range(n):
                if i != j:
                    C[i] += 0.05 * joint_velocities[i] * joint_velocities[j]
        
        return C, G
    
    def inverse_dynamics(self, joint_angles, joint_velocities, joint_accelerations):
        """
        Compute required joint torques using inverse dynamics (Lagrange-Euler method)
        
        :param joint_angles: Joint position vector
        :param joint_velocities: Joint velocity vector
        :param joint_accelerations: Joint acceleration vector
        :return: Required joint torques
        """
        # Compute inertia matrix
        M = self.compute_inertia_matrix(joint_angles)
        
        # Compute Coriolis/centrifugal and gravity terms
        C, G = self.compute_coriolis_gravity(joint_angles, joint_velocities)
        
        # Compute required torques: τ = M(q)q̈ + C(q, q̇)q̇ + G(q)
        joint_acc_vec = np.array(joint_accelerations)
        tau = M @ joint_acc_vec + C + G
        
        return tau
    
    def forward_dynamics(self, joint_angles, joint_velocities, joint_torques):
        """
        Compute joint accelerations from applied torques (forward dynamics)
        
        :param joint_angles: Joint position vector
        :param joint_velocities: Joint velocity vector
        :param joint_torques: Applied joint torque vector
        :return: Joint accelerations
        """
        # Compute inertia matrix
        M = self.compute_inertia_matrix(joint_angles)
        
        # Compute Coriolis/centrifugal and gravity terms
        C, G = self.compute_coriolis_gravity(joint_angles, joint_velocities)
        
        # Compute accelerations: q̈ = M⁻¹(τ - C(q, q̇)q̇ - G(q))
        tau_vec = np.array(joint_torques)
        qddot = np.linalg.inv(M) @ (tau_vec - C - G)
        
        return qddot


def main():
    # Example usage of the advanced humanoid kinematics and dynamics
    print("Advanced Humanoid Kinematics and Dynamics Example")
    
    # Define a simple 6-DOF humanoid arm configuration
    robot_config = {
        'dh_params': [
            [0.0, np.pi/2, 0.1, 0.0],  # Joint 1: Shoulder rotation
            [0.3, 0.0, 0.0, 0.0],      # Joint 2: Shoulder flexion
            [0.0, np.pi/2, 0.0, 0.0],  # Joint 3: Shoulder abduction
            [0.0, -np.pi/2, 0.3, 0.0], # Joint 4: Elbow flexion
            [0.0, np.pi/2, 0.0, 0.0],  # Joint 5: Wrist flexion
            [0.0, 0.0, 0.1, 0.0]       # Joint 6: Wrist rotation
        ],
        'joint_limits': [
            (-np.pi, np.pi),   # Shoulder rotation
            (-np.pi/2, np.pi/2),  # Shoulder flexion
            (-np.pi/4, np.pi/4),  # Shoulder abduction
            (0, np.pi),        # Elbow flexion
            (-np.pi/2, np.pi/2),  # Wrist flexion
            (-np.pi, np.pi)    # Wrist rotation
        ],
        'link_masses': [2.0, 1.5, 1.0, 1.2, 0.5, 0.3],  # Mass of each link
        'link_com': [
            [0.0, 0.0, 0.05],   # COM of link 1
            [0.15, 0.0, 0.0],   # COM of link 2
            [0.0, 0.0, 0.0],    # COM of link 3
            [0.0, 0.0, 0.15],   # COM of link 4
            [0.0, 0.0, 0.0],    # COM of link 5
            [0.0, 0.0, 0.05]    # COM of link 6
        ]
    }
    
    # Initialize the humanoid model
    robot = HumanoidKinematicsDynamics(robot_config)
    
    # Test forward kinematics
    initial_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    T = robot.forward_kinematics(initial_angles)
    print(f"Initial end-effector transform:\n{T}")
    
    # Get end-effector pose
    pos, quat = robot.get_end_effector_pose(initial_angles)
    print(f"Initial end-effector pose: pos={pos}, quat={quat}")
    
    # Calculate Jacobian
    J = robot.jacobian(initial_angles)
    print(f"Jacobian matrix shape: {J.shape}")
    print(f"Jacobian at initial configuration:\n{J}")
    
    # Test inverse kinematics
    target_pos = [0.3, 0.2, 0.4]
    target_rot = R.from_euler('xyz', [0, 0, 0]).as_quat()  # No rotation
    
    solution = robot.inverse_kinematics(target_pos, target_rot, initial_angles)
    print(f"IK solution: {solution}")
    
    # Verify the solution
    final_pos, final_quat = robot.get_end_effector_pose(solution)
    print(f"Reached position: {final_pos}, target position: {target_pos}")
    print(f"Position error: {np.linalg.norm(np.array(final_pos) - np.array(target_pos))}")
    
    # Test dynamics calculations
    joint_angles = [0.1, 0.2, -0.1, 0.3, 0.05, -0.2]
    joint_velocities = [0.1, 0.15, -0.05, 0.2, 0.1, -0.1]
    joint_accelerations = [0.05, 0.05, -0.02, 0.1, 0.05, -0.05]
    
    # Compute inertia matrix
    M = robot.compute_inertia_matrix(joint_angles)
    print(f"Inertia matrix shape: {M.shape}")
    print(f"Diagonal elements of inertia matrix: {np.diag(M)}")
    
    # Compute Coriolis and gravity terms
    C, G = robot.compute_coriolis_gravity(joint_angles, joint_velocities)
    print(f"Coriolis terms: {C}")
    print(f"Gravity terms: {G}")
    
    # Compute inverse dynamics
    tau = robot.inverse_dynamics(joint_angles, joint_velocities, joint_accelerations)
    print(f"Required joint torques: {tau}")
    
    # Compute forward dynamics
    joint_torques = tau  # Use the same torques for verification
    qddot = robot.forward_dynamics(joint_angles, joint_velocities, joint_torques)
    print(f"Computed accelerations: {qddot}")
    print(f"Input accelerations: {joint_accelerations}")
    
    print("\nAdvanced humanoid kinematics and dynamics demonstration completed")


if __name__ == "__main__":
    main()
```

## Simulation Demonstration

This implementation demonstrates key concepts in humanoid kinematics and dynamics, including forward and inverse kinematics, Jacobian computation, and dynamic modeling. The code can be used in conjunction with simulation environments to plan and control complex humanoid robot movements.

## Hands-On Lab: Humanoid Kinematics and Dynamics Implementation

In this lab, you'll implement and test advanced kinematics and dynamics for a humanoid robot:

1. Implement forward and inverse kinematics for a humanoid arm
2. Calculate and analyze the Jacobian matrix
3. Model the dynamics of the robot
4. Simulate the robot following a trajectory
5. Evaluate the performance and stability of your implementation

### Required Equipment:
- ROS 2 Humble environment
- Python development environment
- (Optional) Robot simulation environment (Gazebo, Isaac Sim)

### Instructions:
1. Create a new ROS 2 package: `ros2 pkg create --build-type ament_python humanoid_kinematics_dynamics`
2. Implement the HumanoidKinematicsDynamics class in your package
3. Create a node that demonstrates forward/inverse kinematics
4. Implement Jacobian-based control for the robot
5. Test with different trajectories and evaluate performance
6. Add dynamic modeling to your implementation
7. Compare the results of inverse and forward dynamics
8. Document your findings and any challenges encountered

## Common Pitfalls & Debugging Notes

- **Singularities**: Be aware of configurations where the Jacobian becomes singular
- **Joint Limits**: Always check that your solutions respect physical joint limits
- **Numerical Accuracy**: Small errors in kinematic calculations can accumulate
- **Dynamics Parameters**: Accurate mass, inertia, and friction parameters are crucial for realistic simulation
- **Computational Complexity**: Inverse dynamics can be computationally intensive; consider simplifications for real-time applications
- **Coordinate Frames**: Ensure consistent use of coordinate frames and transformations

## Summary & Key Terms

**Key Terms:**
- **Forward Kinematics**: Calculating end-effector pose from joint angles
- **Inverse Kinematics**: Calculating joint angles for desired end-effector pose
- **Denavit-Hartenberg Parameters**: Convention for defining coordinate frames in kinematic chains
- **Jacobian Matrix**: Relates joint velocities to Cartesian velocities
- **Singularity**: Configuration where robot loses degrees of freedom
- **Inertia Matrix**: Relates accelerations to applied forces in dynamics
- **Coriolis Forces**: Velocity-dependent forces in rotating reference frames
- **Lagrange-Euler Equations**: Mathematical formulation of robot dynamics

## Further Reading & Citations

1. Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2006). "Robot Modeling and Control." John Wiley & Sons.
2. Craig, J. J. (2005). "Introduction to Robotics: Mechanics and Control" (3rd ed.). Pearson Prentice Hall.
3. Featherstone, R. (2008). "Rigid Body Dynamics Algorithms." Springer.
4. Siciliano, B., & Khatib, O. (Eds.). (2016). "Springer Handbook of Robotics." Springer.

## Assessment Questions

1. Explain the difference between forward and inverse kinematics and their respective challenges.
2. What is the significance of the Jacobian matrix in robot control?
3. Describe the steps to calculate the forward kinematics using DH parameters.
4. How do Coriolis and centrifugal forces affect humanoid robot dynamics?
5. What are the main challenges in solving inverse kinematics for redundant robots?

---
**Previous**: [Introduction to Humanoid Robotics](./intro.md)  
**Next**: [Bipedal Locomotion and Walking Control](./locomotion.md)