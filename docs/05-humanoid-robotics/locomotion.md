# Bipedal Locomotion and Walking Control

Bipedal locomotion is one of the most challenging aspects of humanoid robotics, requiring precise control of multiple degrees of freedom while maintaining dynamic balance. Unlike wheeled or tracked robots, humanoid robots must manage their center of mass to prevent falling while walking, running, or performing other dynamic movements. This section explores the principles, algorithms, and control strategies for achieving stable bipedal locomotion.

## Learning Outcomes

After completing this section, you should be able to:
- Understand the fundamental principles of bipedal locomotion
- Analyze the phases of the human walking cycle
- Implement ZMP-based walking pattern generation
- Design controllers for bipedal balance and stepping
- Evaluate the stability of walking gaits
- Compare different walking control approaches for humanoid robots

## Core Concepts

### Walking Cycle Phases
Human walking consists of two main phases:
- **Stance Phase**: When the foot is in contact with the ground (60% of gait cycle)
- **Swing Phase**: When the foot is off the ground moving forward (40% of gait cycle)

### Zero Moment Point (ZMP)
The ZMP is a critical concept for bipedal stability, representing the point where the net moment of the ground reaction forces is zero. For stable walking, the ZMP must remain within the support polygon formed by the feet.

### Center of Mass (CoM) Control
Managing the CoM trajectory is essential for maintaining balance during walking. The CoM typically moves in a sinusoidal pattern in both lateral and vertical directions.

### Support Polygon
The support polygon is the convex hull of the ground contact points. For bipedal robots, this changes between single support (one foot on ground) and double support (both feet on ground) phases.

### Capture Point
The capture point is where a biped would need to step to come to a complete stop, accounting for its current velocity and the dynamics of inverted pendulum motion.

## Equations and Models

### Linear Inverted Pendulum Model (LIPM)
The LIPM is commonly used for walking control:

```
ẍ_com = ω²(x_com - x_zmp)
ÿ_com = ω²(y_com - y_zmp)
```

Where:
- `ω² = g/h`, with `g` being gravity and `h` the height of the COM
- `(x_com, y_com)` is the center of mass position
- `(x_zmp, y_zmp)` is the zero moment point position

### ZMP Calculation
The ZMP position is calculated from ground reaction forces:

```
x_zmp = (Σ(F_z_i * x_i) - Σ(M_y_i)) / Σ(F_z_i)
y_zmp = (Σ(F_z_i * y_i) + Σ(M_x_i)) / Σ(F_z_i)
```

Where:
- `F_z_i` is the vertical force at contact point i
- `(x_i, y_i)` is the position of contact point i
- `M_x_i`, `M_y_i` are the moments about the x and y axes at contact point i

### Capture Point Calculation
The capture point is where a biped must step to stop:

```
x_capture = x_com + ẋ_com/ω
y_capture = y_com + ẏ_com/ω
```

### Preview Control for Walking
Preview control uses future reference trajectories to improve tracking:

```
τ = K_p * e(t) + K_i * ∫e(t)dt + K_d * de(t)/dt + Σ(K_i * r(t+i))
```

Where the last term represents preview of future reference points.

## Code Example: Bipedal Walking Controller

Here's an implementation of a ZMP-based walking controller for a humanoid robot:

```python
import numpy as np
import math
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt


class BipedalWalkingController:
    """
    ZMP-based walking controller for a humanoid robot
    """
    def __init__(self, com_height=0.8, dt=0.01, g=9.81):
        """
        Initialize the walking controller
        
        :param com_height: Height of the center of mass (m)
        :param dt: Time step for control (s)
        :param g: Gravitational acceleration (m/s²)
        """
        self.com_height = com_height
        self.dt = dt
        self.g = g
        self.omega = math.sqrt(self.g / self.com_height)  # Natural frequency
        
        # Walking parameters
        self.step_length = 0.3  # Forward step distance (m)
        self.step_width = 0.2   # Lateral step distance (m)
        self.step_height = 0.05 # Maximum step height (m)
        self.step_duration = 1.0  # Total step time (s)
        self.double_support_ratio = 0.1  # Fraction of time in double support
        
        # Current state
        self.com_x = 0.0
        self.com_y = 0.0
        self.com_z = com_height
        self.com_x_dot = 0.0
        self.com_y_dot = 0.0
        self.com_z_dot = 0.0
        
        # Support polygon (initially left foot at origin)
        self.left_foot = np.array([0.0, self.step_width/2.0, 0.0])
        self.right_foot = np.array([0.0, -self.step_width/2.0, 0.0])
        self.support_foot = 'left'  # Which foot is currently supporting
        
        # Walking state
        self.step_phase = 0.0  # Phase of current step (0.0 to 1.0)
        self.swing_foot_trajectory = []
        self.x_zmp_d = 0.0  # Desired ZMP position
        self.y_zmp_d = 0.0
        
        # Walking pattern parameters
        self.walk_forward = 0.0  # Forward walking speed (m/s)
        self.walk_lateral = 0.0  # Lateral walking speed (m/s)
        self.turn_rate = 0.0     # Turning speed (rad/s)
        
        # Initialize walking pattern
        self.generate_next_step()
    
    def generate_next_step(self):
        """
        Generate the next step based on current walking parameters
        """
        # Calculate next foot position based on walking parameters
        if self.support_foot == 'left':
            # Right foot will step forward and possibly laterally/rotationally
            next_x = self.left_foot[0] + self.step_length * self.walk_forward
            next_y = self.left_foot[1] - self.step_width * np.sign(self.right_foot[1])
            
            # Apply turning effect (simplified)
            if self.turn_rate != 0:
                # Simple turning model - rotate next step position
                cos_turn = math.cos(self.turn_rate * self.step_duration / 2)
                sin_turn = math.sin(self.turn_rate * self.step_duration / 2)
                
                # Rotate around support foot
                rel_pos = np.array([next_x, next_y]) - self.left_foot[:2]
                rotated_pos = np.array([
                    rel_pos[0] * cos_turn - rel_pos[1] * sin_turn,
                    rel_pos[0] * sin_turn + rel_pos[1] * cos_turn
                ])
                next_x, next_y = self.left_foot[:2] + rotated_pos
                
        else:  # Support foot is right
            # Left foot will step
            next_x = self.right_foot[0] + self.step_length * self.walk_forward
            next_y = self.right_foot[1] - self.step_width * np.sign(self.left_foot[1])
            
            # Apply turning effect
            if self.turn_rate != 0:
                cos_turn = math.cos(self.turn_rate * self.step_duration / 2)
                sin_turn = math.sin(self.turn_rate * self.step_duration / 2)
                
                # Rotate around support foot
                rel_pos = np.array([next_x, next_y]) - self.right_foot[:2]
                rotated_pos = np.array([
                    rel_pos[0] * cos_turn - rel_pos[1] * sin_turn,
                    rel_pos[0] * sin_turn + rel_pos[1] * cos_turn
                ])
                next_x, next_y = self.right_foot[:2] + rotated_pos
        
        # Generate swing foot trajectory
        self.swing_foot_trajectory = self.generate_swing_trajectory(
            self.get_swing_foot_position(), 
            [next_x, next_y, 0.0]
        )
        
        # Update support polygon
        if self.support_foot == 'left':
            self.right_foot = np.array([next_x, next_y, 0.0])
        else:
            self.left_foot = np.array([next_x, next_y, 0.0])
    
    def generate_swing_trajectory(self, start_pos, end_pos):
        """
        Generate trajectory for the swing foot using a 3rd order polynomial
        """
        trajectory = []
        
        # Start and end positions
        x0, y0, z0 = start_pos
        x1, y1, z1 = end_pos
        
        # Trajectory times
        total_time = self.step_duration
        double_support_time = self.step_duration * self.double_support_ratio
        
        # Generate 3rd order polynomial trajectory
        for t in np.arange(0, total_time, self.dt):
            # Use normalized time (0 to 1) for polynomial calculation
            if t < double_support_time/2:
                # Beginning double support phase
                s = 0.0
            elif t > total_time - double_support_time/2:
                # End double support phase
                s = 1.0
            else:
                # Single support phase
                s = (t - double_support_time/2) / (total_time - double_support_time)
                s = np.clip(s, 0, 1)
            
            # 3rd order polynomial: h(s) = 3s² - 2s³
            h1 = 3 * s**2 - 2 * s**3
            dh1 = 6 * s - 6 * s**2  # Derivative
            ddh1 = 6 - 12 * s       # Second derivative
            
            # Position interpolation
            x = x0 + h1 * (x1 - x0)
            y = y0 + h1 * (y1 - y0)
            
            # Height trajectory with arc
            z = z0 + h1 * (z1 - z0) + 0.5 * (1 - math.cos(math.pi * s)) * self.step_height
            
            # Add to trajectory
            trajectory.append([x, y, z])
        
        return trajectory
    
    def get_swing_foot_position(self):
        """
        Get the current position of the swing foot
        """
        if self.support_foot == 'left':
            return self.right_foot
        else:
            return self.left_foot
    
    def compute_com_trajectory(self, t, zmp_trajectory):
        """
        Compute CoM trajectory based on ZMP using inverted pendulum model
        """
        # For simplicity, we'll use a simplified approach
        # In practice, this would involve solving the inverted pendulum equation
        # or using a preview control approach
        
        # Calculate desired CoM position relative to ZMP
        # Using LIPM: CoM should follow ZMP with specific dynamics
        zmp_x, zmp_y = zmp_trajectory
        
        # Calculate desired CoM position to realize the ZMP
        # Simplified approach: CoM follows ZMP with a small offset
        desired_com_x = zmp_x
        desired_com_y = zmp_y
        
        return desired_com_x, desired_com_y
    
    def update_balance(self, dt):
        """
        Update the balance control based on current state
        """
        # Determine support polygon based on current phase
        if self.step_phase < self.double_support_ratio/2:
            # Double support at beginning
            support_polygon = self.calculate_support_polygon([self.left_foot, self.right_foot])
        elif self.step_phase > 1.0 - self.double_support_ratio/2:
            # Double support at end
            support_polygon = self.calculate_support_polygon([self.left_foot, self.right_foot])
        else:
            # Single support on stance foot
            stance_foot = self.left_foot if self.support_foot == 'left' else self.right_foot
            support_polygon = self.calculate_support_polygon([stance_foot])
        
        # Calculate desired ZMP based on walking pattern
        self.x_zmp_d, self.y_zmp_d = self.calculate_desired_zmp(support_polygon)
        
        # Update CoM using LIPM dynamics
        com_x_ddot = self.omega**2 * (self.com_x - self.x_zmp_d)
        com_y_ddot = self.omega**2 * (self.com_y - self.y_zmp_d)
        
        # Update CoM state
        self.com_x_dot += com_x_ddot * dt
        self.com_y_dot += com_y_ddot * dt
        
        self.com_x += self.com_x_dot * dt
        self.com_y += self.com_y_dot * dt
        
        # Simple CoM height control to maintain average height
        height_error = self.com_height - self.com_z
        self.com_z_dot += 0.1 * height_error * dt
        self.com_z += self.com_z_dot * dt
        
        # Update step phase
        self.step_phase += dt / self.step_duration
        if self.step_phase >= 1.0:
            # Step completed
            self.step_phase = 0.0
            # Switch support foot
            self.support_foot = 'right' if self.support_foot == 'left' else 'left'
            # Generate next step
            self.generate_next_step()
    
    def calculate_support_polygon(self, feet_positions):
        """
        Calculate support polygon from feet positions
        """
        # For now, return average of feet positions as simplified support polygon center
        if len(feet_positions) > 0:
            avg_pos = np.mean(feet_positions, axis=0)
            return avg_pos[:2]  # Return x, y only
        else:
            return np.array([0.0, 0.0])
    
    def calculate_desired_zmp(self, support_polygon_center):
        """
        Calculate desired ZMP based on walking pattern and support polygon
        """
        # For stable walking, ZMP should be within support polygon
        # For simplicity, we'll place ZMP at a stable position in the support polygon
        # In practice, this would follow a more complex walking pattern
        
        # Calculate ZMP position with some margin from support polygon edges
        margin = 0.05  # 5cm margin from edges
        
        # For now, place ZMP at the center of the support polygon
        # with some stability margin
        zmp_x = support_polygon_center[0]
        zmp_y = support_polygon_center[1]
        
        # Add walking pattern influence
        # This would be more complex in a real implementation
        zmp_x += 0.5 * self.walk_forward * np.sin(self.step_phase * 2 * np.pi)
        zmp_y += 0.3 * self.walk_lateral * np.sin(self.step_phase * 2 * np.pi)
        
        return zmp_x, zmp_y
    
    def set_walking_params(self, forward_speed, lateral_speed, turn_rate):
        """
        Set walking parameters for the robot
        
        :param forward_speed: Forward walking speed (m/s)
        :param lateral_speed: Lateral walking speed (m/s)
        :param turn_rate: Turning rate (rad/s)
        """
        self.walk_forward = forward_speed
        self.walk_lateral = lateral_speed
        self.turn_rate = turn_rate
    
    def step_simulation(self, dt=None):
        """
        Perform one simulation step
        
        :param dt: Time step (if None, use default dt)
        :return: Current state [com_x, com_y, com_z, com_x_dot, com_y_dot, com_z_dot]
        """
        if dt is None:
            dt = self.dt
        
        # Update balance control
        self.update_balance(dt)
        
        # Return current CoM state
        return [
            self.com_x, self.com_y, self.com_z,
            self.com_x_dot, self.com_y_dot, self.com_z_dot
        ]


class WalkingPatternGenerator:
    """
    Advanced walking pattern generator using preview control
    """
    def __init__(self, com_height=0.8, dt=0.01, prediction_time=2.0):
        """
        Initialize the pattern generator
        
        :param com_height: Height of center of mass (m)
        :param dt: Control time step (s)
        :param prediction_time: Time horizon for preview control (s)
        """
        self.com_height = com_height
        self.dt = dt
        self.prediction_time = prediction_time
        self.prediction_steps = int(prediction_time / dt)
        self.g = 9.81
        self.omega = math.sqrt(self.g / self.com_height)
        
        # Generate preview control gains
        self.K_preview = self.calculate_preview_gains()
    
    def calculate_preview_gains(self):
        """
        Calculate preview control gains using LQR method
        """
        # State: [x, x_dot] (simplified single-axis model)
        A = np.array([[0, 1], [self.omega**2, 0]])
        B = np.array([[0], [-self.omega**2]])
        C = np.array([[1, 0]])  # Output is position
        
        # LQR weights
        Q = np.array([[1, 0], [0, 0.1]])  # State cost
        R = np.array([[0.1]])             # Control effort cost
        
        # Discrete-time system
        I = np.eye(2)
        Ad = I + self.dt * A
        Bd = self.dt * B
        
        # Solve DARE to get LQR gain
        # For preview control, we also need the preview gain matrix
        # This is a simplified calculation
        P = np.array([[0.3162, 0], [0, 1.7783]])  # Pre-computed solution for example
        
        K_lqr = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
        
        # For preview control, compute additional gains for future reference tracking
        # This is a simplified approximation
        K_preview = []
        for i in range(self.prediction_steps):
            k = (self.omega * np.exp(-self.omega * i * self.dt) * self.dt)
            K_preview.append(k)
        
        return np.array(K_preview)
    
    def generate_walking_pattern(self, zmp_reference, initial_state):
        """
        Generate CoM trajectory using preview control
        
        :param zmp_reference: Reference ZMP trajectory
        :param initial_state: Initial [com_x, com_x_dot] state
        :return: CoM trajectory that realizes the ZMP reference
        """
        # This is a simplified implementation of preview control
        # In a full implementation, this would use the calculated gains
        # to track a reference ZMP trajectory
        
        x_com = initial_state[0]
        x_com_dot = initial_state[1]
        
        com_trajectory = []
        zmp_trajectory = []
        
        for i, zmp_ref in enumerate(zmp_reference):
            # In preview control, future reference values influence current control
            # For simplicity, compute using LIPM dynamics with a feedback controller
            x_com_ddot = self.omega**2 * (x_com - zmp_ref)
            
            # Simple PD control with preview (simplified)
            feedback_control = -1.0 * (x_com - zmp_ref) - 0.7 * x_com_dot
            
            # Update state
            x_com_dot += self.dt * (self.omega**2 * (x_com - zmp_ref) + feedback_control)
            x_com += self.dt * x_com_dot
            
            com_trajectory.append([x_com, x_com_dot])
            zmp_trajectory.append(zmp_ref)
        
        return np.array(com_trajectory), np.array(zmp_trajectory)


def main():
    # Example usage of the bipedal walking controller
    print("Bipedal Walking Controller Example")
    
    # Initialize the walking controller
    walker = BipedalWalkingController(com_height=0.8, dt=0.01)
    
    # Define walking parameters
    walker.set_walking_params(forward_speed=0.3, lateral_speed=0.0, turn_rate=0.0)
    
    # Simulate walking for a few seconds
    simulation_time = 5.0  # seconds
    steps = int(simulation_time / walker.dt)
    
    # Storage for recording the simulation
    com_history = []
    zmp_history = []
    time_history = []
    
    print(f"Simulating bipedal walking for {simulation_time} seconds...")
    
    for i in range(steps):
        # Perform simulation step
        state = walker.step_simulation()
        com_pos = state[:3]
        time_history.append(i * walker.dt)
        
        # Record CoM and ZMP positions
        com_history.append(com_pos.copy())
        zmp_history.append([walker.x_zmp_d, walker.y_zmp_d])
        
        # Print status every second
        if i % int(1.0 / walker.dt) == 0:
            print(f"Time: {i * walker.dt:.2f}s, CoM: [{com_pos[0]:.3f}, {com_pos[1]:.3f}, {com_pos[2]:.3f}], "
                  f"Support: {walker.support_foot}, Phase: {walker.step_phase:.2f}")
    
    # Convert to numpy arrays for analysis
    com_history = np.array(com_history)
    zmp_history = np.array(zmp_history)
    
    # Print final statistics
    avg_com_x = np.mean(com_history[:, 0])
    avg_com_y = np.mean(com_history[:, 1])
    avg_com_z = np.mean(com_history[:, 2])
    
    print(f"\nWalking statistics:")
    print(f"Average CoM position: [{avg_com_x:.3f}, {avg_com_y:.3f}, {avg_com_z:.3f}]")
    print(f"CoM position range:")
    print(f"  X: [{np.min(com_history[:, 0]):.3f}, {np.max(com_history[:, 0]):.3f}]")
    print(f"  Y: [{np.min(com_history[:, 1]):.3f}, {np.max(com_history[:, 1]):.3f}]")
    print(f"  Z: [{np.min(com_history[:, 2]):.3f}, {np.max(com_history[:, 2]):.3f}]")
    
    # Test the walking pattern generator
    print(f"\nTesting walking pattern generator...")
    
    # Create a simple ZMP reference trajectory (moving forward)
    pattern_gen = WalkingPatternGenerator(com_height=0.8, dt=0.01)
    zmp_ref = np.linspace(0.0, 1.0, int(2.0 / 0.01))  # Move ZMP forward over 2 seconds
    
    # Generate CoM trajectory
    initial_state = [0.0, 0.0]  # [initial_pos, initial_vel]
    com_traj, zmp_traj = pattern_gen.generate_walking_pattern(zmp_ref, initial_state)
    
    print(f"Generated {len(com_traj)} trajectory points")
    print(f"CoM moved from {com_traj[0, 0]:.3f} to {com_traj[-1, 0]:.3f}")
    
    print("\nBipedal walking controller demonstration completed")


if __name__ == "__main__":
    main()
```

## Simulation Demonstration

This implementation demonstrates the core concepts of bipedal locomotion, including ZMP-based control, walking pattern generation, and balance maintenance. The controller can be integrated with simulation environments like Gazebo or Isaac Sim to test walking behaviors on virtual humanoid robots.

## Hands-On Lab: Implementing Bipedal Walking

In this lab, you'll implement and test a complete bipedal walking controller:

1. Implement ZMP-based walking control
2. Generate walking patterns for forward, lateral, and turning motions
3. Test balance control under disturbances
4. Analyze the stability of your walking controller
5. Optimize the walking controller parameters

### Required Equipment:
- ROS 2 Humble environment
- Gazebo or Isaac Sim
- Humanoid robot model (e.g., Atlas, THORMANG3, or custom model)
- Python development environment

### Instructions:
1. Create a new ROS 2 package: `ros2 pkg create --build-type ament_python bipedal_walking_controller`
2. Implement the BipedalWalkingController class in your package
3. Test the controller with different walking speeds and directions
4. Implement a simple state machine for walking control
5. Add disturbance rejection capabilities
6. Evaluate the stability margins of your controller
7. Tune the controller parameters for optimal performance
8. Document the walking patterns and stability characteristics of your controller

## Common Pitfalls & Debugging Notes

- **ZMP Stability**: Ensure the ZMP remains within the support polygon at all times
- **CoM Trajectory**: The CoM trajectory should be smooth to avoid excessive accelerations
- **Step Timing**: Properly manage the transition between single and double support phases
- **Disturbance Handling**: Implement robust control to handle external disturbances
- **Foot Placement**: Proper foot placement is critical for balance recovery
- **Simulation Accuracy**: Realistic simulation models are essential for valid results
- **Parameter Tuning**: Walking controllers often require careful parameter tuning

## Summary & Key Terms

**Key Terms:**
- **Bipedal Locomotion**: Walking on two legs
- **Zero Moment Point (ZMP)**: Point where net moment of ground reaction forces is zero
- **Stance Phase**: When the foot is in contact with ground
- **Swing Phase**: When the foot is off ground moving forward
- **Linear Inverted Pendulum Model (LIPM)**: Simplified model for walking dynamics
- **Support Polygon**: Convex hull of ground contact points
- **Capture Point**: Point where robot must step to come to a complete stop
- **Preview Control**: Control method using future reference trajectories
- **Single Support**: When only one foot is in contact with ground
- **Double Support**: When both feet are in contact with ground

## Further Reading & Citations

1. Kajita, S., et al. (2003). "Biped walking pattern generation by using preview control of zero-moment point." IEEE International Conference on Robotics and Automation.
2. Pratt, J., & Tedrake, R. (2006). "On polynomial-based temporal discretization for impedance control." International Journal of Humanoid Robotics.
3. Takenaka, T., et al. (2009). "Real time motion generation and control for biped robot—1st report: Walking pattern generation." IEEE-RAS International Conference on Humanoid Robots.
4. Shafii, N., et al. (2011). "Fast pattern synthesis by using ZMP preview controller for a biped robot standing on a narrow support area." IEEE-RAS International Conference on Humanoid Robots.

## Assessment Questions

1. Explain the concept of Zero Moment Point (ZMP) and its role in bipedal stability.
2. What are the main phases of the human walking cycle and their characteristics?
3. Describe the Linear Inverted Pendulum Model (LIPM) and its application to walking control.
4. How does preview control improve walking pattern generation?
5. What are the key challenges in implementing stable bipedal locomotion?

---
**Previous**: [Humanoid Kinematics and Dynamics](./kinematics.md)  
**Next**: [Vision-Language-Action Robotics](../06-vla-robotics/intro.md)