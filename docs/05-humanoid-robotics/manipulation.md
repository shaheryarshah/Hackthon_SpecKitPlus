# Manipulation & Humanoid Hands

Manipulation is a fundamental capability that gives humanoid robots the ability to interact with their environment by grasping, manipulating, and releasing objects. This section explores the specialized challenges of dexterous manipulation for humanoid robots, including the biomechanics of human-like hands, grasp planning, and control strategies for performing complex manipulation tasks.

## Learning Outcomes

After completing this section, you should be able to:
- Understand the biomechanics and design principles of humanoid hands
- Implement grasp planning algorithms for various object types
- Design control strategies for dexterous manipulation
- Create synergistic control approaches for multi-fingered hands
- Implement force and tactile feedback methods for manipulation
- Evaluate the manipulation capabilities of humanoid robots
- Address challenges specific to humanoid manipulation such as dexterity and force control

## Core Concepts

### Hand Biomechanics
Humanoid hands must replicate the complex biomechanics of human hands:
- **Degrees of Freedom**: Each finger has multiple joints allowing for complex movements
- **Opposable Thumb**: Essential for precision grasps and tool use
- **Muscle Synergies**: Coordinated activation of multiple muscles to achieve specific functions
- **Tactile Sensing**: Distributed sensing for grip force, texture, and slip detection

### Grasp Types
Humanoid robots need to implement multiple grasp types:
- **Power Grasps**: Strong, stable grasps using palm and fingers for lifting heavy objects
- **Precision Grasps**: Delicate grasps using fingertips for fine control tasks
- **Pinch Grasps**: Grasps between thumb and another finger for small objects
- **Specialized Grasps**: Tool-specific or object-specific grasps

### Grasp Planning
Effective manipulation requires intelligent grasp planning:
- **Geometry Analysis**: Understanding object shape, size, and surface properties
- **Stability Optimization**: Planning grasps that provide stable object holding
- **Force Optimization**: Distributing forces across contact points appropriately
- **Accessibility Planning**: Ensuring the robot can physically reach and grasp the target

### Control Strategies
Precise manipulation requires advanced control approaches:
- **Impedance Control**: Controlling the mechanical impedance of the hand
- **Hybrid Position/Force Control**: Controlling both position and forces simultaneously
- **Adaptive Control**: Adjusting control parameters based on environmental conditions
- **Learning-based Control**: Using experiences to improve future manipulation

## Equations and Models

### Grasp Quality Measures

The grasp quality can be measured using the grasp matrix:

```
G = [n₁  n₂  ...  nₖ]
    [r₁×n₁  r₂×n₂  ...  rₖ×nₖ]
```

Where:
- `nᵢ` is the normal vector at contact point i
- `rᵢ` is the position vector from object center to contact point i
- `k` is the number of contact points

A grasp is force-closure if the convex hull of the contact wrenches contains the origin.

### Grasp Stability Metric

A measure of grasp stability:

```
S = (μ₁ * f₁ + μ₂ * f₂ + ... + μₖ * fₖ) / W
```

Where:
- `μᵢ` is the coefficient of friction at contact point i
- `fᵢ` is the normal force at contact point i
- `W` is the weight of the object
- `k` is the number of contact points

### Multi-Finger Coordination

The synergy between fingers can be expressed as:

```
θ = J · s
```

Where:
- `θ` is the vector of joint angles
- `J` is the synergy Jacobian matrix
- `s` is the synergy vector (activation levels)

### Impedance Control

The relationship between forces, positions, and accelerations in impedance control:

```
M(q)q̈ + C(q,q̇)q̇ + G(q) = τ + JᵀF_ext
```

Where:
- `M(q)` is the mass matrix
- `C(q,q̇)` are Coriolis and centrifugal forces
- `G(q)` is gravitational forces
- `τ` is control torques
- `J` is the Jacobian matrix
- `F_ext` is external forces

## Code Example: Humanoid Manipulation Controller

Here's an implementation of a manipulation controller for humanoid robots:

```python
import numpy as np
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import asyncio


class GraspType(Enum):
    """Types of grasps for humanoid manipulation"""
    POWER_GRASP = "power_grasp"
    PINCH_GRASP = "pinch_grasp"
    PRECISION_GRASP = "precision_grasp"
    LATERAL_PINCH = "lateral_pinch"
    SPHERICAL_GRASP = "spherical_grasp"
    CYLINDRICAL_GRASP = "cylindrical_grasp"


class Finger(Enum):
    """Finger identifiers for humanoid hands"""
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    PINKY = "pinky"


@dataclass
class ObjectProperties:
    """Properties of objects for manipulation planning"""
    shape: str  # 'cylinder', 'sphere', 'box', 'complex'
    dimensions: List[float]  # [length, width, height] or radius
    weight: float  # in kg
    center_of_mass: List[float]  # [x, y, z] relative to object center
    friction_coeff: float  # Coefficient of friction
    material: str  # 'plastic', 'metal', 'paper', etc.
    fragility: float  # 0-1 scale, 1 = very fragile


@dataclass
class GraspConfiguration:
    """Configuration for a specific grasp"""
    grasp_type: GraspType
    contact_points: List[Tuple[float, float, float]]  # (x, y, z) positions
    contact_forces: List[float]  # Force magnitudes at each contact
    finger_positions: Dict[Finger, List[float]]  # Joint positions for each finger
    stability_score: float  # 0-1 scale
    force_closure: bool  # Whether the grasp achieves force closure


class HandKinematics:
    """Model for humanoid hand kinematics and grasp planning"""
    
    def __init__(self, hand_type="human_like"):
        """
        Initialize hand kinematics model
        
        :param hand_type: Type of hand model ('human_like', 'robotic', etc.)
        """
        self.hand_type = hand_type
        self.finger_count = 5  # thumb + 4 fingers
        self.joints_per_finger = 4  # simplified model
        self.finger_lengths = {  # in meters
            Finger.THUMB: 0.04,
            Finger.INDEX: 0.07,
            Finger.MIDDLE: 0.08,
            Finger.RING: 0.07,
            Finger.PINKY: 0.06
        }
        
        # Initialize finger angles (rest position)
        self.finger_angles = {
            Finger.THUMB: [0.0, 0.0, 0.0, 0.0],
            Finger.INDEX: [0.0, 0.0, 0.0, 0.0],
            Finger.MIDDLE: [0.0, 0.0, 0.0, 0.0],
            Finger.RING: [0.0, 0.0, 0.0, 0.0],
            Finger.PINKY: [0.0, 0.0, 0.0, 0.0]
        }
    
    def forward_kinematics(self, finger: Finger, joint_angles: List[float]) -> List[float]:
        """
        Calculate fingertip position from joint angles (simplified 2D model)
        
        :param finger: Finger to calculate for
        :param joint_angles: List of joint angles in radians
        :return: Fingertip position [x, y, z]
        """
        # Simplified forward kinematics for demonstration
        # In a real implementation, this would be more complex with full 3D kinematics
        
        # Start from palm position (assumed at [0, 0, 0])
        x, y, z = 0.0, 0.0, 0.0
        
        # Calculate cumulative position through joints
        segment_length = self.finger_lengths[finger] / len(joint_angles)
        
        for i, angle in enumerate(joint_angles):
            # Project the segment in the direction of accumulated angles
            segment_x = segment_length * math.cos(sum(joint_angles[:i+1]))
            segment_y = segment_length * math.sin(sum(joint_angles[:i+1]))
            
            x += segment_x
            y += segment_y
            # For simplicity, z stays at 0 (would be 3D in real implementation)
        
        return [x, y, z]
    
    def inverse_kinematics(self, finger: Finger, target_pos: List[float]) -> Optional[List[float]]:
        """
        Calculate joint angles to reach target position (simplified)
        
        :param finger: Finger to calculate for
        :param target_pos: Target position [x, y, z]
        :return: Joint angles or None if unreachable
        """
        # Simplified inverse kinematics for demonstration
        # In a real implementation, this would use more sophisticated algorithms
        
        # Calculate distance to target
        dist = math.sqrt(target_pos[0]**2 + target_pos[1]**2 + target_pos[2]**2)
        
        # Check if target is reachable
        max_reach = sum([self.finger_lengths[finger] for finger in Finger]) / 5
        if dist > max_reach:
            return None
        
        # Use geometric approach for simplified 2-link model
        # For real implementation, would use more complex methods
        
        # Calculate required angles (simplified)
        angle_1 = math.atan2(target_pos[1], target_pos[0])
        remaining_dist = max(0, dist - self.finger_lengths[finger]/2)
        
        # For simplicity, distribute remaining motion equally among joints
        remaining_angle = math.asin(remaining_dist / (self.finger_lengths[finger]/2))
        
        return [angle_1 / 2, remaining_angle / 2, remaining_angle / 2, remaining_angle / 2]


class GraspPlanner:
    """Planning grasps for humanoids based on object properties"""
    
    def __init__(self):
        self.hand_model = HandKinematics()
        self.stability_threshold = 0.7
        self.force_closure_threshold = 0.8
    
    def plan_grasp(self, obj_props: ObjectProperties, current_hand_pose: List[float]) -> Optional[GraspConfiguration]:
        """
        Plan an appropriate grasp for the given object
        
        :param obj_props: Properties of the object to grasp
        :param current_hand_pose: Current position/orientation of the hand
        :return: Grasp configuration or None if no suitable grasp found
        """
        # Determine appropriate grasp type based on object properties
        grasp_type = self._select_grasp_type(obj_props)
        
        # Plan contact points based on object geometry
        contact_points = self._plan_contact_points(obj_props, grasp_type)
        
        # Calculate required contact forces
        contact_forces = self._calculate_contact_forces(obj_props, contact_points)
        
        # Determine finger positions
        finger_positions = self._plan_finger_positions(contact_points, grasp_type)
        
        # Evaluate grasp stability
        stability_score = self._evaluate_stability(
            contact_points, 
            contact_forces, 
            obj_props.friction_coeff
        )
        
        force_closure = self._check_force_closure(contact_points)
        
        if stability_score < self.stability_threshold:
            # Try alternative grasp types
            for alt_type in self._get_alternative_grasps(grasp_type):
                alt_config = self._plan_specific_grasp(obj_props, alt_type, current_hand_pose)
                if alt_config and alt_config.stability_score > stability_score:
                    return alt_config
        
        return GraspConfiguration(
            grasp_type=grasp_type,
            contact_points=contact_points,
            contact_forces=contact_forces,
            finger_positions=finger_positions,
            stability_score=stability_score,
            force_closure=force_closure
        )
    
    def _select_grasp_type(self, obj_props: ObjectProperties) -> GraspType:
        """Select appropriate grasp type based on object properties"""
        if obj_props.shape == 'sphere' and obj_props.dimensions[0] < 0.05:  # Small sphere
            return GraspType.PINCH_GRASP
        elif obj_props.shape == 'cylinder' and obj_props.dimensions[0] < 0.03:  # Small cylinder
            return GraspType.PRECISION_GRASP
        elif obj_props.weight > 1.0:  # Heavy object
            return GraspType.POWER_GRASP
        elif obj_props.dimensions[0] > 0.1:  # Large object
            return GraspType.SPHERICAL_GRASP
        else:
            return GraspType.PINCH_GRASP  # Default
    
    def _plan_contact_points(self, obj_props: ObjectProperties, grasp_type: GraspType) -> List[Tuple[float, float, float]]:
        """Plan contact points based on object geometry and grasp type"""
        contacts = []
        
        if obj_props.shape == 'sphere':
            radius = obj_props.dimensions[0]
            if grasp_type == GraspType.PINCH_GRASP:
                # Two contact points for pinch grasp
                contacts.extend([
                    (radius, 0.0, 0.0),   # Right side
                    (-radius, 0.0, 0.0)   # Left side
                ])
            elif grasp_type == GraspType.POWER_GRASP:
                # Multiple contact points around the sphere
                for i in range(6):  # 6 contact points
                    angle = 2 * math.pi * i / 6
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    contacts.append((x, y, 0.0))
        
        elif obj_props.shape == 'cylinder':
            radius, height = obj_props.dimensions[0], obj_props.dimensions[1]
            if grasp_type == GraspType.PINCH_GRASP:
                contacts.extend([
                    (radius, 0.0, height/2),   # Top-right
                    (-radius, 0.0, height/2)   # Top-left
                ])
            elif grasp_type == GraspType.CYLINDRICAL_GRASP:
                # Wrap-around grasp with 4 contact points
                for i in range(4):
                    angle = math.pi * i / 2
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    contacts.append((x, y, 0.0))
        
        elif obj_props.shape == 'box':
            length, width, height = obj_props.dimensions
            if grasp_type in [GraspType.PINCH_GRASP, GraspType.PRECISION_GRASP]:
                # Grasp on the top face
                contacts.extend([
                    (length/2, width/2, height/2),    # Top-right corner
                    (-length/2, width/2, height/2)    # Top-left corner
                ])
        
        return contacts[:8]  # Limit to maximum 8 contacts for computational efficiency
    
    def _calculate_contact_forces(self, obj_props: ObjectProperties, contact_points: List[Tuple[float, float, float]]) -> List[float]:
        """Calculate required contact forces to stably grasp the object"""
        # Calculate total weight
        weight = obj_props.weight * 9.81  # Weight in Newtons
        
        # Distribute force among contact points
        num_contacts = len(contact_points)
        base_force = weight / num_contacts if num_contacts > 0 else 0
        
        # Add safety factor
        safety_factor = 1.5
        forces = [base_force * safety_factor for _ in contact_points]
        
        # Adjust for object fragility
        fragility_factor = 1.0 / (1 + obj_props.fragility * 2)  # More fragile = less force
        forces = [f * fragility_factor for f in forces]
        
        return forces
    
    def _plan_finger_positions(self, contact_points: List[Tuple[float, float, float]], grasp_type: GraspType) -> Dict[Finger, List[float]]:
        """Plan finger positions for the grasp"""
        finger_positions = {}
        
        if grasp_type == GraspType.PINCH_GRASP:
            # Thumb and index finger grasp
            if len(contact_points) >= 2:
                # Position thumb and index finger at contact points
                thumb_target = contact_points[0]
                index_target = contact_points[1]
                
                # Calculate required joint angles (this is highly simplified)
                thumb_angles = [0.0, 0.8, 0.5, 0.3]  # Example joint angles
                index_angles = [0.0, 0.6, 0.4, 0.2]
                
                finger_positions[Finger.THUMB] = thumb_angles
                finger_positions[Finger.INDEX] = index_angles
                
                # Other fingers stay in rest position
                for finger in [Finger.MIDDLE, Finger.RING, Finger.PINKY]:
                    finger_positions[finger] = [0.0, 0.0, 0.0, 0.0]
        
        elif grasp_type == GraspType.POWER_GRASP:
            # Wraparound grasp - all fingers involved
            for finger in Finger:
                # Example: all fingers wrapped around the object
                finger_positions[finger] = [0.2, 0.7, 0.6, 0.4]
        
        elif grasp_type == GraspType.PRECISION_GRASP:
            # Precision grasp with multiple fingertips
            if len(contact_points) >= 3:
                # Thumb, index, and middle finger tips
                for i, finger in enumerate([Finger.THUMB, Finger.INDEX, Finger.MIDDLE]):
                    # Example angles for precision grasp
                    finger_positions[finger] = [0.0, 0.9, 0.7, 0.5]
                
                # Ring and pinky fingers in stabilizing position
                finger_positions[Finger.RING] = [0.0, 0.3, 0.2, 0.1]
                finger_positions[Finger.PINKY] = [0.0, 0.3, 0.2, 0.1]
        
        return finger_positions
    
    def _evaluate_stability(self, contact_points: List[Tuple[float, float, float]], contact_forces: List[float], friction_coeff: float) -> float:
        """Evaluate the stability of the planned grasp"""
        if len(contact_points) < 2 or len(contact_forces) < 2:
            return 0.0
        
        # Simplified stability calculation based on force distribution
        # In a real implementation, this would use more complex grasp analysis
        
        # Calculate average force per contact
        avg_force = sum(contact_forces) / len(contact_forces) if contact_forces else 0
        
        # Calculate variance in forces (more uniform = more stable)
        force_variance = sum([(f - avg_force)**2 for f in contact_forces]) / len(contact_forces) if contact_forces else 0
        
        # Stability score is inversely proportional to variance (and normalized)
        max_variance = 10.0  # Example maximum expected variance
        stability_score = max(0.0, 1.0 - (force_variance / max_variance))
        
        # Also consider friction and geometric factors
        friction_factor = min(1.0, 0.5 + friction_coeff * 0.5)  # Friction contributes to stability
        
        final_score = (stability_score + friction_factor) / 2
        
        return final_score
    
    def _check_force_closure(self, contact_points: List[Tuple[float, float, float]]) -> bool:
        """Check if the grasp achieves force closure"""
        # Simplified force closure check
        # In a real implementation, this would use more rigorous mathematical checks
        return len(contact_points) >= 3
    
    def _get_alternative_grasps(self, primary_grasp: GraspType) -> List[GraspType]:
        """Get alternative grasp types to try if primary fails"""
        alternatives = {
            GraspType.PINCH_GRASP: [GraspType.POWER_GRASP, GraspType.PRECISION_GRASP],
            GraspType.POWER_GRASP: [GraspType.PINCH_GRASP, GraspType.CYLINDRICAL_GRASP],
            GraspType.PRECISION_GRASP: [GraspType.PINCH_GRASP, GraspType.SPHERICAL_GRASP],
            GraspType.LATERAL_PINCH: [GraspType.PINCH_GRASP, GraspType.POWER_GRASP],
            GraspType.SPHERICAL_GRASP: [GraspType.POWER_GRASP, GraspType.PINCH_GRASP],
            GraspType.CYLINDRICAL_GRASP: [GraspType.POWER_GRASP, GraspType.PINCH_GRASP]
        }
        return alternatives.get(primary_grasp, [])
    
    def _plan_specific_grasp(self, obj_props: ObjectProperties, grasp_type: GraspType, current_hand_pose: List[float]) -> Optional[GraspConfiguration]:
        """Plan a specific type of grasp"""
        # This would be called for alternative grasps
        contact_points = self._plan_contact_points(obj_props, grasp_type)
        contact_forces = self._calculate_contact_forces(obj_props, contact_points)
        finger_positions = self._plan_finger_positions(contact_points, grasp_type)
        
        stability_score = self._evaluate_stability(
            contact_points, 
            contact_forces, 
            obj_props.friction_coeff
        )
        
        force_closure = self._check_force_closure(contact_points)
        
        return GraspConfiguration(
            grasp_type=grasp_type,
            contact_points=contact_points,
            contact_forces=contact_forces,
            finger_positions=finger_positions,
            stability_score=stability_score,
            force_closure=force_closure
        )


class ManipulationController:
    """Main controller for humanoid manipulation tasks"""
    
    def __init__(self, robot_arm_dof=7):
        """
        Initialize the manipulation controller
        
        :param robot_arm_dof: Degrees of freedom in the robot arm
        """
        self.grasp_planner = GraspPlanner()
        self.arm_dof = robot_arm_dof
        self.current_object = None
        self.current_grasp = None
        
        # Robot state
        self.hand_state = {
            'position': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0, 1.0],  # quaternion
            'finger_positions': {finger: [0.0, 0.0, 0.0, 0.0] for finger in Finger},
            'gripper_state': 'open'  # 'open', 'closed', 'partially_open'
        }
        
        # Control parameters
        self.impedance_params = {
            'position': 100,  # N/m stiffness
            'orientation': 10,  # Nm/rad stiffness
            'damping_ratio': 1.0  # Critical damping
        }
        
        self.force_limits = {
            'fingertip_normal': 50,  # N
            'lateral_slip': 10,     # N
            'torque': 2            # Nm per joint
        }
    
    def approach_object(self, obj_pose: List[float], approach_distance: float = 0.1) -> bool:
        """
        Move the hand to approach position near the object
        
        :param obj_pose: Object pose [x, y, z, qx, qy, qz, qw] (position + orientation)
        :param approach_distance: Distance from object to start approach
        :return: True if approach successful
        """
        # Calculate approach pose (slightly above and away from object)
        obj_pos = obj_pose[:3]
        obj_orient = obj_pose[3:]
        
        # For approach, we want to be at 'approach_distance' away from object
        # in a direction that makes sense for the planned grasp
        approach_dir = [0, 0, 1]  # Default: approach from above
        
        approach_pos = [
            obj_pos[0] + approach_dir[0] * approach_distance,
            obj_pos[1] + approach_dir[1] * approach_distance,
            obj_pos[2] + approach_dir[2] * approach_distance
        ]
        
        # Move to approach position
        success = self._move_to_pose(approach_pos, obj_orient)
        
        if success:
            self.hand_state['position'] = approach_pos[:]
            self.hand_state['orientation'] = obj_orient[:]
        
        return success
    
    def plan_and_execute_grasp(self, obj_props: ObjectProperties, obj_pose: List[float]) -> bool:
        """
        Plan and execute a grasp on the specified object
        
        :param obj_props: Properties of the object to grasp
        :param obj_pose: Object pose [x, y, z, qx, qy, qz, qw]
        :return: True if grasp successful
        """
        # Plan the grasp
        grasp_config = self.grasp_planner.plan_grasp(obj_props, obj_pose)
        
        if grasp_config is None:
            print("No suitable grasp configuration found")
            return False
        
        self.current_grasp = grasp_config
        print(f"Planned {grasp_config.grasp_type.value} with stability score {grasp_config.stability_score:.3f}")
        
        # Move to pre-grasp position
        pre_grasp_pos = self._calculate_pre_grasp_position(obj_pose, grasp_config)
        if not self._move_to_pose(pre_grasp_pos, obj_pose[3:]):  # Use same orientation
            print("Failed to move to pre-grasp position")
            return False
        
        # Execute the grasp
        success = self._execute_grasp(grasp_config)
        
        if success:
            self.current_object = obj_props
            print("Grasp successful")
        else:
            print("Grasp failed")
        
        return success
    
    def _calculate_pre_grasp_position(self, obj_pose: List[float], grasp_config: GraspConfiguration) -> List[float]:
        """Calculate position for pre-grasp approach"""
        obj_pos = obj_pose[:3]
        obj_orient = obj_pose[3:]
        
        # Simplified: approach from a safe distance along the approach direction
        approach_offset = [0.05, 0.0, 0.0]  # 5cm offset
        
        # Transform offset to object frame
        # This is a simplified transformation, real implementation would use full rotation
        pre_grasp_pos = [
            obj_pos[0] + approach_offset[0],
            obj_pos[1] + approach_offset[1], 
            obj_pos[2] + approach_offset[2]
        ]
        
        return pre_grasp_pos
    
    def _move_to_pose(self, position: List[float], orientation: List[float]) -> bool:
        """Move the end-effector to specified pose (simulated)"""
        # In a real implementation, this would call the robot's motion controller
        # For this simulation, we'll just update the state
        
        # Simulate movement time
        time.sleep(0.5)  # Simulate 0.5 seconds movement time
        
        self.hand_state['position'] = position[:]
        self.hand_state['orientation'] = orientation[:]
        
        # In real implementation, check for collisions and path feasibility
        return True
    
    def _execute_grasp(self, grasp_config: GraspConfiguration) -> bool:
        """Execute the planned grasp"""
        print(f"Executing {grasp_config.grasp_type.value} grasp")
        
        # 1. Move fingers to pre-contact positions
        if not self._move_fingers_to_pre_contact(grasp_config):
            return False
        
        # 2. Close fingers with appropriate forces
        success = self._close_fingers_for_grasp(grasp_config)
        
        if success:
            self.hand_state['gripper_state'] = 'closed'
            # Update finger positions based on grasp configuration
            self.hand_state['finger_positions'] = grasp_config.finger_positions.copy()
        else:
            print("Grasp execution failed")
        
        return success
    
    def _move_fingers_to_pre_contact(self, grasp_config: GraspConfiguration) -> bool:
        """Move fingers to positions just before contact for precision"""
        # In a real implementation, this would move fingers to just before contact
        # For simulation, we'll just wait
        time.sleep(0.2)  # 200ms for pre-positioning
        
        return True
    
    def _close_fingers_for_grasp(self, grasp_config: GraspConfiguration) -> bool:
        """Close fingers with appropriate forces for the grasp"""
        # Apply forces gradually to avoid slippage
        max_force = max(grasp_config.contact_forces) if grasp_config.contact_forces else 10.0
        target_finger_positions = grasp_config.finger_positions
        
        # For simulation, we'll just update finger positions
        # In real implementation, this would involve force control
        for finger, position in target_finger_positions.items():
            self.hand_state['finger_positions'][finger] = position[:]
        
        # Check if grasp is stable
        time.sleep(0.3)  # 300ms for fingers to close and stabilize
        
        # Simulate tactile feedback to verify grasp success
        grasp_verified = self._verify_grasp_stability(grasp_config)
        
        return grasp_verified
    
    def _verify_grasp_stability(self, grasp_config: GraspConfiguration) -> bool:
        """Verify that the grasp is stable"""
        # In a real implementation, this would use tactile sensors
        # For simulation, check the stability score from the planner
        if grasp_config.stability_score >= self.grasp_planner.stability_threshold:
            return True
        else:
            return False
    
    def release_object(self) -> bool:
        """Release the currently grasped object"""
        if self.hand_state['gripper_state'] != 'closed':
            print("No object to release")
            return False
        
        # Open fingers gradually to avoid dropping object suddenly
        for finger in Finger:
            self.hand_state['finger_positions'][finger] = [0.0, 0.0, 0.0, 0.0]  # Open position
        
        self.hand_state['gripper_state'] = 'open'
        
        success = True  # In simulation, always successful
        print("Object released")
        
        return success
    
    def move_object(self, target_pose: List[float]) -> bool:
        """Move the currently grasped object to a target pose"""
        if self.hand_state['gripper_state'] != 'closed':
            print("No object grasped to move")
            return False
        
        if not self._move_to_pose(target_pose[:3], target_pose[3:]):
            print("Failed to move object to target pose")
            return False
        
        print("Object moved to target position")
        return True
    
    def get_manipulation_status(self) -> Dict[str, any]:
        """Get current status of the manipulation system"""
        return {
            'hand_position': self.hand_state['position'],
            'hand_orientation': self.hand_state['orientation'],
            'gripper_state': self.hand_state['gripper_state'],
            'holding_object': self.current_object is not None,
            'current_grasp': self.current_grasp.grasp_type.value if self.current_grasp else None,
            'impedance_params': self.impedance_params,
            'force_limits': self.force_limits
        }


def main():
    """Example usage of the humanoid manipulation system"""
    print("Humanoid Manipulation System Example")
    
    # Initialize the manipulation controller
    manipulator = ManipulationController()
    
    # Create an example object to manipulate
    small_sphere = ObjectProperties(
        shape='sphere',
        dimensions=[0.025],  # 2.5cm radius
        weight=0.05,  # 50g
        center_of_mass=[0.0, 0.0, 0.0],
        friction_coeff=0.8,
        material='plastic',
        fragility=0.3
    )
    
    # Define object pose in world coordinates
    sphere_pose = [0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]  # x, y, z, qx, qy, qz, qw
    
    print(f"\n--- Attempting to grasp sphere ---")
    print(f"Object: sphere, radius 2.5cm, weight 50g")
    print(f"Pose: [{sphere_pose[0]:.2f}, {sphere_pose[1]:.2f}, {sphere_pose[2]:.2f}]")
    
    # Approach the object
    print("\nApproaching object...")
    if not manipulator.approach_object(sphere_pose, approach_distance=0.1):
        print("Failed to approach object")
        return
    
    # Plan and execute grasp
    success = manipulator.plan_and_execute_grasp(small_sphere, sphere_pose)
    
    if success:
        print("\nGrasp successful!")
        
        # Show current status
        status = manipulator.get_manipulation_status()
        print(f"Current status: {status}")
        
        # Move object to a new location
        new_pose = [0.5, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0]  # New position
        print(f"\nMoving object to new location: [{new_pose[0]:.2f}, {new_pose[1]:.2f}, {new_pose[2]:.2f}]")
        
        move_success = manipulator.move_object(new_pose)
        if move_success:
            print("Object moved successfully!")
        
        # Release the object
        print("\nReleasing object...")
        manipulator.release_object()
        
        # Check final status
        final_status = manipulator.get_manipulation_status()
        print(f"Final status: {final_status}")
    else:
        print("\nGrasp failed!")
    
    # Test with a different object - a cylinder
    print(f"\n--- Attempting to grasp cylinder ---")
    
    cylinder = ObjectProperties(
        shape='cylinder',
        dimensions=[0.015, 0.1],  # 1.5cm radius, 10cm height
        weight=0.15,  # 150g
        center_of_mass=[0.0, 0.0, 0.0],
        friction_coeff=0.6,
        material='metal',
        fragility=0.1
    )
    
    cylinder_pose = [0.6, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
    
    print(f"Object: cylinder, radius 1.5cm, height 10cm, weight 150g")
    print(f"Pose: [{cylinder_pose[0]:.2f}, {cylinder_pose[1]:.2f}, {cylinder_pose[2]:.2f}]")
    
    # Approach and grasp the cylinder
    manipulator.approach_object(cylinder_pose, approach_distance=0.1)
    success2 = manipulator.plan_and_execute_grasp(cylinder, cylinder_pose)
    
    if success2:
        print("Cylinder grasp successful!")
        
        # Show the planned grasp configuration
        if manipulator.current_grasp:
            print(f"Grasp type: {manipulator.current_grasp.grasp_type.value}")
            print(f"Stability score: {manipulator.current_grasp.stability_score:.3f}")
            print(f"Force closure: {manipulator.current_grasp.force_closure}")
            print(f"Contact points: {len(manipulator.current_grasp.contact_points)}")
    else:
        print("Cylinder grasp failed!")
    
    print("\nHumanoid manipulation system example completed")


if __name__ == "__main__":
    main()
```

## Simulation Demonstration

This implementation demonstrates sophisticated manipulation capabilities for humanoid robots, including grasp planning, dexterous control, and stable object handling. The system can adapt its grasping approach based on object properties and plan stable, force-closure grasps. The code can be integrated with robot simulation environments to test manipulation behaviors on virtual humanoid robots.

## Hands-On Lab: Humanoid Manipulation Implementation

In this lab, you'll implement and test advanced manipulation techniques for humanoid robots:

1. Set up the manipulation controller with grasp planning
2. Implement different grasp types for various objects
3. Test manipulation with objects of different shapes and properties
4. Implement force control for stable grasping
5. Evaluate the manipulation performance and dexterity

### Required Equipment:
- ROS 2 Humble environment
- Robot simulation environment (Gazebo, Isaac Sim)
- Humanoid robot model with manipulator arms
- (Optional) Physical humanoid robot for testing

### Instructions:
1. Create a new ROS 2 package: `ros2 pkg create --build-type ament_python humanoid_manipulation`
2. Implement the HandKinematics and GraspPlanner classes
3. Create a node that handles manipulation tasks
4. Test with different object shapes and properties
5. Implement force feedback and tactile sensing
6. Test grasp stability and adjustment strategies
7. Evaluate the system's manipulation success rate
8. Document the effectiveness of different grasp types

## Common Pitfalls & Debugging Notes

- **Singularities**: Watch for configurations where the hand loses degrees of freedom
- **Force Control**: Balancing enough force to grip without damaging objects
- **Slippage**: Ensuring grasp stability when lifting or moving objects
- **Collision Avoidance**: Preventing self-collisions during manipulation
- **Model Accuracy**: Ensuring hand models match physical robots for real applications
- **Tactile Feedback**: Handling cases where objects slip during grasp
- **Dynamic Objects**: Grasping objects that may move during approach

## Summary & Key Terms

**Key Terms:**
- **Dexterous Manipulation**: Fine control of objects using multi-fingered hands
- **Grasp Planning**: Computational methods to determine optimal grasp configurations
- **Force Closure**: Property of a grasp that can resist any external force/torque
- **Power Grasp**: Strong grasp using palm and multiple fingers
- **Precision Grasp**: Fine control grasp using fingertips
- **Hand Kinematics**: Geometric relationships in humanoid hand structure
- **Tactile Feedback**: Sensory information from touch and pressure

## Further Reading & Citations

1. Feix, T., et al. (2009). "A comprehensive grasp taxonomy." RSS Workshop on Understanding the Human Hand for Advancing Robotic Manipulation.
2. Cutkosky, M. R. (1989). "On grasp choice, grasp models, and the design of hands for manufacturing tasks." IEEE Transactions on Robotics and Automation.
3. Roa, M. A., & Suárez, R. (2015). "Grasp quality metrics: review and performance." Autonomous Robots.
4. Ajoudani, A., et al. (2015). "The child-size SARAH rehab robot: Towards more effective robotic rehabilitation of gait and posture in children." IEEE International Conference on Rehabilitation Robotics.

## Assessment Questions

1. Explain the different types of grasps and when each is most appropriate.
2. What are the key factors in planning a stable grasp for an object?
3. Describe the role of tactile feedback in humanoid manipulation.
4. How do you calculate whether a grasp achieves force closure?
5. What are the main challenges in implementing dexterous manipulation for humanoid robots?

---
**Previous**: [Human-Robot Interaction](./hri.md)  
**Next**: [Capstone Project Introduction](../../07-capstone/intro.md)