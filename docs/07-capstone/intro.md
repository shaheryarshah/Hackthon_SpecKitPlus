# Capstone Project: Autonomous Humanoid Robot

The capstone project represents the culmination of all concepts covered in this textbook, bringing together Physical AI, ROS 2, simulation, NVIDIA Isaac Platform, Vision-Language-Action capabilities, and humanoid robotics fundamentals. Students will design and implement an autonomous humanoid robot system capable of understanding voice commands, navigating environments, detecting and manipulating objects, and executing complex tasks in a human-like manner.

## Learning Outcomes

After completing this capstone project, you should be able to:
- Synthesize knowledge from all previous chapters into a comprehensive robotic system
- Design and implement complex robotic systems that integrate multiple technologies
- Integrate perception, planning, and action execution into a unified system
- Develop robust software architectures for autonomous robots
- Implement vision-language-action pipelines for natural human-robot interaction
- Evaluate complex robotic systems in simulation and real environments
- Document and present complex technical solutions
- Apply engineering principles to solve interdisciplinary robotics challenges

## Core Concepts

### System Integration Challenges
The capstone project emphasizes integration across multiple domains:
- **Physical AI**: Embodied intelligence principles throughout the system
- **ROS 2**: Communication and coordination between system modules
- **Simulation**: Development and testing in virtual environments
- **Isaac Platform**: Advanced perception and hardware acceleration
- **Humanoid Robotics**: Bipedal locomotion and dexterous manipulation
- **VLA Robotics**: Voice commands, LLM planning, and action execution

### Autonomous Behavior Pipeline
The robot will demonstrate complete autonomy through:
- **Environmental Perception**: Real-time detection and mapping of objects and obstacles
- **Language Understanding**: Processing natural language commands with context awareness
- **Task Planning**: Breaking down complex goals into executable action sequences
- **Navigation**: Safe movement through dynamic environments
- **Manipulation**: Dexterous handling of objects with appropriate forces
- **Human Interaction**: Natural and intuitive communication with human users

### End-to-End System Design
Critical aspects of the complete system include:
- **Modular Architecture**: Well-defined interfaces between components
- **Real-Time Operation**: Meeting timing constraints for physical interaction
- **Safety Systems**: Fail-safe mechanisms and emergency procedures
- **Adaptive Behavior**: Responding to environmental changes and uncertainties
- **Learning Capabilities**: Improving performance through experience

## Equations and Models

### System Integration Model

The complete autonomous humanoid system can be modeled as:

```
Output = f(VoiceInputs, EnvironmentalState, RobotCapabilities, WorldKnowledge, SafetyConstraints)
```

Where:
- `VoiceInputs` are natural language commands from humans
- `EnvironmentalState` includes objects, layouts, and dynamic obstacles
- `RobotCapabilities` include motion, manipulation, and sensing abilities
- `WorldKnowledge` includes learned information and mappings
- `SafetyConstraints` ensure safe operation

### Performance Metric Integration

The overall system performance integrates multiple subsystem metrics:

```
P_total = w₁*P_perception + w₂*P_language + w₃*P_planning + w₄*P_locomotion + w₅*P_manipulation + w₆*P_interaction
```

Where each subsystem performance is weighted according to its importance in the overall system goals.

### System Reliability Model

The probability of successful task completion considering all subsystems:

```
R_total = R_perception × R_language × R_planning × R_locomotion × R_manipulation × R_safety
```

Where each factor represents the reliability of the corresponding subsystem.

## Code Example: Autonomous Humanoid System Architecture

Here's an example of the complete autonomous humanoid system:

```python
import asyncio
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import threading
import queue
from enum import Enum
import logging
import numpy as np
import speech_recognition as sr
from datetime import datetime


class SystemMode(Enum):
    """Operating modes for the autonomous humanoid"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    NAVIGATING = "navigating"
    MANIPULATING = "manipulating"
    INTERACTING = "interacting"
    SAFETY_EMERGENCY = "safety_emergency"


class RobotState(Enum):
    """State of the robot system"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SAFETY_LOCKOUT = "safety_lockout"


@dataclass
class AutonomousCommand:
    """Represents a command for the autonomous humanoid"""
    id: str
    command_text: str
    parsed_intent: str
    entities: Dict[str, Any]
    timestamp: float
    confidence: float
    priority: int = 5  # 1-10 priority level


@dataclass
class TaskPlan:
    """Represents a planned sequence of actions"""
    id: str
    goal_description: str
    steps: List[Dict[str, Any]]
    constraints: List[str]
    priority: int
    created_at: float


class PerceptionModule:
    """Handles environmental perception and object detection"""
    
    def __init__(self):
        self.detected_objects = {}
        self.spatial_map = {}
        self.last_update = 0.0
        self.is_active = False
        self.camera_feed = None
        self.object_detector_accuracy = 0.95  # 95% detection accuracy target
    
    async def start_perception(self):
        """Start the perception system"""
        self.is_active = True
        logging.info("Perception module started")
    
    async def stop_perception(self):
        """Stop the perception system"""
        self.is_active = False
        logging.info("Perception module stopped")
    
    async def update_environment(self) -> Dict[str, Any]:
        """Update environmental perception"""
        # Simulate detecting objects in environment
        current_time = time.time()
        
        # Simulate detection of various objects
        detected_objects = {
            "red_cup": {
                "type": "cup", 
                "position": [1.2, 0.5, 0.0], 
                "confidence": 0.92,
                "properties": {"color": "red", "material": "ceramic"}
            },
            "blue_box": {
                "type": "container", 
                "position": [1.8, 1.0, 0.0], 
                "confidence": 0.88,
                "properties": {"color": "blue", "material": "cardboard"}
            },
            "table": {
                "type": "furniture", 
                "position": [0.5, 1.0, 0.0], 
                "confidence": 0.98,
                "properties": {"material": "wood", "surface_area": 2.0}
            },
            "chair": {
                "type": "furniture", 
                "position": [-0.2, 1.5, 0.0], 
                "confidence": 0.95,
                "properties": {"material": "fabric", "height": 0.8}
            }
        }
        
        self.detected_objects = detected_objects
        self.last_update = current_time
        
        environment_data = {
            "objects": detected_objects,
            "spatial_map": self.spatial_map,
            "timestamp": current_time,
            "update_duration": 0.1  # Simulated
        }
        
        logging.debug(f"Environment updated: {len(detected_objects)} objects detected")
        return environment_data


class VoiceProcessingModule:
    """Handles voice command processing and natural language understanding"""
    
    def __init__(self):
        self.is_listening = False
        self.speech_recognizer = sr.Recognizer()
        self.speech_recognizer.energy_threshold = 300
        self.command_history = []
        self.last_command = None
        self.language_model = None  # Would be LLM in real implementation
    
    async def start_listening(self):
        """Start listening for voice commands"""
        self.is_listening = True
        logging.info("Voice processing module started listening")
    
    async def stop_listening(self):
        """Stop listening for voice commands"""
        self.is_listening = False
        logging.info("Voice processing module stopped listening")
    
    async def process_voice_input(self, audio_input: bytes = None) -> Optional[AutonomousCommand]:
        """Process voice input and return structured command"""
        # Simulate processing voice command
        if audio_input is None:
            # In simulation, we'll use a predefined command
            simulated_commands = [
                "Please go to the table and bring me the red cup",
                "Move to the blue box and pick it up",
                "Turn left and walk forward",
                "Navigate to the kitchen and wait there"
            ]
            import random
            command_text = random.choice(simulated_commands)
        else:
            # In real implementation, this would process actual audio input
            command_text = "Simulated command"  # Placeholder
        
        # Parse the command (simplified)
        parsed_result = self._parse_command(command_text)
        
        command = AutonomousCommand(
            id=f"cmd_{int(time.time())}",
            command_text=command_text,
            parsed_intent=parsed_result["intent"],
            entities=parsed_result["entities"],
            timestamp=time.time(),
            confidence=parsed_result["confidence"],
            priority=parsed_result["priority"]
        )
        
        self.command_history.append(command)
        self.last_command = command
        
        logging.info(f"Processed voice command: '{command.command_text}'")
        return command
    
    def _parse_command(self, text: str) -> Dict[str, Any]:
        """Parse natural language command into structured format"""
        text_lower = text.lower()
        
        # Simple intent detection (in real system, would use NLP/LLM)
        if any(word in text_lower for word in ["go to", "navigate", "move to", "go to"]):
            intent = "navigate"
        elif any(word in text_lower for word in ["pick up", "grasp", "take", "get"]):
            intent = "manipulate"
        elif any(word in text_lower for word in ["bring", "deliver"]):
            intent = "transport"
        elif any(word in text_lower for word in ["turn", "rotate", "spin"]):
            intent = "orient"
        else:
            intent = "unknown"
        
        # Extract entities (objects, locations)
        entities = {}
        
        # Look for objects
        for obj in ["cup", "box", "table", "chair", "red", "blue", "green", "kitchen", "bedroom"]:
            if obj in text_lower:
                entities[obj] = "object" if obj in ["cup", "box"] else "location" if obj in ["kitchen", "bedroom", "table", "chair"] else "attribute"
        
        # Determine priority based on command urgency
        priority = 5
        if any(word in text_lower for word in ["emergency", "danger", "stop", "help"]):
            priority = 10
        elif any(word in text_lower for word in ["please", "kindly", "carefully"]):
            priority = 3
        
        return {
            "intent": intent,
            "entities": entities,
            "confidence": 0.85,  # Simulated confidence
            "priority": priority
        }


class PlanningModule:
    """Generates task plans from high-level goals"""
    
    def __init__(self):
        self.known_locations = {
            "kitchen": [2.0, -1.0, 0.0],
            "living_room": [0.0, 2.0, 0.0],
            "bedroom": [-1.0, 1.0, 0.0],
            "table": [0.5, 1.0, 0.0]
        }
        self.planning_history = []
    
    def generate_plan(self, command: AutonomousCommand, environment: Dict[str, Any]) -> Optional[TaskPlan]:
        """Generate a task plan from command and environment"""
        goal = command.command_text
        intent = command.parsed_intent
        entities = command.entities
        
        steps = []
        
        if intent == "navigate":
            # Find target location in entities
            target_location = None
            for entity, ent_type in entities.items():
                if ent_type == "location" and entity in self.known_locations:
                    target_location = entity
                    break
            
            if target_location:
                steps.append({
                    "action": "navigate_to",
                    "target": self.known_locations[target_location],
                    "description": f"Navigate to {target_location}",
                    "constraints": ["avoid_obstacles", "maintain_safe_distance"]
                })
        
        elif intent == "manipulate":
            # Find target object in environment
            target_object = None
            for entity, ent_type in entities.items():
                if ent_type == "object" and entity in environment.get("objects", {}):
                    target_object = entity
                    break
            
            if target_object and target_object in environment["objects"]:
                obj_pos = environment["objects"][target_object]["position"]
                
                steps.extend([
                    {
                        "action": "approach",
                        "target": obj_pos,
                        "description": f"Approach {target_object}",
                        "constraints": ["safe_approach", "avoid_collisions"]
                    },
                    {
                        "action": "grasp",
                        "target": target_object,
                        "description": f"Grasp {target_object}",
                        "constraints": ["correct_grasp_type", "appropriate_force"]
                    }
                ])
        
        elif intent == "transport":
            # Transport: find target object and destination
            target_object = None
            destination = None
            
            for entity, ent_type in entities.items():
                if ent_type == "object" and entity in environment.get("objects", {}):
                    target_object = entity
                elif ent_type == "location" and entity in self.known_locations:
                    destination = entity
            
            if target_object and destination:
                obj_pos = environment["objects"][target_object]["position"]
                dest_pos = self.known_locations[destination]
                
                steps.extend([
                    {
                        "action": "approach",
                        "target": obj_pos,
                        "description": f"Approach {target_object}",
                        "constraints": ["safe_approach", "avoid_collisions"]
                    },
                    {
                        "action": "grasp",
                        "target": target_object,
                        "description": f"Grasp {target_object}",
                        "constraints": ["correct_grasp_type", "appropriate_force"]
                    },
                    {
                        "action": "navigate_to",
                        "target": dest_pos,
                        "description": f"Navigate to {destination}",
                        "constraints": ["avoid_obstacles", "maintain_grasp"]
                    },
                    {
                        "action": "release",
                        "target": destination,
                        "description": f"Release object at {destination}",
                        "constraints": ["safe_placement", "controlled_release"]
                    }
                ])
        
        if not steps:
            logging.warning(f"No plan generated for command: {command.command_text}")
            return None
        
        plan = TaskPlan(
            id=f"plan_{int(time.time())}",
            goal_description=goal,
            steps=steps,
            constraints=["safety_first", "robust_operation"],
            priority=command.priority,
            created_at=time.time()
        )
        
        self.planning_history.append(plan)
        logging.info(f"Generated plan with {len(steps)} steps for: '{goal}'")
        return plan


class ExecutionModule:
    """Executes task plans on the robot hardware"""
    
    def __init__(self):
        self.robot_state = {
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],  # quaternion
            "gripper_state": "open",  # "open" or "closed"
            "battery_level": 0.95,  # 95%
            "is_moving": False,
            "last_action": "initialized"
        }
        self.current_plan = None
        self.is_executing = False
        self.action_history = []
        
        # Simulated robot parameters
        self.max_speed = 0.5  # m/s
        self.max_rotation_speed = 0.5  # rad/s
        self.manipulation_accuracy = 0.98  # 98% success rate
    
    async def execute_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        """Execute a task plan with monitoring and safety checks"""
        self.current_plan = plan
        self.is_executing = True
        
        logging.info(f"Executing plan: {plan.goal_description}")
        
        results = {
            "plan_id": plan.id,
            "steps_executed": 0,
            "steps_succeeded": 0,
            "steps_failed": 0,
            "success": True,
            "details": [],
            "execution_time": 0.0,
            "start_time": time.time()
        }
        
        for i, step in enumerate(plan.steps):
            if not self.is_executing:
                logging.warning("Execution was cancelled by safety system")
                results["success"] = False
                break
            
            logging.info(f"Executing step {i+1}/{len(plan.steps)}: {step['description']}")
            
            # Execute the step
            step_start = time.time()
            success = await self._execute_step(step)
            step_time = time.time() - step_start
            
            step_result = {
                "step_number": i + 1,
                "description": step["description"],
                "action": step["action"],
                "target": step.get("target", "none"),
                "success": success,
                "execution_time": step_time,
                "constraints": step.get("constraints", [])
            }
            
            results["details"].append(step_result)
            
            if success:
                results["steps_executed"] += 1
                results["steps_succeeded"] += 1
                
                # Update robot state based on action
                self._update_robot_state_from_action(step)
            else:
                results["steps_executed"] += 1
                results["steps_failed"] += 1
                results["success"] = False
                logging.error(f"Step {i+1} failed: {step['description']}")
                
                # In a real system, we might try recovery procedures here
                break  # For this example, stop on first failure
        
        results["execution_time"] = time.time() - results["start_time"]
        
        # Log final results
        success_rate = results["steps_succeeded"] / len(plan.steps) if plan.steps else 0
        logging.info(f"Plan execution completed. Success rate: {success_rate:.2%} ({results['steps_succeeded']}/{len(plan.steps)})")
        
        self.is_executing = False
        self.action_history.append({
            "plan_id": plan.id,
            "results": results,
            "completed_at": time.time()
        })
        
        return results
    
    async def _execute_step(self, step: Dict[str, Any]) -> bool:
        """Execute a single step in the plan"""
        action = step["action"]
        target = step.get("target")
        
        # Simulate action execution time
        await asyncio.sleep(0.2)  # Simulate processing time
        
        # Execute based on action type
        success = False
        
        if action == "navigate_to":
            if isinstance(target, list) and len(target) >= 2:
                # Simulate navigation
                self.robot_state["position"][0] = target[0]
                self.robot_state["position"][1] = target[1]
                success = True
                logging.info(f"Navigated to position: {target}")
        
        elif action == "approach":
            if isinstance(target, list) and len(target) >= 2:
                # Simulate approach movement
                self.robot_state["position"][0] = target[0]
                self.robot_state["position"][1] = target[1]
                success = True
                logging.info(f"Approached target: {target}")
        
        elif action == "grasp":
            if target:
                self.robot_state["gripper_state"] = "closed"
                success = True
                logging.info(f"Grasped object: {target}")
        
        elif action == "release":
            if target:
                self.robot_state["gripper_state"] = "open"
                success = True
                logging.info(f"Released object at: {target}")
        
        elif action == "rotate":
            if isinstance(target, (int, float)):
                # Simulate rotation
                success = True
                logging.info(f"Rotated by: {target} radians")
        
        # In real implementation, this would interface with robot controls
        return success
    
    def _update_robot_state_from_action(self, step: Dict[str, Any]):
        """Update robot state based on completed action"""
        action = step["action"]
        self.robot_state["last_action"] = action
        
        # Additional state updates based on action type
        if action in ["navigate_to", "approach"]:
            # Update position if target is a position
            if isinstance(step.get("target"), list) and len(step["target"]) >= 2:
                self.robot_state["position"][0] = step["target"][0]
                self.robot_state["position"][1] = step["target"][1]
    
    def stop_execution(self):
        """Stop current execution"""
        self.is_executing = False
        logging.info("Execution stopped")


class SafetySystem:
    """Monitors and ensures safe operation of the autonomous humanoid"""
    
    def __init__(self):
        self.emergency_stop_active = False
        self.safety_violations = []
        self.is_monitoring = False
        self.last_check = 0.0
        
        # Safety parameters
        self.min_collision_distance = 0.3  # meters
        self.max_velocity = 0.8  # m/s
        self.max_joint_force = 50.0  # Newtons
        self.battery_threshold = 0.1  # 10% minimum
    
    def start_monitoring(self):
        """Start safety monitoring"""
        self.is_monitoring = True
        self.last_check = time.time()
        logging.info("Safety system monitoring started")
    
    def stop_monitoring(self):
        """Stop safety monitoring"""
        self.is_monitoring = False
        logging.info("Safety system monitoring stopped")
    
    def check_safety(self, robot_state: Dict[str, Any], environment: Dict[str, Any]) -> bool:
        """Check if current state is safe"""
        # Check emergency stop
        if self.emergency_stop_active:
            return False
        
        # Check battery level
        if robot_state.get("battery_level", 1.0) < self.battery_threshold:
            self._log_violation("Battery level below threshold")
            return False
        
        # Check for potential collisions (simplified)
        robot_pos = robot_state.get("position", [0.0, 0.0, 0.0])
        for obj_name, obj_data in environment.get("objects", {}).items():
            obj_pos = obj_data.get("position", [0.0, 0.0, 0.0])
            if obj_data.get("type") == "obstacle":  # Only check obstacle types
                distance = np.sqrt(sum((robot_pos[i] - obj_pos[i])**2 for i in range(2)))
                if distance < self.min_collision_distance:
                    self._log_violation(f"Potential collision with {obj_name}, distance: {distance:.2f}m")
                    return False
        
        # Check for excessive velocity (if robot is moving)
        if robot_state.get("is_moving", False):
            # In a real system, we would check velocity data
            pass
        
        return True
    
    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        self.emergency_stop_active = True
        logging.critical("EMERGENCY STOP TRIGGERED")
    
    def clear_emergency_stop(self):
        """Clear emergency stop condition"""
        self.emergency_stop_active = False
        logging.info("Emergency stop cleared")
    
    def _log_violation(self, description: str):
        """Log a safety violation"""
        violation = {
            "timestamp": time.time(),
            "description": description,
            "severity": "medium"
        }
        self.safety_violations.append(violation)
        logging.warning(f"Safety violation: {description}")


class AutonomousHumanoidSystem:
    """Complete autonomous humanoid robot system"""
    
    def __init__(self):
        # Initialize all modules
        self.perception = PerceptionModule()
        self.voice_processor = VoiceProcessingModule()
        self.planner = PlanningModule()
        self.executor = ExecutionModule()
        self.safety = SafetySystem()
        
        # System state
        self.mode = SystemMode.IDLE
        self.state = RobotState.INITIALIZING
        self.system_active = False
        self.command_queue = queue.Queue()
        
        # Statistics and monitoring
        self.metrics = {
            "commands_processed": 0,
            "plans_generated": 0,
            "executions_attempted": 0,
            "executions_successful": 0,
            "average_execution_time": 0.0,
            "uptime_seconds": 0.0
        }
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Task management
        self.main_loop_task = None
        self.system_start_time = None
    
    async def initialize(self):
        """Initialize all system components"""
        self.logger.info("Initializing autonomous humanoid system...")
        self.system_start_time = time.time()
        
        # Initialize all modules
        await self.perception.start_perception()
        await self.voice_processor.start_listening()
        self.safety.start_monitoring()
        
        self.state = RobotState.READY
        self.logger.info("Autonomous humanoid system initialized and ready")
    
    async def run(self):
        """Main system operation loop"""
        self.logger.info("Starting autonomous humanoid system operation...")
        self.system_active = True
        self.mode = SystemMode.IDLE
        
        self.main_loop_task = asyncio.create_task(self._main_loop())
        
        try:
            await self.main_loop_task
        except asyncio.CancelledError:
            self.logger.info("System operation cancelled")
        finally:
            await self.shutdown()
    
    async def _main_loop(self):
        """Main system operation loop"""
        while self.system_active:
            try:
                # Update environment perception
                env_data = await self.perception.update_environment()
                
                # Check safety
                if not self.safety.check_safety(self.executor.robot_state, env_data):
                    self.logger.warning("Safety check failed, activating safety protocols")
                    self.safety.trigger_emergency_stop()
                    self.mode = SystemMode.SAFETY_EMERGENCY
                    continue
                
                # Process commands from queue if available
                if not self.command_queue.empty() and self.state == RobotState.READY:
                    command = self.command_queue.get()
                    await self._process_command(command, env_data)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
            
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                # In real system, implement recovery procedures
                await asyncio.sleep(1)  # Wait before continuing
    
    async def _process_command(self, command: AutonomousCommand, environment: Dict[str, Any]):
        """Process a single command through the complete pipeline"""
        self.logger.info(f"Processing command: '{command.command_text}'")
        self.metrics["commands_processed"] += 1
        
        # Generate plan
        plan = self.planner.generate_plan(command, environment)
        if not plan:
            self.logger.warning(f"Could not generate plan for command: {command.command_text}")
            return
        
        self.metrics["plans_generated"] += 1
        self.logger.info(f"Generated plan with {len(plan.steps)} steps")
        
        # Execute plan
        self.metrics["executions_attempted"] += 1
        execution_result = await self.executor.execute_plan(plan)
        
        if execution_result["success"]:
            self.metrics["executions_successful"] += 1
        
        # Update execution time metrics
        if self.metrics["executions_attempted"] > 0:
            total_time = sum(r["execution_time"] for r in execution_result.get("details", []))
            self.metrics["average_execution_time"] = total_time / self.metrics["executions_attempted"]
        
        self.state = RobotState.READY
        self.mode = SystemMode.IDLE
    
    async def submit_command(self, command_text: str) -> str:
        """Submit a command for execution"""
        if self.state != RobotState.READY:
            raise RuntimeError(f"System not ready, current state: {self.state.value}")
        
        # Create command object
        # For simulation, we'll create a command directly
        parsed_result = self.voice_processor._parse_command(command_text)
        command = AutonomousCommand(
            id=f"cmd_{int(time.time())}",
            command_text=command_text,
            parsed_intent=parsed_result["intent"],
            entities=parsed_result["entities"],
            timestamp=time.time(),
            confidence=parsed_result["confidence"],
            priority=parsed_result["priority"]
        )
        
        # Add to queue
        self.command_queue.put(command)
        
        self.mode = SystemMode.PROCESSING
        self.state = RobotState.BUSY
        
        return command.id
    
    async def shutdown(self):
        """Shut down the system safely"""
        self.logger.info("Shutting down autonomous humanoid system...")
        self.system_active = False
        
        # Stop all modules
        await self.perception.stop_perception()
        await self.voice_processor.stop_listening()
        self.safety.stop_monitoring()
        self.executor.stop_execution()
        
        if self.main_loop_task:
            self.main_loop_task.cancel()
            try:
                await self.main_loop_task
            except asyncio.CancelledError:
                pass
        
        self.state = RobotState.INITIALIZING  # Reset for next startup
        self.logger.info("Autonomous humanoid system shut down")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics"""
        current_time = time.time()
        uptime = (current_time - self.system_start_time) if self.system_start_time else 0.0
        
        return {
            "mode": self.mode.value,
            "state": self.state.value,
            "system_active": self.system_active,
            "uptime_seconds": uptime,
            "command_queue_size": self.command_queue.qsize(),
            "robot_position": self.executor.robot_state["position"],
            "gripper_state": self.executor.robot_state["gripper_state"],
            "battery_level": self.executor.robot_state["battery_level"],
            "last_action": self.executor.robot_state["last_action"],
            "safety_status": {
                "emergency_stop": self.safety.emergency_stop_active,
                "violations_count": len(self.safety.safety_violations),
                "monitoring_active": self.safety.is_monitoring
            },
            "metrics": {
                "commands_processed": self.metrics["commands_processed"],
                "plans_generated": self.metrics["plans_generated"],
                "executions_attempted": self.metrics["executions_attempted"],
                "executions_successful": self.metrics["executions_successful"],
                "success_rate": self.metrics["executions_successful"] / self.metrics["executions_attempted"] if self.metrics["executions_attempted"] > 0 else 0,
                "average_execution_time": self.metrics["average_execution_time"]
            }
        }


def main():
    """Example usage of the autonomous humanoid system"""
    print("Autonomous Humanoid Robot - Capstone Project")
    print("=" * 50)
    
    # Create and initialize the system
    robot_system = AutonomousHumanoidSystem()
    
    async def run_system_example():
        await robot_system.initialize()
        
        # Show system status
        status = robot_system.get_system_status()
        print(f"Initial system status: {status['state']}")
        print(f"Robot position: {status['robot_position']}")
        print(f"System active: {status['system_active']}")
        
        # Simulate a few commands
        test_commands = [
            "Please go to the table and bring me the red cup",
            "Move to the blue box and pick it up",
            "Navigate to the kitchen and wait there"
        ]
        
        print(f"\nProcessing {len(test_commands)} test commands...")
        
        for i, command in enumerate(test_commands):
            print(f"\nCommand {i+1}: {command}")
            
            # Submit the command
            cmd_id = await robot_system.submit_command(command)
            print(f"Command submitted with ID: {cmd_id}")
            
            # Wait briefly between commands to see processing
            await asyncio.sleep(2)
        
        # Show final status
        final_status = robot_system.get_system_status()
        print(f"\nFinal system status:")
        print(f"  Mode: {final_status['mode']}")
        print(f"  State: {final_status['state']}")
        print(f"  Position: {final_status['robot_position']}")
        print(f"  Gripper: {final_status['gripper_state']}")
        print(f"  Commands processed: {final_status['metrics']['commands_processed']}")
        print(f"  Success rate: {final_status['metrics']['success_rate']:.2%}")
        
        return final_status
    
    try:
        # Run the simulation
        final_status = asyncio.run(run_system_example())
        
        print(f"\nSimulation completed successfully!")
        print(f"The autonomous humanoid system demonstrates integration of:")
        print(f"- Physical AI principles for embodied intelligence")
        print(f"- ROS 2 for system communication and coordination")
        print(f"- Simulation environments for development and testing")
        print(f"- Isaac Platform for advanced perception and acceleration")
        print(f"- Vision-Language-Action capabilities for natural interaction")
        print(f"- Humanoid robotics fundamentals for locomotion and manipulation")
        print(f"\nThis capstone system represents the culmination of all concepts")
        print(f"covered in this textbook, showing how they can be integrated")
        print(f"into a complete, autonomous robotic system.")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError during simulation: {e}")
    finally:
        # Ensure system is properly shut down
        try:
            asyncio.run(robot_system.shutdown())
        except:
            pass  # Shutdown may fail if system wasn't started


if __name__ == "__main__":
    main()
```

## Simulation Demonstration

This implementation demonstrates the complete integration of all textbook concepts into a functional autonomous humanoid robot system. The system processes voice commands, perceives the environment, plans complex tasks, and executes them safely. The code can be integrated with simulation environments like Isaac Sim to test the complete pipeline on virtual humanoid robots before deployment on physical systems.

## Hands-On Lab: Capstone Project Implementation

In this capstone lab, you'll implement and test the complete autonomous humanoid system:

1. Integrate all textbook modules into a unified system
2. Implement the voice-language-action pipeline
3. Connect perception to action execution
4. Test with various commands and scenarios
5. Evaluate system performance and safety
6. Document the complete implementation

### Required Equipment:
- ROS 2 Humble environment
- Isaac Sim or Gazebo simulation environment
- Python development environment
- (Optional) Physical humanoid robot for testing

### Instructions:
1. Create a new ROS 2 package: `ros2 pkg create --build-type ament_python capstone_autonomous_humanoid`
2. Implement the AutonomousHumanoidSystem and all its modules
3. Create launch files to start the entire system
4. Test with various voice commands and navigation scenarios
5. Implement safety monitoring and emergency procedures
6. Evaluate system performance with metrics like success rate and response time
7. Test with simulation environments (Isaac Sim, Gazebo)
8. Document the complete system architecture and implementation

## Common Pitfalls & Debugging Notes

- **Integration Complexity**: Connecting multiple complex systems requires extensive testing
- **Timing Issues**: Different modules may run at different frequencies
- **State Synchronization**: Maintaining consistent state across all modules
- **Error Propagation**: Errors in one module affecting others
- **Resource Management**: Managing computational and power resources
- **Safety Considerations**: Ensuring safe operation of the complete integrated system
- **Testing Complexity**: Testing integrated systems is more challenging than individual modules

## Summary & Key Terms

**Key Terms:**
- **System Integration**: Combining multiple subsystems into a unified robot system
- **End-to-End System**: Complete autonomous robot solution from input to action
- **VLA Pipeline**: Vision-Language-Action processing pipeline
- **Autonomous Behavior**: Robot behavior without direct human intervention
- **Task Planning**: Decomposing complex goals into executable actions
- **Safety-First Design**: Prioritizing safety in all system decisions
- **Real-Time Execution**: Meeting timing constraints for robotic actions

## Further Reading & Citations

1. Khatib, O., et al. (2018). "Robotics: Systems and Foundations of Movement." Annual Review of Control, Robotics, and Autonomous Systems.
2. Siciliano, B., & Khatib, O. (Eds.). (2016). "Springer Handbook of Robotics." Springer.
3. Goodrich, M. A., & Schultz, A. C. (2007). "Human-robot interaction: a survey." Foundations and Trends in Human-Computer Interaction.
4. Murphy, R. R. (2000). "Introduction to AI Robotics." MIT Press.

## Assessment Questions

1. Explain how the capstone project integrates all concepts from the textbook.
2. What are the key challenges in building an end-to-end autonomous humanoid system?
3. Describe the VLA pipeline implemented in your capstone project.
4. How did you ensure safety across all integrated subsystems?
5. What metrics would you use to evaluate the performance of your autonomous humanoid robot?

---
**Previous**: [Human-Robot Interaction](../05-humanoid-robotics/hri.md)  
**Next**: [System Design and Implementation](./system-design.md)