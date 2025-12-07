# System Design and Implementation

The system design of our autonomous humanoid robot represents the culmination of all concepts covered in this textbook. This chapter details the architectural decisions, implementation strategies, and engineering considerations that went into creating a complete, functional robotic system. The design follows modern robotics principles while integrating Physical AI, ROS 2, simulation, NVIDIA Isaac, Vision-Language-Action capabilities, and humanoid robotics fundamentals.

## Learning Outcomes

After studying this chapter, you should be able to:
- Understand the architectural design of complex autonomous robotic systems
- Apply design patterns and principles to robotic system development
- Implement modular systems with well-defined interfaces
- Design safety mechanisms and fail-safe procedures
- Create scalable architectures that can accommodate future enhancements
- Evaluate design decisions for their impact on system performance and safety

## Core Concepts

### Modular Architecture
The system is designed with modularity in mind:
- **Decoupled Components**: Each module can function independently
- **Well-Defined Interfaces**: Clear input/output specifications
- **Replaceable Components**: Modules can be swapped without affecting others
- **Testable Units**: Individual modules can be validated in isolation

### Real-Time Considerations
The system design accounts for:
- **Hard Real-Time Requirements**: Safety-critical actions with strict timing
- **Soft Real-Time Requirements**: Performance-critical but flexible timing
- **Resource Management**: Efficient use of computational resources
- **Scheduling**: Prioritizing critical tasks appropriately

### Safety-First Design
Critical safety aspects include:
- **Fail-Safe Mechanisms**: Default safe states when systems fail
- **Redundancy**: Multiple pathways for critical functions
- **Monitoring**: Continuous checks on system health
- **Emergency Procedures**: Quick response to dangerous situations

### Scalability Principles
The design supports:
- **Future Expansion**: Adding new capabilities without redesign
- **Performance Scaling**: Handling increased complexity
- **Platform Portability**: Adapting to different robot platforms
- **Multi-Robot Coordination**: Extending to multiple robots

## Equations and Models

### System Architecture Model

The system architecture can be modeled as:

```
S = M₁ × M₂ × ... × Mₙ → R
```

Where:
- `S` is the system input (sensors, commands)
- `Mᵢ` are the system modules
- `R` is the system output (actions, responses)

### System Reliability Model

The overall system reliability:

```
R_system = R_voice × R_perception × R_planning × R_execution × R_safety
```

Where each reliability factor represents the probability that the respective module functions correctly.

### Performance Bottleneck Analysis

Identifying system bottlenecks:

```
T_total = T_voice + T_perception + T_planning + T_execution + T_communication
```

Where `T_total` must be less than the system's real-time constraints.

## Code Example: System Architecture Implementation

Here's an implementation of the complete system architecture with design patterns:

```python
import asyncio
import threading
import queue
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
import logging
from concurrent.futures import ThreadPoolExecutor


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Status of system components"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class SystemMessage:
    """Message passed between system components"""
    message_type: str
    content: Any
    timestamp: float = 0.0
    source: str = ""
    priority: int = 5  # 1-10 priority level
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class SystemBus:
    """Central message bus for component communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_queue = queue.Queue()
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    def subscribe(self, message_type: str, callback: Callable):
        """Subscribe to specific message types"""
        if message_type not in self.subscribers:
            self.subscribers[message_type] = []
        self.subscribers[message_type].append(callback)
    
    def publish(self, message: SystemMessage):
        """Publish a message to all subscribers"""
        if message.message_type in self.subscribers:
            for callback in self.subscribers[message.message_type]:
                # Use executor to run callbacks in separate threads
                self.executor.submit(callback, message)
    
    def start(self):
        """Start the message bus"""
        self.running = True
        logger.info("System bus started")
    
    def stop(self):
        """Stop the message bus"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("System bus stopped")


class SystemComponent(ABC):
    """Abstract base class for system components"""
    
    def __init__(self, name: str, system_bus: SystemBus):
        self.name = name
        self.system_bus = system_bus
        self.status = ComponentStatus.INITIALIZING
        self.start_time = 0.0
        self.error_count = 0
        self.health_score = 100.0  # 0-100 scale
    
    @abstractmethod
    async def initialize(self):
        """Initialize the component"""
        pass
    
    @abstractmethod
    async def process(self, message: SystemMessage):
        """Process a message"""
        pass
    
    @abstractmethod
    def get_status(self) -> ComponentStatus:
        """Get component status"""
        return self.status
    
    def report_error(self, error: Exception):
        """Report an error and update health score"""
        self.error_count += 1
        self.health_score = max(0, self.health_score - 10)
        logger.error(f"Component {self.name} error: {error}")
    
    def update_health(self, performance_score: float):
        """Update health score based on performance"""
        self.health_score = min(100, 0.7 * self.health_score + 0.3 * performance_score)


class SafetyManager(SystemComponent):
    """Manages safety monitoring and emergency procedures"""
    
    def __init__(self, name: str, system_bus: SystemBus):
        super().__init__(name, system_bus)
        self.safety_constraints = {
            "max_velocity": 1.0,  # m/s
            "collision_distance": 0.5,  # meters
            "joint_limits": [-3.14, 3.14],  # radians
            "battery_threshold": 10.0  # percentage
        }
        self.emergency_active = False
        self.safety_violations = []
        self.robot_state = {
            "position": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "battery_level": 100.0,
            "gripper_state": "open"
        }
    
    async def initialize(self):
        """Initialize the safety manager"""
        self.system_bus.subscribe("robot_state", self.handle_robot_state)
        self.system_bus.subscribe("emergency_stop", self.handle_emergency_stop)
        self.system_bus.subscribe("action_request", self.handle_action_request)
        self.status = ComponentStatus.READY
        logger.info(f"{self.name} initialized")
    
    async def process(self, message: SystemMessage):
        """Process safety-related messages"""
        try:
            if message.message_type == "action_request":
                # Check if action is safe
                is_safe = self._check_action_safety(message.content)
                if not is_safe:
                    logger.warning(f"Action blocked by safety manager: {message.content}")
                    self.system_bus.publish(SystemMessage(
                        "action_blocked",
                        {"reason": "safety_violation", "action": message.content},
                        source=self.name
                    ))
                    return False
            elif message.message_type == "robot_state":
                # Update robot state for safety monitoring
                self.robot_state.update(message.content)
            
            return True
        except Exception as e:
            self.report_error(e)
            return False
    
    def _check_action_safety(self, action: Dict[str, Any]) -> bool:
        """Check if an action is safe to execute"""
        # Check velocity constraints
        if action.get("type") == "move" and action.get("velocity", 0) > self.safety_constraints["max_velocity"]:
            self.safety_violations.append(f"Velocity limit exceeded: {action['velocity']}")
            return False
        
        # Check collision constraints
        if action.get("type") == "navigate" and action.get("target"):
            # Simulate distance check
            target = action["target"]
            current_pos = self.robot_state["position"]
            distance = sum((target[i] - current_pos[i])**2 for i in range(2))**0.5
            
            if distance < self.safety_constraints["collision_distance"]:
                self.safety_violations.append(f"Collision risk: target too close: {distance}")
                return False
        
        # Check battery level
        if self.robot_state.get("battery_level", 100) < self.safety_constraints["battery_threshold"]:
            self.safety_violations.append("Battery level too low")
            return False
        
        return True
    
    def handle_robot_state(self, message: SystemMessage):
        """Handle robot state updates"""
        if message.message_type == "robot_state":
            self.robot_state.update(message.content)
    
    def handle_emergency_stop(self, message: SystemMessage):
        """Handle emergency stop request"""
        if message.message_type == "emergency_stop":
            self.emergency_active = True
            logger.warning("EMERGENCY STOP ACTIVATED")
    
    def handle_action_request(self, message: SystemMessage):
        """Handle action request with safety checks"""
        if message.message_type == "action_request":
            is_safe = self._check_action_safety(message.content)
            if not is_safe:
                self.system_bus.publish(SystemMessage(
                    "safety_violation",
                    {"violations": self.safety_violations[-1:]},  # Last violation
                    source=self.name
                ))
    
    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        self.emergency_active = True
        self.system_bus.publish(SystemMessage(
            "emergency_stop", 
            {"source": self.name}, 
            source=self.name
        ))
    
    def get_status(self) -> ComponentStatus:
        """Get safety manager status"""
        if self.emergency_active:
            return ComponentStatus.ERROR
        return self.status


class VoiceProcessingComponent(SystemComponent):
    """Component for processing voice commands"""
    
    def __init__(self, name: str, system_bus: SystemBus):
        super().__init__(name, system_bus)
        self.command_history = []
        self.active_listening = False
    
    async def initialize(self):
        """Initialize voice processing"""
        self.system_bus.subscribe("voice_input", self.handle_voice_input)
        self.status = ComponentStatus.READY
        logger.info(f"{self.name} initialized")
    
    async def process(self, message: SystemMessage):
        """Process voice-related messages"""
        try:
            if message.message_type == "voice_input":
                # Simulate voice processing
                text = self._process_audio(message.content)
                confidence = self._estimate_confidence(text)
                
                if confidence > 0.7:  # Confidence threshold
                    parsed_command = self._parse_command(text)
                    self.system_bus.publish(SystemMessage(
                        "command_parsed",
                        {
                            "text": text,
                            "confidence": confidence,
                            "parsed": parsed_command,
                            "timestamp": message.timestamp
                        },
                        source=self.name
                    ))
                else:
                    logger.warning(f"Low confidence voice command: {confidence}")
                
                return True
            return False
        except Exception as e:
            self.report_error(e)
            return False
    
    def _process_audio(self, audio_data: bytes) -> str:
        """Process audio data (simulated)"""
        # In a real system, this would use ASR like Whisper
        # For simulation, return a fixed command
        import random
        commands = [
            "Go to the kitchen and bring me a cup",
            "Move to the table and grasp the red object",
            "Navigate to the bedroom",
            "Pick up the blue bottle"
        ]
        return random.choice(commands) if random.random() > 0.3 else "Unknown command"
    
    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence in transcription"""
        if text == "Unknown command":
            return 0.3
        return 0.8 + (len(text) % 5) * 0.05  # Simulate confidence based on text length
    
    def _parse_command(self, text: str) -> Dict[str, Any]:
        """Parse the command into structured format"""
        # Simple parsing for demonstration
        tokens = text.lower().split()
        entities = []
        actions = []
        
        for token in tokens:
            if token in ["kitchen", "bedroom", "table", "living room"]:
                entities.append({"type": "location", "value": token})
            elif token in ["cup", "bottle", "box", "object"]:
                entities.append({"type": "object", "value": token})
            elif token in ["go", "move", "navigate", "pick", "grasp"]:
                actions.append(token)
        
        return {
            "original_text": text,
            "actions": actions,
            "entities": entities,
            "intent": self._determine_intent(text)
        }
    
    def _determine_intent(self, text: str) -> str:
        """Determine the intent of the command"""
        text_lower = text.lower()
        if any(word in text_lower for word in ["go", "navigate", "move"]):
            return "navigation"
        elif any(word in text_lower for word in ["pick", "grasp", "take"]):
            return "manipulation"
        else:
            return "unknown"
    
    def handle_voice_input(self, message: SystemMessage):
        """Handle voice input directly"""
        if message.message_type == "voice_input":
            asyncio.create_task(self.process(message))


class PerceptionComponent(SystemComponent):
    """Component for environment perception"""
    
    def __init__(self, name: str, system_bus: SystemBus):
        super().__init__(name, system_bus)
        self.objects = {}
        self.spatial_map = {}
        self.last_update = 0.0
    
    async def initialize(self):
        """Initialize perception component"""
        self.system_bus.subscribe("request_perception", self.handle_perception_request)
        self.status = ComponentStatus.READY
        logger.info(f"{self.name} initialized")
    
    async def process(self, message: SystemMessage):
        """Process perception-related messages"""
        try:
            if message.message_type == "request_perception":
                perception_data = await self._update_perception()
                self.system_bus.publish(SystemMessage(
                    "perception_update",
                    perception_data,
                    source=self.name
                ))
                return True
            return False
        except Exception as e:
            self.report_error(e)
            return False
    
    async def _update_perception(self) -> Dict[str, Any]:
        """Update perception data (simulated)"""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Simulate detecting objects
        new_objects = {
            "red_cup": {"position": [1.0, 0.5, 0.0], "type": "cup", "color": "red", "confidence": 0.9},
            "blue_bottle": {"position": [1.5, 1.0, 0.0], "type": "bottle", "color": "blue", "confidence": 0.85},
            "table": {"position": [0.0, 1.0, 0.0], "type": "furniture", "confidence": 0.95}
        }
        
        self.objects.update(new_objects)
        self.last_update = time.time()
        
        return {
            "objects": new_objects,
            "timestamp": self.last_update,
            "update_duration": 0.1  # Simulated
        }
    
    def handle_perception_request(self, message: SystemMessage):
        """Handle perception update request"""
        if message.message_type == "request_perception":
            asyncio.create_task(self.process(message))


class PlanningComponent(SystemComponent):
    """Component for task planning"""
    
    def __init__(self, name: str, system_bus: SystemBus):
        super().__init__(name, system_bus)
        self.known_locations = {
            "kitchen": [2.0, 0.0, 0.0],
            "bedroom": [-1.0, 1.0, 0.0],
            "living_room": [0.0, 2.0, 0.0],
            "table": [0.5, 1.0, 0.0]
        }
        self.current_plan = None
        self.plan_history = []
    
    async def initialize(self):
        """Initialize planning component"""
        self.system_bus.subscribe("command_parsed", self.handle_parsed_command)
        self.system_bus.subscribe("request_plan", self.handle_plan_request)
        self.status = ComponentStatus.READY
        logger.info(f"{self.name} initialized")
    
    async def process(self, message: SystemMessage):
        """Process planning-related messages"""
        try:
            if message.message_type == "command_parsed":
                # Generate plan based on parsed command
                plan = self._generate_plan(message.content)
                self.current_plan = plan
                self.plan_history.append(plan)
                
                self.system_bus.publish(SystemMessage(
                    "plan_generated",
                    plan,
                    source=self.name
                ))
                
                return True
            elif message.message_type == "request_plan":
                if self.current_plan:
                    self.system_bus.publish(SystemMessage(
                        "current_plan",
                        self.current_plan,
                        source=self.name
                    ))
                return True
            
            return False
        except Exception as e:
            self.report_error(e)
            return False
    
    def _generate_plan(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a plan from parsed command"""
        text = command_data["text"]
        entities = command_data["parsed"]["entities"]
        intent = command_data["parsed"]["intent"]
        
        plan_steps = []
        
        # Simple planning for demonstration
        if intent == "navigation":
            for entity in entities:
                if entity["type"] == "location" and entity["value"] in self.known_locations:
                    target_pos = self.known_locations[entity["value"]]
                    plan_steps.append({
                        "action": "navigate",
                        "target": target_pos,
                        "description": f"Navigate to {entity['value']}"
                    })
        
        elif intent == "manipulation":
            for entity in entities:
                if entity["type"] == "object":
                    plan_steps.append({
                        "action": "detect",
                        "target": entity["value"],
                        "description": f"Detect {entity['value']}"
                    })
                    plan_steps.append({
                        "action": "move_to",
                        "target": entity["value"],
                        "description": f"Move to {entity['value']}"
                    })
                    plan_steps.append({
                        "action": "grasp",
                        "target": entity["value"],
                        "description": f"Grasp {entity['value']}"
                    })
        
        return {
            "goal": text,
            "steps": plan_steps,
            "generated_at": time.time(),
            "confidence": command_data["confidence"]
        }
    
    def handle_parsed_command(self, message: SystemMessage):
        """Handle parsed command for planning"""
        if message.message_type == "command_parsed":
            asyncio.create_task(self.process(message))
    
    def handle_plan_request(self, message: SystemMessage):
        """Handle plan request"""
        if message.message_type == "request_plan":
            asyncio.create_task(self.process(message))


class ExecutionComponent(SystemComponent):
    """Component for action execution"""
    
    def __init__(self, name: str, system_bus: SystemBus):
        super().__init__(name, system_bus)
        self.robot_position = [0.0, 0.0, 0.0]
        self.gripper_state = "open"
        self.current_action = None
        self.action_queue = queue.Queue()
        self.executor_running = False
    
    async def initialize(self):
        """Initialize execution component"""
        self.system_bus.subscribe("plan_generated", self.handle_plan_generated)
        self.system_bus.subscribe("execute_action", self.handle_execute_action)
        self.executor_running = True
        self.status = ComponentStatus.READY
        logger.info(f"{self.name} initialized")
    
    async def process(self, message: SystemMessage):
        """Process execution-related messages"""
        try:
            if message.message_type == "plan_generated":
                # Execute the generated plan
                plan = message.content
                await self._execute_plan(plan)
                return True
            elif message.message_type == "execute_action":
                # Execute a single action
                action = message.content
                success = await self._execute_action(action)
                self.system_bus.publish(SystemMessage(
                    "action_result",
                    {"action": action, "success": success},
                    source=self.name
                ))
                return success
            
            return False
        except Exception as e:
            self.report_error(e)
            return False
    
    async def _execute_plan(self, plan: Dict[str, Any]):
        """Execute a complete plan"""
        logger.info(f"Executing plan with {len(plan['steps'])} steps")
        
        for i, step in enumerate(plan["steps"]):
            logger.info(f"Executing step {i+1}: {step['description']}")
            
            # Check safety before executing each action
            if not self._is_action_safe(step):
                logger.error(f"Action {step['description']} is not safe, stopping plan")
                break
            
            success = await self._execute_action(step)
            
            if not success:
                logger.error(f"Action {step['description']} failed, stopping plan")
                break
            
            # Update robot state based on action
            self._update_robot_state(step)
            
            # Publish progress update
            self.system_bus.publish(SystemMessage(
                "plan_progress",
                {"step": i, "total": len(plan["steps"]), "completed": step["description"]},
                source=self.name
            ))
        
        # Plan completed
        self.system_bus.publish(SystemMessage(
            "plan_completed",
            {"plan": plan["goal"], "success": success},
            source=self.name
        ))
    
    async def _execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute a single action"""
        # Simulate action execution
        await asyncio.sleep(0.2)  # Simulate action time
        
        action_type = action["action"]
        
        if action_type == "navigate":
            if "target" in action:
                target = action["target"]
                if isinstance(target, list) and len(target) >= 2:
                    self.robot_position[0] = target[0]
                    self.robot_position[1] = target[1]
                    logger.info(f"Navigated to position: {target}")
                    return True
        elif action_type == "move_to":
            # In simulation, just log
            logger.info(f"Moved to: {action.get('target', 'unknown')}")
            return True
        elif action_type == "grasp":
            self.gripper_state = "closed"
            logger.info("Grasped object")
            return True
        elif action_type == "detect":
            # In simulation, assume detection succeeds
            logger.info(f"Detected: {action.get('target', 'unknown')}")
            return True
        
        return False  # Unknown action or failed
    
    def _is_action_safe(self, action: Dict[str, Any]) -> bool:
        """Check if an action is safe to execute"""
        # In a real system, this would interface with safety manager
        # For simulation, assume all actions are safe
        return True
    
    def _update_robot_state(self, action: Dict[str, Any]):
        """Update robot state based on action"""
        action_type = action["action"]
        
        if action_type == "navigate" and "target" in action:
            target = action["target"]
            if isinstance(target, list) and len(target) >= 2:
                self.robot_position[0] = target[0]
                self.robot_position[1] = target[1]
        elif action_type == "grasp":
            self.gripper_state = "closed"
    
    def handle_plan_generated(self, message: SystemMessage):
        """Handle plan generation"""
        if message.message_type == "plan_generated":
            asyncio.create_task(self.process(message))
    
    def handle_execute_action(self, message: SystemMessage):
        """Handle single action execution"""
        if message.message_type == "execute_action":
            asyncio.create_task(self.process(message))
    
    def stop_execution(self):
        """Stop current execution"""
        self.executor_running = False


class SystemMonitor:
    """Monitors overall system health and performance"""
    
    def __init__(self, system_bus: SystemBus):
        self.system_bus = system_bus
        self.system_bus.subscribe("component_status", self.handle_component_status)
        self.system_bus.subscribe("error_report", self.handle_error_report)
        self.system_bus.subscribe("performance_metrics", self.handle_performance_metrics)
        
        self.component_statuses = {}
        self.error_log = []
        self.performance_history = []
        self.health_score = 100.0
        self.running = True
    
    def handle_component_status(self, message: SystemMessage):
        """Handle component status updates"""
        if message.message_type == "component_status":
            self.component_statuses[message.source] = message.content
            self._update_overall_health()
    
    def handle_error_report(self, message: SystemMessage):
        """Handle error reports"""
        if message.message_type == "error_report":
            self.error_log.append({
                "timestamp": message.timestamp,
                "component": message.source,
                "error": message.content
            })
            self._update_overall_health()
    
    def handle_performance_metrics(self, message: SystemMessage):
        """Handle performance metrics"""
        if message.message_type == "performance_metrics":
            self.performance_history.append({
                "timestamp": message.timestamp,
                "component": message.source,
                "metrics": message.content
            })
    
    def _update_overall_health(self):
        """Update overall system health score"""
        # Calculate health based on component statuses
        active_components = len([status for status in self.component_statuses.values() 
                                if status in [ComponentStatus.READY, ComponentStatus.PROCESSING]])
        total_components = len(self.component_statuses)
        
        if total_components > 0:
            component_health = (active_components / total_components) * 100
        else:
            component_health = 100
        
        # Factor in errors
        recent_errors = len([e for e in self.error_log if time.time() - e["timestamp"] < 60])  # Last minute
        error_penalty = min(50, recent_errors * 10)  # Up to 50% penalty
        
        self.health_score = max(0, component_health - error_penalty)
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get a comprehensive health report"""
        return {
            "health_score": self.health_score,
            "component_statuses": self.component_statuses,
            "error_count": len(self.error_log),
            "recent_errors": self.error_log[-5:],  # Last 5 errors
            "active_components": len([s for s in self.component_statuses.values() 
                                     if s in [ComponentStatus.READY, ComponentStatus.PROCESSING]]),
            "performance_metrics": self.performance_history[-10:]  # Last 10 metrics
        }


class AutonomousHumanoidSystem:
    """Complete autonomous humanoid robot system"""
    
    def __init__(self):
        # Initialize system bus
        self.system_bus = SystemBus()
        
        # Initialize components
        self.safety_manager = SafetyManager("SafetyManager", self.system_bus)
        self.voice_component = VoiceProcessingComponent("VoiceProcessor", self.system_bus)
        self.perception_component = PerceptionComponent("Perception", self.system_bus)
        self.planning_component = PlanningComponent("Planner", self.system_bus)
        self.execution_component = ExecutionComponent("Executor", self.system_bus)
        
        # Initialize monitor
        self.monitor = SystemMonitor(self.system_bus)
        
        # System state
        self.running = False
        self.start_time = 0.0
        
        # Component list for initialization
        self.components = [
            self.safety_manager,
            self.voice_component,
            self.perception_component,
            self.planning_component,
            self.execution_component
        ]
    
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing autonomous humanoid system...")
        
        self.start_time = time.time()
        
        # Initialize all components concurrently
        init_tasks = [comp.initialize() for comp in self.components]
        results = await asyncio.gather(*init_tasks, return_exceptions=True)
        
        # Check initialization results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Component {self.components[i].name} failed to initialize: {result}")
            else:
                logger.info(f"Component {self.components[i].name} initialized successfully")
        
        # Start the system bus
        self.system_bus.start()
        
        logger.info("Autonomous humanoid system initialized")
    
    async def run(self):
        """Run the system continuously"""
        logger.info("Starting autonomous humanoid system operation...")
        self.running = True
        
        # Simulate system operation
        try:
            while self.running:
                # Periodically request system updates
                if time.time() - self.start_time > 5:  # After 5 seconds
                    # Publish a sample voice command for demonstration
                    self.system_bus.publish(SystemMessage(
                        "voice_input",
                        b"simulated_audio_data",  # In real system, this would be actual audio
                        source="SimulatedSource"
                    ))
                
                # Publish system status periodically
                self.system_bus.publish(SystemMessage(
                    "system_status_request",
                    {},
                    source="System"
                ))
                
                await asyncio.sleep(1)  # Update every second
        
        except asyncio.CancelledError:
            logger.info("System operation cancelled")
        except Exception as e:
            logger.error(f"System operation error: {e}")
    
    def stop(self):
        """Stop the system"""
        logger.info("Stopping autonomous humanoid system...")
        self.running = False
        
        # Stop execution component
        self.execution_component.stop_execution()
        
        # Stop system bus
        self.system_bus.stop()
        
        logger.info("Autonomous humanoid system stopped")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "system_uptime": time.time() - self.start_time if self.start_time > 0 else 0,
            "running": self.running,
            "components": {comp.name: comp.get_status().value for comp in self.components},
            "system_health": self.monitor.get_system_health_report()
        }


def main():
    """Example usage of the system architecture"""
    print("Autonomous Humanoid System Architecture")
    print("=" * 50)
    
    # Create and initialize the system
    robot_system = AutonomousHumanoidSystem()
    
    async def run_system():
        # Initialize the system
        await robot_system.initialize()
        
        # Run the system for a short time in this example
        run_task = asyncio.create_task(robot_system.run())
        
        # Let it run for 30 seconds then stop
        await asyncio.sleep(30)
        
        # Stop the system
        robot_system.stop()
        
        # Wait for the run task to complete
        try:
            await asyncio.wait_for(run_task, timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("Run task did not complete in time, cancelling...")
            run_task.cancel()
    
    try:
        # Run the system
        asyncio.run(run_system())
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    
    # Print final system info
    system_info = robot_system.get_system_info()
    print(f"\nFinal system info:")
    print(f"  Uptime: {system_info['system_uptime']:.2f}s")
    print(f"  Running: {system_info['running']}")
    print(f"  Components: {system_info['components']}")
    print(f"  Health score: {system_info['system_health']['health_score']}")
    
    print("\nSystem architecture implemented with:")
    print("- Modular components with well-defined interfaces")
    print("- Central message bus for communication")
    print("- Safety-first design with monitoring")
    print("- Asynchronous processing for real-time performance")
    print("- Error handling and health monitoring")
    print("\nThe system demonstrates proper software architecture")
    print("principles applied to robotics systems.")


if __name__ == "__main__":
    main()
```

## Simulation Demonstration

This implementation demonstrates proper system design principles for autonomous robotic systems, featuring modularity, clear interfaces, safety mechanisms, and asynchronous processing. The architecture allows for independent development and testing of individual components while maintaining the ability to work as a unified system. The system can be integrated with ROS 2 and real robotic platforms.

## Hands-On Lab: System Architecture Implementation

In this lab, you'll implement and test the complete system architecture:

1. Implement the modular component architecture
2. Create the central message bus for communication
3. Implement safety monitoring and error handling
4. Test component integration and communication
5. Evaluate system performance and reliability

### Required Equipment:
- ROS 2 Humble environment
- Python development environment
- (Optional) Robot simulation environment

### Instructions:
1. Create a new ROS 2 package: `ros2 pkg create --build-type ament_python system_architecture_lab`
2. Implement the SystemComponent base class and message bus
3. Create individual components (SafetyManager, VoiceProcessor, etc.)
4. Implement the main system orchestrator
5. Test component communication and message passing
6. Add monitoring and health reporting features
7. Implement error handling and recovery procedures
8. Document your system architecture decisions

## Common Pitfalls & Debugging Notes

- **Tight Coupling**: Ensure components remain independent and loosely coupled
- **Message Floods**: Implement proper message rate limiting and buffering
- **Race Conditions**: Use proper synchronization mechanisms for shared resources
- **Deadlocks**: Avoid circular dependencies between components
- **Memory Leaks**: Properly manage resources and message queues
- **Initialization Order**: Components may depend on others being ready first
- **Performance Bottlenecks**: Monitor system performance to identify bottlenecks

## Summary & Key Terms

**Key Terms:**
- **System Architecture**: High-level structure of the robotic system
- **Modular Design**: Breaking system into independent, replaceable components
- **Message Bus**: Central communication mechanism between components
- **Component Interface**: Well-defined inputs and outputs for modules
- **Safety-First Design**: Prioritizing safety in all system decisions
- **Decoupled Architecture**: Components independent of each other
- **System Health Monitoring**: Continuous assessment of system status

## Further Reading & Citations

1. Shaw, M., & Garlan, D. (1996). "Software Architecture: Perspectives on an Emerging Discipline." Prentice Hall.
2. Bass, L., Clements, P., & Kazman, R. (2012). "Software Architecture in Practice" (3rd ed.). Addison-Wesley.
3. Pahl, C., & Dustdar, S. (2006). "A self-protection approach for business process management systems." International Conference on Business Process Management.
4. Gamma, E., et al. (1995). "Design Patterns: Elements of Reusable Object-Oriented Software." Addison-Wesley.

## Assessment Questions

1. Explain the benefits of a modular architecture for robotic systems.
2. How does the message bus architecture enable component communication?
3. What safety measures are built into the system design?
4. Describe how you would extend this architecture to support multiple robots.
5. What patterns would you use to ensure scalability of the robotic system?

---
**Previous**: [Capstone Project Introduction](./intro.md)  
**Next**: [Appendices - Hardware Requirements](../08-appendices/hardware.md)