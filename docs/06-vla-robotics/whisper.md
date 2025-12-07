# Whisper for Voice Recognition

Whisper is a state-of-the-art automatic speech recognition (ASR) system developed by OpenAI that plays a crucial role in Vision-Language-Action (VLA) robotic systems. It enables robots to understand natural language commands through speech, providing an intuitive and human-like interaction modality. Whisper's robust performance across multiple languages and its ability to handle varying audio conditions make it valuable for robotics applications.

## Learning Outcomes

After completing this section, you should be able to:
- Understand the architecture and capabilities of the Whisper ASR system
- Integrate Whisper into robotic systems for voice command processing
- Preprocess audio inputs for optimal Whisper performance
- Handle Whisper outputs and integrate them with downstream NLP systems
- Address common challenges in deploying Whisper in robotics environments
- Evaluate the accuracy and latency of speech recognition in robotic contexts

## Core Concepts

### Transformer Architecture
Whisper is built on a transformer architecture that consists of:
- **Encoder**: Processes audio inputs with convolutional feature extraction followed by transformer layers
- **Decoder**: Generates text outputs using cross-attention to the encoded audio features
- **Multilingual Capability**: Trained on 98+ languages, enabling recognition across diverse linguistic contexts

### Audio Preprocessing
For effective Whisper performance, audio preprocessing includes:
- **Resampling**: Converting audio to the required sample rate (typically 16kHz)
- **Normalization**: Adjusting audio levels for consistent processing
- **Noise Reduction**: Filtering out background noise when possible
- **Audio Segmentation**: Breaking long audio streams into appropriate chunks

### Confidence Scoring
Whisper provides confidence scores for transcriptions, which is important for robotics applications where:
- Low-confidence transcriptions may require confirmation
- High-confidence transcriptions can be processed immediately
- Confidence levels can trigger different system behaviors

### Real-time vs Batch Processing
Whisper can be used in:
- **Batch Mode**: For processing complete audio clips with high accuracy
- **Streaming Mode**: For real-time processing (with additional engineering considerations)

## Equations and Models

### Audio Feature Extraction
Whisper processes audio by transforming it into features:

```
F = STFT(x, window, n_fft)
```

Where:
- `F` is the spectrogram of the audio signal
- `STFT` is the Short-Time Fourier Transform
- `x` is the input audio signal
- `window` is the window function (typically Hann window)
- `n_fft` is the FFT size

### Transformer Attention Mechanism
The attention mechanism in Whisper's transformer layers:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- `Q` is the query matrix (input representation)
- `K` is the key matrix (input representation)
- `V` is the value matrix (input representation)
- `d_k` is the dimension of the key vectors

### Confidence in Transcription
The confidence of a generated sequence can be approximated by:

```
Confidence(S) = Π_i P(w_i | w_1, ..., w_{i-1}, F)
```

Where:
- `S` is the transcript sequence `w_1, w_2, ..., w_n`
- `F` is the audio features
- `P(w_i | ...)` is the probability of word `w_i` given context

## Code Example: Whisper Integration for Robotics

Here's an example of integrating Whisper for voice recognition in a robotics system:

```python
import asyncio
import numpy as np
import torch
import whisper
import speech_recognition as sr
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class VoiceCommand:
    """Represents a recognized voice command with metadata"""
    text: str
    confidence: float
    timestamp: float
    language: str
    audio_duration: float


class WhisperRobotListener:
    """
    Whisper-based voice command recognizer for robotics
    """
    def __init__(self, model_size="base", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Whisper-based voice recognition system
        
        :param model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
        :param device: Device to run the model on ('cpu', 'cuda', 'mps')
        """
        self.device = device
        self.model = whisper.load_model(model_size, device=device)
        
        # Initialize the speech recognizer for audio capture
        self.speech_recognizer = sr.Recognizer()
        self.speech_recognizer.energy_threshold = 300  # Adjust based on environment
        self.speech_recognizer.dynamic_energy_threshold = True
        
        # Audio queue for streaming
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.listening_thread = None
        
        # Statistics
        self.stats = {
            'total_commands': 0,
            'avg_processing_time': 0.0,
            'avg_confidence': 0.0,
            'language_distribution': {}
        }
        
        print(f"Whisper model loaded on {device}")
    
    def transcribe_audio_file(self, audio_file_path):
        """
        Transcribe audio from a file using Whisper
        
        :param audio_file_path: Path to audio file
        :return: VoiceCommand with transcription and metadata
        """
        start_time = time.time()
        
        # Load and transcribe audio
        result = self.model.transcribe(audio_file_path)
        
        processing_time = time.time() - start_time
        
        # Extract transcription details
        text = result['text'].strip()
        confidence = self._estimate_confidence(result)
        language = result.get('language', 'unknown')
        audio_duration = result.get('duration', 0.0)
        
        command = VoiceCommand(
            text=text,
            confidence=confidence,
            timestamp=time.time(),
            language=language,
            audio_duration=audio_duration
        )
        
        # Update statistics
        self._update_statistics(command, processing_time)
        
        return command
    
    def transcribe_audio_buffer(self, audio_buffer):
        """
        Transcribe raw audio buffer using Whisper
        
        :param audio_buffer: Audio data in bytes
        :return: VoiceCommand with transcription and metadata
        """
        # Convert audio buffer to audio data
        audio_data = sr.AudioData(audio_buffer, 16000, 2)  # Assuming 16kHz, 16-bit
        
        # Save to temporary file (Whisper expects file input)
        import io
        import wave
        temp_filename = f"temp_audio_{int(time.time())}.wav"
        
        # Write raw audio data to WAV file
        with wave.open(temp_filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(audio_buffer)
        
        try:
            # Transcribe the temporary file
            result = self.model.transcribe(temp_filename)
            
            # Clean up
            import os
            os.remove(temp_filename)
            
            # Extract transcription details
            text = result['text'].strip()
            confidence = self._estimate_confidence(result)
            language = result.get('language', 'unknown')
            audio_duration = result.get('duration', len(audio_buffer) / (16000 * 2))  # Estimate
            
            command = VoiceCommand(
                text=text,
                confidence=confidence,
                timestamp=time.time(),
                language=language,
                audio_duration=audio_duration
            )
            
            return command
        except Exception as e:
            print(f"Error transcribing audio buffer: {e}")
            import os
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return None
    
    def _estimate_confidence(self, result):
        """
        Estimate confidence from Whisper result
        Note: Whisper doesn't provide direct confidence scores, so we'll use a heuristic
        """
        # In real implementations, you might use token probabilities or other metrics
        # For now, using a simple heuristic based on the most common tokens
        if 'segments' in result and len(result['segments']) > 0:
            # Average the temperatures across segments as a proxy for confidence
            temp_sum = sum([seg.get('temperature', 0.0) for seg in result['segments']])
            avg_temp = temp_sum / len(result['segments'])
            # Convert temperature to confidence (lower temperature = higher confidence)
            confidence = max(0.0, 1.0 - avg_temp)
        else:
            confidence = 0.8  # Default confidence
        
        return confidence
    
    def _update_statistics(self, command, processing_time):
        """Update internal statistics"""
        self.stats['total_commands'] += 1
        old_avg = self.stats['avg_processing_time']
        self.stats['avg_processing_time'] = (
            (old_avg * (self.stats['total_commands'] - 1) + processing_time) / 
            self.stats['total_commands']
        )
        
        old_conf_avg = self.stats['avg_confidence']
        self.stats['avg_confidence'] = (
            (old_conf_avg * (self.stats['total_commands'] - 1) + command.confidence) / 
            self.stats['total_commands']
        )
        
        # Update language distribution
        if command.language in self.stats['language_distribution']:
            self.stats['language_distribution'][command.language] += 1
        else:
            self.stats['language_distribution'][command.language] = 1
    
    def start_continuous_listening(self, callback_func=None):
        """
        Start continuous listening in a background thread
        
        :param callback_func: Function to call when a command is recognized
        """
        if self.is_listening:
            print("Already listening")
            return
        
        self.is_listening = True
        self.listening_thread = threading.Thread(
            target=self._continuous_listening_worker, 
            args=(callback_func,)
        )
        self.listening_thread.start()
        
        print("Started continuous listening")
    
    def stop_continuous_listening(self):
        """Stop continuous listening"""
        self.is_listening = False
        if self.listening_thread:
            self.listening_thread.join()
        print("Stopped continuous listening")
    
    def _continuous_listening_worker(self, callback_func):
        """
        Worker function for continuous listening in background thread
        """
        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            self.speech_recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening for voice commands...")
            
            while self.is_listening:
                try:
                    # Listen for audio with timeout
                    audio = self.speech_recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                    # Convert to raw data for Whisper processing
                    audio_buffer = audio.get_raw_data()
                    
                    # Process with Whisper
                    command = self.transcribe_audio_buffer(audio_buffer)
                    
                    if command and command.text:
                        print(f"Heard: '{command.text}' (confidence: {command.confidence:.2f})")
                        
                        # Call the callback if provided
                        if callback_func:
                            try:
                                callback_func(command)
                            except Exception as e:
                                print(f"Error in callback function: {e}")
                
                except sr.WaitTimeoutError:
                    # This is expected when there's no speech
                    continue
                except sr.UnknownValueError:
                    print("Could not understand audio")
                    continue
                except sr.RequestError as e:
                    print(f"Error with speech recognition: {e}")
                    continue
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    continue
    
    def get_statistics(self):
        """Get current statistics about voice recognition performance"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.stats = {
            'total_commands': 0,
            'avg_processing_time': 0.0,
            'avg_confidence': 0.0,
            'language_distribution': {}
        }


class VoiceCommandProcessor:
    """
    Processor for voice commands that integrates with Whisper
    """
    def __init__(self):
        self.whisper_listener = WhisperRobotListener()
        self.command_history = []
        self.confidence_threshold = 0.7  # Minimum confidence to accept command
        self.understanding_phrases = [
            "I will", "Okay, I'll", "Sure, processing", "Got it, I'll"
        ]
        self.error_phrases = [
            "I didn't understand", "Could you repeat", "I'm sorry, I didn't catch"
        ]
    
    def process_voice_command(self, audio_input):
        """
        Process voice command from audio input and return processed result
        
        :param audio_input: Can be file path or audio buffer
        :return: Processed command with validation and context
        """
        if isinstance(audio_input, str):
            # File path
            command = self.whisper_listener.transcribe_audio_file(audio_input)
        else:
            # Audio buffer
            command = self.whisper_listener.transcribe_audio_buffer(audio_input)
        
        if command is None:
            return None
        
        # Validate command confidence
        if command.confidence < self.confidence_threshold:
            # Low confidence - request clarification
            return {
                'status': 'uncertain',
                'command': command,
                'feedback': f"I heard '{command.text}' but I'm not confident (confidence: {command.confidence:.2f}). Could you repeat that?"
            }
        
        # High confidence - process normally
        processed_result = self._process_valid_command(command)
        
        # Add to history
        self.command_history.append(processed_result)
        
        return processed_result
    
    def _process_valid_command(self, command):
        """
        Process a valid high-confidence command
        
        :param command: VoiceCommand object with high confidence
        :return: Dictionary with processed command information
        """
        # Here you would typically:
        # 1. Parse the command to extract intent and entities
        # 2. Validate against robot capabilities
        # 3. Generate a robot action plan
        # 4. Return appropriate response
        
        # Simple command parsing for demonstration
        text_lower = command.text.lower()
        intent = self._determine_intent(text_lower)
        
        # Extract relevant entities
        entities = self._extract_entities(text_lower)
        
        result = {
            'status': 'recognized',
            'command': command,
            'intent': intent,
            'entities': entities,
            'feedback': f"I will {self._generate_response(intent, entities)}",
            'timestamp': time.time()
        }
        
        return result
    
    def _determine_intent(self, text):
        """
        Determine the intent of the voice command (simplified parser)
        """
        if any(word in text for word in ['move', 'go', 'navigate', 'walk', 'come']):
            return 'navigation'
        elif any(word in text for word in ['pick', 'grasp', 'take', 'grab', 'lift']):
            return 'manipulation'
        elif any(word in text for word in ['turn', 'rotate', 'spin']):
            return 'rotation'
        elif any(word in text for word in ['stop', 'halt', 'pause']):
            return 'stop'
        elif any(word in text for word in ['follow', 'come after', ' accompany']):
            return 'follow'
        else:
            return 'unknown'
    
    def _extract_entities(self, text):
        """
        Extract relevant entities from the text (simplified extractor)
        """
        entities = {}
        
        # Object recognition
        objects = ['cup', 'box', 'ball', 'bottle', 'book', 'table', 'chair', 'door', 'window']
        for obj in objects:
            if obj in text:
                if 'objects' not in entities:
                    entities['objects'] = []
                entities['objects'].append(obj)
        
        # Location recognition
        locations = ['kitchen', 'living room', 'bedroom', 'office', 'table', 'counter', 'shelf']
        for loc in locations:
            if loc in text:
                if 'locations' not in entities:
                    entities['locations'] = []
                entities['locations'].append(loc)
        
        # Direction recognition
        directions = ['left', 'right', 'forward', 'backward', 'up', 'down']
        for dir in directions:
            if dir in text:
                if 'directions' not in entities:
                    entities['directions'] = []
                entities['directions'].append(dir)
        
        # Distance recognition (simplified)
        distances = ['close', 'near', 'far', 'behind', 'in front', 'next to']
        for dist in distances:
            if dist in text:
                if 'distances' not in entities:
                    entities['distances'] = []
                entities['distances'].append(dist)
        
        return entities
    
    def _generate_response(self, intent, entities):
        """
        Generate a verbal response based on intent and entities
        """
        if intent == 'navigation':
            if entities.get('locations'):
                return f"move to the {entities['locations'][0]}"
            elif entities.get('directions'):
                return f"move to the {entities['directions'][0]}"
            else:
                return "move to the requested location"
        elif intent == 'manipulation':
            if entities.get('objects'):
                return f"grasp the {entities['objects'][0]}"
            else:
                return "perform the requested manipulation"
        elif intent == 'rotation':
            if entities.get('directions'):
                return f"turn to the {entities['directions'][0]}"
            else:
                return "rotate as requested"
        else:
            return "execute the command"
    
    def start_robot_assistant(self, callback_func=None):
        """
        Start the robot assistant that listens and processes voice commands
        """
        def whisper_callback(command):
            if callback_func:
                # Process the command and call user callback
                result = self._process_valid_command(command)
                callback_func(result)
        
        print("Starting robot voice assistant...")
        self.whisper_listener.start_continuous_listening(whisper_callback)
    
    def stop_robot_assistant(self):
        """Stop the robot assistant"""
        self.whisper_listener.stop_continuous_listening()
        print("Stopped robot voice assistant")


def main():
    """Example usage of Whisper integration for robotics"""
    print("Whisper Voice Recognition for Robotics Example")
    
    # Initialize the voice command processor
    processor = VoiceCommandProcessor()
    
    # Example 1: Process a pre-recorded command (simulated)
    print("\n--- Example 1: Processing audio command ---")
    
    # Instead of actual audio, we'll simulate by creating a mock command
    # In practice, you'd call processor.process_voice_command(audio_file_path)
    mock_command = VoiceCommand(
        text="Please go to the kitchen and bring me a cup",
        confidence=0.85,
        timestamp=time.time(),
        language="en",
        audio_duration=3.5
    )
    
    # Process the mock command
    result = {
        'status': 'recognized',
        'command': mock_command,
        'intent': processor._determine_intent(mock_command.text.lower()),
        'entities': processor._extract_entities(mock_command.text.lower()),
        'feedback': f"I will {processor._generate_response(processor._determine_intent(mock_command.text.lower()), processor._extract_entities(mock_command.text.lower()))}",
        'timestamp': time.time()
    }
    
    print(f"Recognized command: '{mock_command.text}'")
    print(f"Intent: {result['intent']}")
    print(f"Entities: {result['entities']}")
    print(f"Feedback: {result['feedback']}")
    
    # Example 2: Show statistics
    print("\n--- Example 2: Voice recognition statistics ---")
    stats = processor.whisper_listener.get_statistics()
    print(f"Total commands processed: {stats['total_commands']}")
    print(f"Average processing time: {stats['avg_processing_time']:.3f}s")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    print(f"Language distribution: {stats['language_distribution']}")
    
    # Example 3: Simulate robot assistant
    print("\n--- Example 3: Robot assistant simulation ---")
    
    def command_callback(result):
        print(f"Robot processing: '{result['command'].text}'")
        print(f"Intent: {result['intent']}, Confidence: {result['command'].confidence:.2f}")
    
    # In a real system, this would start continuous listening
    # processor.start_robot_assistant(command_callback)
    
    # For this example, just show the available functionality
    print("Robot assistant functionality ready")
    print("Commands will be processed as they are recognized")
    
    print("\nWhisper voice recognition integration example completed")


if __name__ == "__main__":
    main()
```

## Simulation Demonstration

This implementation demonstrates how Whisper can be integrated into a robotics system for voice recognition. The system handles audio preprocessing, transcription, confidence estimation, and integration with downstream language processing. The code can be combined with ROS 2 to create voice-controlled robotic systems.

## Hands-On Lab: Whisper Integration for Robotics

In this lab, you'll implement and test Whisper integration with a robotic system:

1. Set up Whisper model for robotic voice recognition
2. Implement audio preprocessing pipeline
3. Test Whisper performance with various audio conditions
4. Integrate Whisper outputs with robotic action planning
5. Evaluate the system's robustness to noise and accents

### Required Equipment:
- ROS 2 Humble environment
- CUDA-compatible GPU (for efficient Whisper processing)
- Microphone for audio input
- Whisper models installed (`pip install openai-whisper`)
- Speech Recognition library (`pip install SpeechRecognition`)

### Instructions:
1. Create a new ROS 2 package: `ros2 pkg create --build-type ament_python whisper_robotic_control`
2. Implement the WhisperRobotListener and VoiceCommandProcessor classes
3. Create a launch file to start the voice recognition node
4. Test with different voice commands in a quiet environment
5. Evaluate system performance with varying audio quality
6. Implement confidence-based command validation
7. Integrate with a robot simulator (e.g., TurtleBot3) for real testing
8. Document the recognition accuracy and response times

## Common Pitfalls & Debugging Notes

- **Audio Quality**: Whisper performance degrades significantly with poor audio quality
- **Computational Resources**: Whisper models can be computationally intensive; select appropriate model size
- **Latency**: Real-time applications require careful consideration of processing time
- **Microphone Setup**: Proper microphone placement and audio input configuration are crucial
- **Background Noise**: Implement noise filtering if operating in noisy environments
- **Language Settings**: Ensure Whisper model matches the expected language
- **Memory Management**: Large Whisper models consume significant memory

## Summary & Key Terms

**Key Terms:**
- **Whisper**: OpenAI's automatic speech recognition system
- **Automatic Speech Recognition (ASR)**: Technology that converts speech to text
- **Transformer Architecture**: Neural network architecture used in Whisper
- **Confidence Scoring**: Measure of reliability for ASR outputs
- **Audio Preprocessing**: Steps to prepare audio for ASR processing
- **Streaming ASR**: Real-time speech recognition on audio streams
- **Language Identification**: Determining the language of spoken input

## Further Reading & Citations

1. Radford, A., et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision." arXiv preprint arXiv:2212.04356.
2. OpenAI. (2022). "Introducing Whisper." OpenAI Blog. https://openai.com/research/whisper
3. Zhang, Y., et al. (2022). "Transformer-Based Acoustic Modeling for Speech Recognition." IEEE Signal Processing Magazine.
4. Hori, T., et al. (2022). "End-to-End Speech Recognition and Understanding with Transformers." IEEE International Conference on Acoustics, Speech and Signal Processing.

## Assessment Questions

1. Explain how Whisper's transformer architecture differs from traditional speech recognition systems.
2. What are the computational requirements for running different Whisper model sizes?
3. Describe how to preprocess audio data for optimal Whisper performance in robotics.
4. How can confidence scores from Whisper be used to improve robotic command execution?
5. What are the key challenges in deploying Whisper for real-time robotic applications?

---
**Previous**: [Introduction to Vision-Language-Action Robotics](./intro.md)  
**Next**: [LLM-Based Planning for Robotics](./llm-planning.md)