# core/action_manager.py
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import json

@dataclass
class Action:
    name: str
    target: str
    required_elements: List[str]
    timeout: float = 5.0
    retry_interval: float = 0.5

class ActionSequence:
    def __init__(self, name: str):
        self.name = name
        self.actions: List[Action] = []
        
    def add_action(self, action: Action):
        self.actions.append(action)

class ActionManager:
    """Manages game actions and sequences"""
    
    def __init__(self, window_manager, element_detector):
        self.window_manager = window_manager
        self.element_detector = element_detector
        self.sequences: Dict[str, ActionSequence] = {}
        
    def load_sequences(self, config_path: str):
        """Load action sequences from config file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        for seq_name, seq_data in config['sequences'].items():
            sequence = ActionSequence(seq_name)
            for action_data in seq_data['actions']:
                action = Action(**action_data)
                sequence.add_action(action)
            self.sequences[seq_name] = sequence
    
    def execute_sequence(self, 
                        sequence_name: str, 
                        window,
                        max_retries: int = 3) -> bool:
        """Execute a sequence of actions on specified window"""
        if sequence_name not in self.sequences:
            raise ValueError(f"Unknown sequence: {sequence_name}")
            
        sequence = self.sequences[sequence_name]
        
        for action in sequence.actions:
            retry_count = 0
            while retry_count < max_retries:
                # Capture current window state
                image = self.window_manager.capture_window(window)
                
                # Detect required elements
                elements = self.element_detector.detect_elements(image)
                
                # Find target element
                target = next((e for e in elements 
                             if e.name == action.target 
                             and e.confidence > 0.7), None)
                
                if target:
                    # Execute click at relative coordinates
                    self.window_manager.click_at(window, target.x, target.y)
                    break
                    
                retry_count += 1
                time.sleep(action.retry_interval)
            
            if retry_count >= max_retries:
                return False
                
        return True
