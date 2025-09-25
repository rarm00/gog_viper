# Directory structure:
# game_automation/
# ├── __init__.py
# ├── core/
# │   ├── __init__.py
# │   ├── window_manager.py
# │   ├── element_detector.py
# │   ├── action_manager.py
# │   └── game_state.py
# ├── ml/
# │   ├── __init__.py
# │   ├── model.py
# │   └── dataset.py
# ├── config/
# │   ├── __init__.py
# │   └── settings.py
# └── utils/
#     ├── __init__.py
#     └── logger.py

# Example usage:
from core.window_manager import WindowManager
from core.element_detection import ElementDetector
from core.action_manager import ActionManager


if __name__ == "__main__":
    # Initialize components
    window_manager = WindowManager()
    element_detector = ElementDetector("path/to/model")
    action_manager = ActionManager(window_manager, element_detector)
    
    # Find game windows
    windows = window_manager.find_game_windows("Game Title")
    
    if windows:
        # Load action sequences
        action_manager.load_sequences("config/sequences.json")
        
        # Execute mining sequence on first window
        success = action_manager.execute_sequence("mine", windows[0])