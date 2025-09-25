# core/window_manager.py
import win32gui
import win32ui
import win32con
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class Window:
    handle: int
    title: str
    x: int
    y: int
    width: int
    height: int

class WindowManager:
    """Manages game windows and their coordinates"""
    
    def __init__(self):
        self.windows: dict[int, Window] = {}
        
    def find_game_windows(self, window_title: str) -> list[Window]:
        """Find all game windows matching the title pattern"""
        def enum_callback(hwnd, results):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if window_title in window_text:
                    rect = win32gui.GetWindowRect(hwnd)
                    x, y, right, bottom = rect
                    width = right - x
                    height = bottom - y
                    window = Window(hwnd, window_text, x, y, width, height)
                    self.windows[hwnd] = window
                    results.append(window)
        
        results = []
        win32gui.EnumWindows(enum_callback, results)
        return results
    
    def capture_window(self, window: Window) -> np.ndarray:
        """Capture screenshot of specific window"""
        hwnd = window.handle
        
        # Get window DC
        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        
        # Create bitmap object
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(dcObj, window.width, window.height)
        cDC.SelectObject(bmp)
        
        # Copy window content
        cDC.BitBlt((0, 0), (window.width, window.height), dcObj, (0, 0), win32con.SRCCOPY)
        
        # Convert to numpy array
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (window.height, window.width, 4)
        
        # Free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(bmp.GetHandle())
        
        return img

    def click_at(self, window: Window, x: float, y: float) -> None:
        """Convert relative coordinates to absolute and perform click"""
        abs_x = window.x + int(x * window.width)
        abs_y = window.y + int(y * window.height)
        # Implement click using pyautogui or win32api
        pass
