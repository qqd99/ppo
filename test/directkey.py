import ctypes
import time
# Virtual key code mapping
KEY_MAP = {
    'A': 0x41,
    'B': 0x42,
    'C': 0x43,
    'D': 0x44,
    'E': 0x45,
    'F': 0x46,
    'G': 0x47,
    'H': 0x48,
    'I': 0x49,
    'J': 0x4A,
    'K': 0x4B,
    'L': 0x4C,
    'M': 0x4D,
    'N': 0x4E,
    'O': 0x4F,
    'P': 0x50,
    'Q': 0x51,
    'R': 0x52,
    'S': 0x53,
    'T': 0x54,
    'U': 0x55,
    'V': 0x56,
    'W': 0x57,
    'X': 0x58,
    'Y': 0x59,
    'Z': 0x5A,
    '0': 0x30,
    '1': 0x31,
    '2': 0x32,
    '3': 0x33,
    '4': 0x34,
    '5': 0x35,
    '6': 0x36,
    '7': 0x37,
    '8': 0x38,
    '9': 0x39,
    'ENTER': 0x0D,
    'SPACE': 0x20,
    'ESCAPE': 0x1B,
    'BACKSPACE': 0x08,
    'TAB': 0x09,
    'SHIFT': 0x10,
    'CTRL': 0x11,
    'ALT': 0x12,
    'CAPSLOCK': 0x14,
    'NUMLOCK': 0x90,
    'SCROLLLOCK': 0x91,
    'UP': 0x26,
    'DOWN': 0x28,
    'LEFT': 0x25,
    'RIGHT': 0x27,
    'F1': 0x70,
    'F2': 0x71,
    'F3': 0x72,
    'F4': 0x73,
    'F5': 0x74,
    'F6': 0x75,
    'F7': 0x76,
    'F8': 0x77,
    'F9': 0x78,
    'F10': 0x79,
    'F11': 0x7A,
    'F12': 0x7B,
    'MOUSE_EVENT_LEFTDOWN' : 0x0002,
    'MOUSE_EVENT_LEFTUP' : 0x0004,
    'MOUSE_EVENT_RIGHTDOWN' : 0x0008,
    'MOUSE_EVENT_RIGHTUP' : 0x0010,
}


# Constants for mouse events
MOUSE_EVENT_LEFTDOWN = 0x0002
MOUSE_EVENT_LEFTUP = 0x0004
MOUSE_EVENT_RIGHTDOWN = 0x0008
MOUSE_EVENT_RIGHTUP = 0x0010

# Function to move the mouse to specified coordinates
def move_mouse(x, y):
    user32 = ctypes.windll.user32
    user32.SetCursorPos(x, y)

# Function to perform a left click
def left_click():
    user32 = ctypes.windll.user32
    user32.mouse_event(MOUSE_EVENT_LEFTDOWN, 0, 0, 0, 0)
    user32.mouse_event(MOUSE_EVENT_LEFTUP, 0, 0, 0, 0)

# Function to perform a right click
def right_click():
    user32= ctypes.windll.user32
    user32.mouse_event(MOUSE_EVENT_RIGHTDOWN, 0, 0, 0, 0)
    user32.mouse_event(MOUSE_EVENT_RIGHTUP, 0, 0, 0, 0)

# Function to press a key
def press_key(key):
    user32 = ctypes.windll.user32
    virtual_key = KEY_MAP.get(key.upper())
    if virtual_key is not None:
        user32.keybd_event(virtual_key, 0, 0, 0)
        #user32.keybd_event(virtual_key, 0, 2, 0)  # Release the key

# Function to release a key
def release_key(key):
    user32 = ctypes.windll.user32
    virtual_key = KEY_MAP.get(key.upper())
    if virtual_key is not None:
        user32.keybd_event(virtual_key, 0, 2, 0)  # Release the key

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)
# Example usage
press_key('j')
right_click()



