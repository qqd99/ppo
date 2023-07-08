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

def press_key(key):
    user32 = ctypes.windll.user32
    # Check if the key is a mouse event
    if key.startswith('MOUSE_EVENT'):
        if key == 'MOUSE_EVENT_LEFTDOWN':
            user32.mouse_event(0x0002, 0, 0, 0, 0)
            user32.mouse_event(0x0004, 0, 0, 0, 0)

        elif key == 'MOUSE_EVENT_RIGHTDOWN':
            user32.mouse_event(0x0008, 0, 0, 0, 0)
            user32.mouse_event(0x0010, 0, 0, 0, 0)

    else:
        # Convert key to virtual key code
        key_code = KEY_MAP.get(key.upper())
        if key_code is not None:
            # Simulate key press
            user32.keybd_event(key_code, 0, 0, 0)
            time.sleep(0.05)
            user32.keybd_event(key_code, 0, 2, 0)
    
for i in list(range(2))[::-1]:
    print(i+1)
    time.sleep(1)
last_time = time.time()
press_key('MOUSE_EVENT_LEFTDOWN')
print(last_time - time.time())

for i in 'hello':
    last_time = time.time()
    press_key(i)
    print(last_time - time.time())
    time.sleep(0.01) 
