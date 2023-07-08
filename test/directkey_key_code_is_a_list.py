import ctypes
import time

# Virtual key code
KEY_MAP =['0x41', '0x42', '0x43', '0x44', '0x45', '0x46', '0x47', '0x48', '0x49', '0x4a', '0x4b', '0x4c', '0x4d', '0x4e', '0x4f', '0x50', '0x51', '0x52', '0x53', '0x54', '0x55', '0x56', '0x57', '0x58', '0x59', '0x5a', '0x30', '0x31', '0x32', '0x33', '0x34', '0x35', '0x36', '0x37', '0x38', '0x39', '0xd', '0x20', '0x9', '0x10', '0x11', '0x12', '0x2', '0x4', '0x8', '0x10']

'''
def press_key(key):
    user32 = ctypes.windll.user32
    # Check if the key is a mouse event
    if key.startswith('MOUSE_EVENT'):
        print("yes")
        if key == 'MOUSE_EVENT_LEFTDOWN':
            user32.mouse_event(0x0002, 0, 0, 0, 0)
            user32.mouse_event(0x0004, 0, 0, 0, 0)

        elif key == 'MOUSE_EVENT_RIGHTDOWN':
            user32.mouse_event(0x0008, 0, 0, 0, 0)
            user32.mouse_event(0x0010, 0, 0, 0, 0)

    else:
        # Convert key to virtual key code
        key_code = key.upper()
        if key_code is not None:
            # Simulate key press
            user32.keybd_event(key_code, 0, 0, 0)
            time.sleep(0.05)
            user32.keybd_event(key_code, 0, 2, 0)
'''
def press_key(key):
    user32 = ctypes.windll.user32

    key_code = key.upper()
    if key_code is not None:
        # Simulate key press
        user32.keybd_event(key_code, 0, 0, 0)
        time.sleep(0.05)
        user32.keybd_event(key_code, 0, 2, 0)

for i in list(range(2))[::-1]:
    print(i+1)
    time.sleep(1)            
#press_key('MOUSE_EVENT_LEFTDOWN')

press_key(0x44)

