import time
import numpy as np
from PIL import ImageGrab
from pynput import keyboard, mouse
import os
import pickle
import random

class Buffer:
    def __init__(self, maxlen=100, save_dir="."):
        self.obs = []
        self.act = []
        self.pref = []
        self.save_dir = save_dir
        self.index = 0
        self.maxlen = maxlen

        # Create the save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
    def append(self, observation, action, preference):
        self.obs.append(observation)
        self.act.append(action)
        self.pref.append(preference)
        print(len(self.obs))
        if len(self.obs) >= self.maxlen:
            self.save_and_clear()

    def append_unique(value):
         self.unique_acts.add(value)
    
    def save_and_clear(self):
        file_path = os.path.join(self.save_dir, f"traj_{str(self.index)}.pkl")
        data = {'observations': self.obs, 'actions': self.act, 'preference':self.pref}
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
            print(file_path)

        self.obs.clear()
        self.act.clear()
        self.index += 1
        
    def get(self):
        return self.obs, self.act

#random.choice(['left', 'right'])


# Capture the screen
def capture_screen():
    #screenshot = ImageGrab.grab()
    #screenshot = screenshot.resize((256, 256))
    screenshot = ImageGrab.grab(bbox =(550, 290, 1420, 810))
    return np.array(screenshot)

def on_key_press(key):
    # Record keypress
    mouse_pos = mouse.Controller().position
    
    buffer.append(capture_screen(), (key.char if hasattr(key, 'char') else key.name),random.choice(['left', 'right']))

def on_click(x, y, button, pressed):
    # Record mouse click
    if button == mouse.Button.left:
        buffer.append(capture_screen(), ('left click') ,random.choice(['left', 'right']))
    elif button == mouse.Button.right:
        buffer.append(capture_screen(), ('right click') ,random.choice(['left', 'right']))

def on_move(x, y):
    # Record mouse position
    global last_x, last_y, i
    v = ''
    h = ''
    i+=1
    if i >20:
        i = 0
        if last_x is not None and last_y is not None:
            delta_x = x - last_x
            delta_y = y - last_y

            if delta_x > 0:
                v = 'right'
            elif delta_x < 0:
                v = 'left'

            if delta_y > 0:
                h = 'down'
            elif delta_y < 0:
                h = 'up'
        buffer.append(capture_screen(), (v,h),random.choice(['left', 'right']))
    last_x, last_y = x, y
    

# Initialize the last known mouse coordinates as None
last_x, last_y = None, None
buffer = Buffer()
i = 0
# Create mouse and keyboard listeners
mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
keyboard_listener = keyboard.Listener(on_press=on_key_press)

# Start the listeners
mouse_listener.start()
keyboard_listener.start()

time.sleep(0.1)

# Wait for the listeners to finish
mouse_listener.join()
keyboard_listener.join()
