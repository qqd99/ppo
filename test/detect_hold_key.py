from pynput.keyboard import Listener, Key
import time

def on_key_press(key):
    if key == Key.esc:
        # Stop the listener if the 'Esc' key is pressed
        return False

    # Record the time when the key is pressed
    print(key)
    global press_time
    press_time = time.time()

def on_key_release(key):
    # Calculate the time duration between press and release
    release_time = time.time()
    duration = release_time - press_time

    if duration < 0.5:
        # If the key is released quickly (tapped), append "tap" to the action
        action = f"tap {key}"
    else:
        # If the key is held down, append "hold" to the action
        action = f"hold {key}"

    # Do whatever you want with the action (e.g., append it to a buffer)
    print(action)

# Create a keyboard listener
keyboard_listener = Listener(on_press=on_key_press, on_release=on_key_release)

# Start the listener
keyboard_listener.start()

# Keep the listener running
keyboard_listener.join()
