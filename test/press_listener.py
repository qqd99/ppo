from pynput import keyboard, mouse

# Lists to store recorded events
recorded_key_presses = []
recorded_mouse_clicks = []

# Keyboard event listener
def on_key_press(key):
    recorded_key_presses.append(('press', key))

def on_key_release(key):
    recorded_key_presses.append(('release', key))

# Mouse event listener
def on_mouse_click(x, y, button, pressed):
    recorded_mouse_clicks.append((x, y, button, pressed))

# Create keyboard and mouse event listeners
keyboard_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
mouse_listener = mouse.Listener(on_click=on_mouse_click)

# Start listening for events
keyboard_listener.start()
mouse_listener.start()

# Run your main code here
for i in range(1000):
    print(i)


# Print recorded events
print("Recorded Key Presses:")
for event in recorded_key_presses:
    event_type, key = event
    print(f"{event_type}: {key}")

print("\nRecorded Mouse Clicks:")
for event in recorded_mouse_clicks:
    x, y, button, pressed = event
    print(f"X: {x}, Y: {y}, Button: {button}, Pressed: {pressed}")

