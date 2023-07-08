import numpy as np
import keyboard  # Import the keyboard module to simulate key presses
import pyautogui  # Import the pyautogui module to simulate mouse movements


def shoot_enemy(observation, enemy_boxes):
    # Determine whether or not to shoot based on the detected enemy boxes
    if len(enemy_boxes) > 0:
        # Get the coordinates of the center of each enemy box
        enemy_centers = [(int((box[0] + box[2])/2), int((box[1] + box[3])/2)) for box in enemy_boxes]

        # Get the coordinates of the center of the player's view
        player_center = (int(observation.shape[1]/2), int(observation.shape[0]/2))

        # Find the closest enemy to the player's view
        closest_enemy = min(enemy_centers, key=lambda c: np.linalg.norm(np.array(c) - np.array(player_center)))

        # Determine whether the closest enemy is within a certain distance threshold
        distance_threshold = 50
        if np.linalg.norm(np.array(closest_enemy) - np.array(player_center)) < distance_threshold:
            # If the enemy is close enough, take the shoot action
            return True

    # If no enemy is close enough, do not take the shoot action
    return False


def action_to_keyboard_command(action):
    # Define a dictionary that maps actions to keyboard commands
    action_to_key = {
        'move_left': 'left arrow',
        'move_right': 'right arrow',
        'jump': 'space',
        'shoot': 'left mouse down',
        'reload': 'r',
        'switch_weapon': 'tab'
        # Add more actions and corresponding keyboard commands as needed
    }
    
    # Look up the corresponding keyboard command for the given action
    if action in action_to_key:
        key = action_to_key[action]
    else:
        # If the action is not recognized, return an empty string
        key = ''
    
    # Simulate the key press using the keyboard module
    if key:
        keyboard.press(key)
        keyboard.release(key)
    
    # Return the keyboard command as a string
    return key

import pyautogui  # Import the pyautogui module to simulate keyboard commands

def send_keyboard_command(keyboard_command):
    # Simulate the keyboard command using the pyautogui module
    if keyboard_command == 'left arrow':
        pyautogui.keyDown('left')
        pyautogui.keyUp('left')
    elif keyboard_command == 'right arrow':
        pyautogui.keyDown('right')
        pyautogui.keyUp('right')
    elif keyboard_command == 'space':
        pyautogui.press('space')
    elif keyboard_command == 'left mouse down':
        pyautogui.mouseDown(button='left')
        pyautogui.mouseUp(button='left')
    elif keyboard_command == 'r':
        pyautogui.press('r')
    elif keyboard_command == 'tab':
        pyautogui.press('tab')
    # Add more keyboard commands as needed


def look_around(x, y):
    # Calculate the current position of the mouse
    current_x, current_y = pyautogui.position()
    
    # Calculate the new position of the mouse based on the input x and y offsets
    new_x = current_x + x
    new_y = current_y + y
    
    # Move the mouse to the new position using the pyautogui module
    pyautogui.moveTo(new_x, new_y)
