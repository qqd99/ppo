import cv2

def get_health_reward(prev_health, image_path):
    # Load the game screenshot as a grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Extract the region of the image containing the health bar
    health_bar = image[10:20, 10:200]
    
    # Calculate the current health value based on the brightness of the health bar
    current_health = health_bar.mean()
    
    # Calculate the reward based on the change in health since the previous time step
    reward = current_health - prev_health
    
    # Update the previous health value for the next time step
    prev_health = current_health
    
    # Return the reward and the updated previous health value
    return reward, prev_health

def reward_function(observation, action):
    
    base_reward = 0.1
    reward = base_reward# Define a base reward for taking the "look" action
    
    # Penalize the agent for unnecessary "look" actions
    if action == "look":
        if observation["last_action"] == "look":
            reward -= 0.1
    
    # Reward the agent for looking at specific objects
    if observation["object_of_interest"] in observation["current_view"]:
        reward += 0.5
    
    # Penalize the agent for looking away from important objects
    if observation["object_of_interest"] not in observation["current_view"]:
        reward -= 0.2
    
    return reward
