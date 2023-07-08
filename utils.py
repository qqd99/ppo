import numpy as np
import cv2
import tensorflow as tf

#Step 1: Initialize the agent and collect initial trajectories
# Function to capture game screenshots and preprocess them
def capture_screen():
    # Code to capture and preprocess the game screen as an image
    screen = capture_game_screen()
    preprocessed_screen = preprocess_screen(screen)
    return preprocessed_screen

# Function to simulate pressing a key based on the agent's action
def press_key(key):
    # Code to simulate pressing the specified key in the game
    simulate_key_press(key)

# Function to collect trajectories by running the agent in the game
def collect_trajectories(agent):
    trajectories = []
    observation = capture_screen()

    while True:
        # Run the agent's policy to choose an action
        action = agent.get_action(observation)

        # Simulate pressing the chosen key in the game
        press_key(action)

        # Capture the next observation after the action
        next_observation = capture_screen()

        # Add the transition to the trajectory
        trajectory = {'observation': observation, 'action': action, 'next_observation': next_observation}
        trajectories.append(trajectory)

        # Check if the stop condition is reached
        if action == '/':
            break

        # Update the current observation
        observation = next_observation

    return trajectories


#Step 2: Collect human preferences
def segment_trajectories(trajectories, segment_length=100):
    trajectory_pairs = []
    
    for trajectory in trajectories:
        # Check if the trajectory is long enough for segmentation
        if len(trajectory['observations']) >= segment_length:
            num_segments = len(trajectory['observations']) // segment_length
            
            # Segment the trajectory into pairs of trajectory segments
            for i in range(num_segments - 1):
                start_index = i * segment_length
                end_index = (i + 1) * segment_length
                
                segment_1 = {
                    'observations': trajectory['observations'][start_index:end_index],
                    'actions': trajectory['actions'][start_index:end_index],
                    'rewards': trajectory['rewards'][start_index:end_index]
                }
                
                segment_2 = {
                    'observations': trajectory['observations'][end_index:end_index + segment_length],
                    'actions': trajectory['actions'][end_index:end_index + segment_length],
                    'rewards': trajectory['rewards'][end_index:end_index + segment_length]
                }
                
                trajectory_pairs.append((segment_1, segment_2))
    
    return trajectory_pairs

def get_human_preference():
    while True:
        print("Please compare the two trajectory segments:")
        print("1. left")
        print("2. right")
        print("3. tie")
        print("4. abstain")
        choice = input("Enter your preference (1/2/3/4): ")
        
        if choice in ['left', 'right', 'tie', 'abstain']:
            break
        else:
            print("Invalid choice. Please try again.")
    return choice

def display_segments(segment_1, segment_2):
    # Extract the observations from the segments
    observations_1 = segment_1['observations']
    observations_2 = segment_2['observations']

    # Create a window to display the segments side by side
    window_name = 'Trajectory Segments'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 400)

    # Display the segments side by side
    for obs_1, obs_2 in zip(observations_1, observations_2):
        # Resize the observations to a common size if necessary
        # Resize obs_1 and obs_2 accordingly

        # Concatenate the two observations horizontally
        concatenated = cv2.hconcat([obs_1, obs_2])

        # Display the concatenated image in the window
        cv2.imshow(window_name, concatenated)
        cv2.waitKey(100)  # Adjust the delay between frames if necessary

    # Wait for the user to close the window
    cv2.waitKey(0)

    # Close the window
    cv2.destroyAllWindows()

# Function to ask human evaluators for preferences
def collect_human_preferences(trajectory_pairs):
    human_preferences = []

    for pair in trajectory_pairs:
        segment1 = pair[0]
        segment2 = pair[1]

        # Show the segments to the human evaluator as movie clips
        show_movie_clips(segment1, segment2)

        # Collect the preference from the human evaluator
        preference = get_human_preference()

        # If preference is None, skip this comparison
        if preference is None:
            continue

        # Add the preference to the human preferences list
        human_preferences.append((segment1, segment2, preference))

    return human_preferences

# Step 3: Train the preference predictor
def create_preference_predictor(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def train_preference_predictor(preference_predictors, human_preferences):
    
    for i, predictor in enumerate(preference_predictors):
        # Set a random seed for reproducibility
        random_seed = random.randint(0, 9999)
        tf.random.set_seed(random_seed)

        # Prepare the training data
        observations_1 = []
        observations_2 = []
        preferences = []

        # Shuffle the human_preferences dataset randomly
        shuffled_preferences = random.sample(human_preferences, len(human_preferences))

        # Select a random subset from the shuffled dataset
        subset_size = 100  # Adjust the subset size as desired
        selected_preferences = shuffled_preferences[:subset_size]

        # Extract the observations and preferences from the selected subset
        for preference in selected_preferences:
            segment_1, segment_2, pref = preference
            observations_1.append(segment_1['observations'])
            observations_2.append(segment_2['observations'])
            preferences.append(pref)

        observations_1 = tf.concat(observations_1, axis=0)
        observations_2 = tf.concat(observations_2, axis=0)
        preferences = tf.convert_to_tensor(preferences, dtype=tf.float32)
        
        # Train the predictor using the dataset
        model.fit([observations_1, observations_2], preferences, epochs=10, batch_size=32)
        # Use the same dataset for training with different random seeds
        
        print(f"Preference predictor {i+1}/{len(preference_predictors)} trained

#Step 4: Generate new queries


def generate_new_queries(preference_predictor, trajectory_pairs):
    evi_values = []

    # Compute uncertainty for each pair of trajectory segments
    uncertainties = compute_uncertainties(preference_predictor, trajectory_pairs)

    for pair in trajectory_pairs:
        segment_1, segment_2 = pair

        # Calculate expected prediction before querying
        expected_prediction_before = preference_predictor[0].predict([segment_1, segment_2])

        # Simulate preference query and retrain the preference predictor
        simulated_preference = preference_predictor[1].predict([segment_1, segment_2])
        simulated_preferences = [(segment_1, segment_2, simulated_preference)]
        retrained_predictor = train_preference_predictor(simulated_preferences)

        # Calculate expected prediction after querying
        expected_prediction_after = retrained_predictor.predict([segment_1, segment_2])

        # Calculate information gain
        information_gain = expected_prediction_after - expected_prediction_before

        # Compute EVI by combining uncertainty and information gain
        evi = compute_evi(uncertainties[pair], information_gain)

        evi_values.append(evi)

    # Sort trajectory pairs based on EVI values in descending order
    sorted_pairs = [pair for _, pair in sorted(zip(evi_values, trajectory_pairs), reverse=True)]

    return sorted_pairs


#Step 5: Train the agent using preference-based loss
