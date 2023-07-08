help me with generate_new_queries(preference_predictor). To be more specific, we will query preferences based on the expected value of information (EVI) of the query, aiming to choose the queries that would be most informative and valuable in improving the understanding of the reward function. To calculate the EVI, several factors are considered: Uncertainty: This relates to the variability or lack of confidence in the current predictions of the reward function. It can be measured, for example, by calculating the variance or entropy of the predictions.
Information Gain: This quantifies how much new information can be gained by querying a particular preference. It takes into account the difference between the expected prediction before querying and the expected prediction after querying. It aims to select queries that have the potential to provide significant new insights into the reward function.

Help me with compute_uncertainties, suppose we using k number of predictor models to get the variant of preference, if necessary, use tensorflow library if it more optimize. The preference has three value: 1 if it choose segment 1, 2 if it choose segment 2, 0 if it has uncertainty to choose between the two

From now on, I need use to write all code using tensorflow library. Now help me with collect_trajectories. I am not using gym and no environment. I let the PPO agent play the game by choosing the key to press follow its initial policy. The state of the game after key press is screenshot capture to make te observation. No reward specified in this state. The game is consider continuous so it is a single continuous episode. When you want the agent to take action, use press_key(key) function, key is the action that agent output provide the observation. Because the game is continuous so it only collect 1 continuous. The condition to stop is press_key('/') . After that, the trajectory will be saved into files and flush the memory for trajectories vatiable

import random
import tensorflow as tf

# Step 3: Train the preference predictor
def create_preference_predictor(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# Step 4: Train the preference predictors
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
        
        print(f"Preference predictor {i+1}/{len(preference_predictors)} trained with random seed {random_seed}")

# Usage
input_shape = (...)
num_predictors = k
preference_predictors = [create_preference_predictor(input_shape) for _ in range(num_predictors)]
train_preference_predictor(preference_predictors, human_preferences, num_epochs)
