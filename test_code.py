

#--------------------------sample_segment_from_path--------
def _slice_path(path, segment_length, start_pos=0):
    """
    Find in the trajectory and take out the 'obs', "actions", 'original_rewards'
    """
    return {
        k: np.asarray(v[start_pos:(start_pos + segment_length)])
        for k, v in path.items()
        if k in ['obs', "actions", 'original_rewards']}

def create_segment_q_states(segment):
    obs_Ds = segment["obs"]
    act_Ds = segment["actions"]
    return np.concatenate([obs_Ds, act_Ds], axis=1)

def sample_segment_from_path(path, segment_length):
    """Returns a segment sampled from a random place in a path.
    Returns a segment 'obs', "actions", 'original_rewards',"q_states"
    Returns None if the path is too short
"""
    path_length = len(path["obs"])
    if path_length < segment_length:
        return None

    start_pos = np.random.randint(0, path_length - segment_length + 1)

    # Build segment
    segment = _slice_path(path, segment_length, start_pos)

    # Add q_states
    segment["q_states"] = create_segment_q_states(segment)
    return segment




#-----------------------------ComparisonCollector----------

class SyntheticComparisonCollector(object):
    def __init__(self):
        self._comparisons = []

    def add_segment_pair(self, left_seg, right_seg):
        """Add a new unlabeled comparison from a segment pair"""
        comparison = {
            "left": left_seg,
            "right": right_seg,
            "label": None
        }
        self._comparisons.append(comparison)

    def __len__(self):
        return len(self._comparisons)

    @property
    def labeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is not None]

    @property
    def labeled_decisive_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] in [0, 1]]

    @property
    def unlabeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is None]

    def label_unlabeled_comparisons(self):
        for comp in self.unlabeled_comparisons:
            self._add_synthetic_label(comp)

    @staticmethod
    def _add_synthetic_label(comparison):
        left_seg = comparison['left']
        right_seg = comparison['right']
        left_has_more_rew = np.sum(left_seg["original_rewards"]) > np.sum(right_seg["original_rewards"])

        # Mutate the comparison and give it the new label
        comparison['label'] = 0 if left_has_more_rew else 1


class HumanComparisonCollector():
    def __init__(self, env_id, experiment_name):


        self._comparisons = []

        self._upload_workers = multiprocessing.Pool(4)



    def add_segment_pair(self, left_seg, right_seg):
        """Add a new unlabeled comparison from a segment pair"""
            comparison = {
            "left": left_seg,
            "right": right_seg,
            "label": None
        }

        self._comparisons.append(comparison)

    def __len__(self):
        return len(self._comparisons)

    @property
    def labeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is not None]

    @property
    def labeled_decisive_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] in [0, 1]]

    @property
    def unlabeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is None]

    def label_unlabeled_comparisons(self):
        for comparison in self.unlabeled_comparisons:
            display_segments(comparison["left"], comparison["right"])
            db_comp = get_human_preference()
            if db_comp.response == 'left':
                comparison['label'] = 0
            elif db_comp.response == 'right':
                comparison['label'] = 1
            elif db_comp.response == 'tie' or db_comp.response == 'abstain':
                comparison['label'] = 'equal'
                # If we did not match, then there is no response yet, so we just wait

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
#-----------------------------ComparisonCollector----------
class FullyConnectedMLP(object):
    """Vanilla two hidden layer multi-layer perceptron"""

    def __init__(self, obs_shape, act_shape, h_size=64):
        super(FullyConnectedMLP, self).__init__()
        input_dim = np.prod(obs_shape) + np.prod(act_shape)

        self.model = keras.Sequential([
            layers.Dense(h_size, input_dim=input_dim),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(h_size),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(1)
        ])

    def run(self, obs, act):
        flat_obs = tf.keras.layers.Flatten()(obs)
        x = tf.concat([flat_obs, act], axis=1)
        return self.model(x)

def _predict_rewards(self, obs_segments, act_segments, network):
    """
    :param obs_segments: tensor with shape = (batch_size, segment_length) + obs_shape
    :param act_segments: tensor with shape = (batch_size, segment_length) + act_shape
    :param network: neural net with .run() that maps obs and act tensors into a (scalar) value tensor
    :return: tensor with shape = (batch_size, segment_length)
    """
    batchsize = tf.shape(obs_segments)[0]
    segment_length = tf.shape(obs_segments)[1]

    # Temporarily chop up segments into individual observations and actions
    obs = tf.reshape(obs_segments, (-1,) + self.obs_shape)
    acts = tf.reshape(act_segments, (-1,) + self.act_shape)

    # Run them through our neural network
    rewards = network.run(obs, acts)

    # Group the rewards back into their segments
    return tf.reshape(rewards, (batchsize, segment_length))
tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
