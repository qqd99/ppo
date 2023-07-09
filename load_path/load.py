import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def encode_list(list_A, list_B):
    return [1 if element == actions[0] else 0 for element in unique_list]



def load_pickle_file(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

# Example usage:
file_path = "traj_0.pkl"
loaded_data = load_pickle_file(file_path)

observations = loaded_data['observations']
actions = loaded_data['actions']
unique_list = list(set(actions)) 
ecd_list= []
"""
for img in observations:
    cv2.imshow('window', np.array(img))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break"""
for i in range(len(actions)):

    ecd = encode_list(actions[i], unique_list)
    ecd_list.append(ecd)
