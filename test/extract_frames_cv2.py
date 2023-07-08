import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import os

def extract_frames(video_path, save_dir, frame_interval, output_size=(480, 480), duration_percentage=0.5):
    # Create a directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    frame_count = 0
    total_frames = 0

    # Get the total number of frames and calculate the total frames to extract
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_to_extract = int(total_frames * duration_percentage)

    # Iterate over frames and save selected frames as images
    while True:
        # Read the next frame
        ret, frame = video.read()

        # Break if no frame is returned
        if not ret:
            break

        print(frame_count)

        # Check if the current frame should be saved
        if frame_count % frame_interval == 0 :
            # Resize the frame
            resized_frame = cv2.resize(frame, output_size)

            # Save the frame as an image
            frame_path = os.path.join(save_dir, f"frame_{frame_count}.png")
            cv2.imwrite(frame_path, resized_frame)

        frame_count += 1

        # Calculate the progress percentage
        progress = (frame_count / total_frames_to_extract) * 100

        # Clear the previous line and print the progress
        print(f"Progress: {progress:.2f}%", end="\r")


    # Release the video file
    video.release()

    # Print a new line after progress
    print()

# Example usage
video_path = "video2.mp4"
frame_save_dir = "curl_training_data"
frame_interval = 30  # Save one frame every 30 frames
output_size = (128, 128)  # Output image size
duration_percentage = 0.5  # Use 50% of the video duration

extract_frames(video_path, frame_save_dir, frame_interval, output_size, duration_percentage)