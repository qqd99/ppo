import cv2
import os

def extract_frames(video_path, save_dir, frame_interval):
    # Create a directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    frame_count = 0
    saved_frame_count = 0
    
    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Iterate over frames and save selected frames as images
    while True:
        # Read the next frame
        ret, frame = video.read()
        
        # Break if no frame is returned
        if not ret:
            break
        
        # Check if the current frame should be saved
        if frame_count % frame_interval == 0:
            # Save the frame as an image
            frame_path = os.path.join(save_dir, f"frame_{saved_frame_count}.png")
            cv2.imwrite(frame_path, frame)
            
            saved_frame_count += 1
        
        frame_count += 1
        
        # Calculate and display the progress
        progress = (frame_count / total_frames) * 100
        print(f"Progress: {progress:.2f}%")
    
    # Release the video file
    video.release()

# Example usage
video_path = "video2.mp4"
frame_save_dir = "curl_training_data"
frame_interval = 30  # Save one frame every 30 frames

extract_frames(video_path, frame_save_dir, frame_interval)
