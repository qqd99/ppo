import os
import time
import pyautogui
from PIL import ImageGrab
from datetime import datetime
import numpy as np
import cv2

class ScreenCapture:
    def __init__(self, fps, save_dir, bbox):
        self.fps = fps
        self.save_dir = save_dir
        self.bbox = bbox
    
    def capture(self):
        # Calculate the time interval between each frame
        interval = 1 / self.fps
        
        # Create the save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        while True:
            # Capture the specified portion of the screen as an Image object
            screen = ImageGrab.grab(bbox=self.bbox)
            
            # Convert the Image object to a numpy array for processing (if needed)
            screen_array = np.array(screen)
            
            # Process the image and save it to disk
            processed_screen = self.process_image(screen_array, (64, 64))
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = os.path.join(self.save_dir, f"{timestamp}.png")
            cv2.imwrite(filename, processed_screen)
            
            # Wait for the specified time interval before capturing the next frame
            time.sleep(interval)
    

    def process_screenshot(self,screenshot):
        # Resize the screenshot to a smaller size to reduce computational complexity
        resized = cv2.resize(screenshot, (640, 640), interpolation=cv2.INTER_AREA)
        
        # Convert the screenshot to grayscale to reduce the number of channels
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Convert the grayscale image to a binary image to reduce noise and simplify the image
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Reshape the binary image into a 1D array to create a state representation
        state = np.reshape(binary, (640 * 640,))
        
        return state

    def read_and_process_image(self, file_path):
        # Load the image from file
        image = cv2.imread(file_path)

        # Process the image
        processed_image = self.process_screenshot(image)

        return processed_image

 '''   
capturer = ScreenCapture(4,'img', (100, 100, 500, 500))
capturer.capture()

# Process the saved screenshot
processed_image = capturer.read_and_process_image(filename)

# Display the processed image (optional)
cv2.imshow('Processed Image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
