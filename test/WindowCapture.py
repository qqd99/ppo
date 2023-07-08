import win32gui
import win32ui
import win32con
import win32api
from PIL import Image
import time
import numpy as np


class WindowCapture:
    def __init__(self, window_name, frame_size=(84, 84), fps=None, debug=False):
        self.window_name = window_name
        self.frame_size = frame_size
        self.fps = fps
        self.debug = debug
        self.frame_num = 0
        self.frames = np.zeros((frame_size[1], frame_size[0], 4), dtype=np.uint8)
        self.hwnd = win32gui.FindWindow(None, window_name)

    def capture_screen(self):
        # Get the dimensions of the game window
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        width = right - left
        height = bottom - top
        # Get the device context of the game window
        hdc = win32gui.GetWindowDC(self.hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hdc)
        saveDC = mfcDC.CreateCompatibleDC()
        # Create a bitmap object to hold the screen capture
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        # Select the bitmap object into the devicecontext
        saveDC.SelectObject(saveBitMap)
        # Capture the screen at the specified frame rate
        if self.fps is not None:
            interval = 1 / self.fps
            start_time = time.time()
            frame_count = 0
            while True:
                # Copy the screen into the bitmap object
                result = win32gui.FindWindow(self.hwnd, saveDC.GetSafeHdc())
                # Convert the bitmap object to a PIL image
                bmpinfo = saveBitMap.GetInfo()
                bmpstr = saveBitMap.GetBitmapBits(True)
                img = Image.frombuffer(
                    'RGB',
                    (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                    bmpstr,
                    'raw',
                    'BGRX',
                    0,
                    1
                )
                # Process the frame and return the pre-processed image
                processed_frame = self.process_frame(np.array(img))
                # Wait for the next frame
                frame_count += 1
                elapsed_time = time.time() - start_time
                wait_time = interval * frame_count - elapsed_time
                if wait_time > 0:
                    time.sleep(wait_time)
                else:
                    return processed_frame
        # Capture the screen as fast as possible
        else:
            # Copy the screen into the bitmap object
            result = win32gui.FindWindow(self.hwnd, saveDC.GetSafeHdc())
            # Convert the bitmap object to a PIL image
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            img = Image.frombuffer(
                'RGB',
                (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                bmpstr,
                'raw',
                'BGRX',
                0,
                1
            )
            # Process the frame and return the pre-processed image
            processed_frame = self.process_frame(np.array(img))
            # Clean up
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hdc)
            return processed_frame

    def process_frame(self, frame):
        # Convert the image to gray
        state_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize the image to the desired frame size
        state_resized = cv2.resize(state_gray, self.frame_size)
        # Crop the image to the desired aspect ratio
        gray_final = state_resized[16:100, :]
        # Add the pre-processed image to the frame buffer
        if self.frame_num == 0:
            self.frames[:, :, 0] = gray_final
            self.frames[:, :, 1] = gray_final
            self.frames[:, :, 2] = gray_final
            self.frames[:, :,3] = gray_final
        else:
            self.frames[:, :, 3] = self.frames[:, :, 2]
            self.frames[:, :, 2] = self.frames[:, :, 1]
            self.frames[:, :, 1] = self.frames[:, :, 0]
            self.frames[:, :, 0] = gray_final
        # Increment the frame counter
        self.frame_num += 1
        # Show the pre-processed image if the debug flag is set
        if self.debug:
            cv2.imshow('Game', gray_final)
        # Return the pre-processed image as a copy of the frame buffer
        return self.frames.copy()

__all__ = ['WindowCapture']
