import cv2
import numpy as np

def extract_frames(video_path, timestamp_ranges):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get the frame rate and total number of frames in the video
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize the list of frames
    frames = []
    
    # Loop through the timestamp ranges
    for i, timestamp_range in enumerate(timestamp_ranges):
        start_time = timestamp_range[0]
        end_time = timestamp_range[1]
        
        # Calculate the frames to extract
        frames_to_extract = np.linspace(start_time, end_time, 6)
        
        for j, frame_time in enumerate(frames_to_extract):
            # Calculate the frame number for the given time-stamp
            frame_number = int(frame_time * frame_rate)
        
            # Check if the frame number is within the range of the video
            if frame_number >= total_frames:
                print("Error: Frame number out of range")
                continue
            
            # Set the video position to the desired frame
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read the frame
            _, frame = video.read()
            
            # Append the frame to the list with timestamp_index and frame_sequencing number
            frames.append((frame,i,j))
    
    # Release the video object and return the frames list
    video.release()
    return frames



