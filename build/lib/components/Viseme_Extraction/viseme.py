import cv2
import numpy as np

def extract_frames(video_file, timestamp_ranges):
    # Load the video
    video = cv2.VideoCapture(video_file)

    # Get the video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = video.get(cv2.CAP_PROP_FPS)

    # Initialize an empty list to store the extracted frames
    frames = []

    # Loop through the timestamp ranges
    for start_stamp, end_stamp in timestamp_ranges:
        # Calculate the frame numbers for the given start and end timestamps
        start_frame = int(start_stamp * frame_rate)
        end_frame = int(end_stamp * frame_rate)

        # Check if the frame numbers are within the range of the video
        if end_frame >= total_frames:
            print("Error: Frame number out of range")
            continue

        # Calculate the frame interval between each extracted frame
        num_frames = end_frame - start_frame + 1
        frame_interval = max(num_frames // 6, 1)

        # Loop through the range of frame numbers and extract frames at intervals of the frame interval
        for i in range(start_frame, end_frame + 1, frame_interval):
            # Set the video position to the desired frame
            video.set(cv2.CAP_PROP_POS_FRAMES, i)

            # Read the frame
            _, frame = video.read()

            # Add the frame to the list
            frames.append(frame)

    # Release the video
    video.release()

    # Return the list of extracted frames
    return frames

