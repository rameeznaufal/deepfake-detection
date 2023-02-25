import numpy as np

def calculate_mismatch_error_rate(frame_values, MOUTH_AR_THRESH):
    # Array of frame classification values | 1: open, 0: closed
    frame_values[frame_values >= MOUTH_AR_THRESH] = 1
    frame_values[frame_values < MOUTH_AR_THRESH] = 0

    # Create an empty 1D NumPy array with the same number of rows as frame_values
    classification = np.ones(frame_values.shape[0], dtype=int)

    # Iterate over the rows of the array
    for i in range(frame_values.shape[0]):
        # Check if the row contains any 0s
        if 0 in frame_values[i]:
            classification[i] = 0

    # Calculate the percentage of 1's in the array
    num_zeros = np.sum(classification == 1)
    mismatch = (num_zeros / len(classification)) * 100

    return mismatch
