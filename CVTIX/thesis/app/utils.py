import cv2
import os

def save_snapshot(frame, bbox, track_id, output_dir="snapshots"):
    """
    Save a cropped image of the detected vehicle.
    - frame: full video frame (numpy array)
    - bbox: (x1, y1, x2, y2)
    - track_id: DeepSORT ID
    - output_dir: folder to store snapshots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    file_path = os.path.join(output_dir, f"vehicle_{track_id}.jpg")
    cv2.imwrite(file_path, crop)
    return file_path
