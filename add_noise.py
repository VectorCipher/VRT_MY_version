import cv2
import numpy as np
import os

input_video = r"D:\Projects\OpenGait\sample_walking.mp4"
output_dir = r"D:\Projects\VRT\testsets\sample_walking_noisy\000"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(input_video)
sigma = 20
frame_idx = 0

print(f"Reading {input_video}...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Generate Gaussian noise
    noise = np.random.normal(0, sigma, frame.shape)
    
    # Add noise and clip
    noisy_frame = frame + noise
    noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
    
    # Save frame
    out_path = os.path.join(output_dir, f"{frame_idx:05d}.png")
    cv2.imwrite(out_path, noisy_frame)
    frame_idx += 1

cap.release()
print(f"Saved {frame_idx} noisy frames to {output_dir}")
