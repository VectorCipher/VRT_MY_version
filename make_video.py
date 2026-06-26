import cv2
import os
import glob

input_dir = r"D:\Projects\VRT\testsets\sample_walking_noisy\000"
output_video = r"D:\Projects\OpenGait\sample_walking_noisy.mp4"

# Get all png files in input_dir
frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))

if not frames:
    print(f"No frames found in {input_dir}")
    exit(1)

# Get frame dimensions from the first frame
first_frame = cv2.imread(frames[0])
height, width, layers = first_frame.shape

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height)) 

print(f"Writing video to {output_video}...")
for frame_path in frames:
    frame = cv2.imread(frame_path)
    out.write(frame)

out.release()
print(f"Successfully created {output_video}")
