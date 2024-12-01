print("Importing packages...")
import cv2, os, random
from google.cloud import vision

print("Setting env variables...")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./surfjudge-443400-035fd5609c22.json"

# Function to extract 5 random frames from a video
def extract_random_frames(video_path, num_frames=5):
    frames = []
    video_capture = cv2.VideoCapture(video_path)
    
    # Get total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = random.sample(range(total_frames), num_frames)  # Randomly select frame indices

    # Extract frames at the random indices
    for frame_idx in frame_indices:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Set the video capture to the frame index
        ret, frame = video_capture.read()
        if ret:
            frame_filename = f"frame_{frame_idx}.jpg"
            cv2.imwrite(frame_filename, frame)  # Save the frame as an image file
            frames.append(frame_filename)

    video_capture.release()
    return frames

# Function to analyze an image with Google Cloud Vision API
def analyze_image(image_path):
    client = vision.ImageAnnotatorClient()

    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    # Perform label detection
    response = client.label_detection(image=image)
    labels = response.label_annotations
    
    return labels

# Replace with the path to your video file
video_path1 = "../tmp/IMG_1546.MOV"
video_path2 = "../tmp/IMG_1548.MOV"
video_path3 = "../tmp/pb020235.MOV"
video_path4 = "../tmp/pb020236.MOV"

# Step 1: Extract random frames
print("Extracting random frames from the video...")
frames = extract_random_frames(video_path4, num_frames=5)

# List to store all labels from all frames
all_labels = []

# Step 2: Analyze each extracted frame using Cloud Vision API
for frame in frames:
    print(f"Analyzing {frame}...")
    labels = analyze_image(frame)
    for label in labels:
        # Collect label description and its score
        all_labels.append((label.description, label.score))

    # Delete the frame after analysis
    os.remove(frame)
    
# Sort the labels by score in descending order
sorted_labels = sorted(all_labels, key=lambda x: x[1], reverse=True)

# Print all the labels and their scores in descending order
print("Collected labels from all frames (sorted by confidence score):")
for label, score in sorted_labels:
    print(f" - {label} (confidence: {score:.3f})")