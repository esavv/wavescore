print("Importing packages...")
import base64, csv, cv2, os, random
from google.cloud import vision

print("Setting env variables...")
if os.path.exists("./surfjudge-443400-035fd5609c22.json"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./surfjudge-443400-035fd5609c22.json"
else:
    # Decode the Base64 key and write it to a temporary file
    base64_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_B64")
    if base64_key:
        with open("service_account_key.json", "wb") as temp_file:
            temp_file.write(base64.b64decode(base64_key))
        # Point the library to the temporary file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account_key.json"
    else:
        raise EnvironmentError("Missing GOOGLE_APPLICATION_CREDENTIALS_B64 environment variable")

def is_surf_video(video_path):
    # Step 1: Extract random frames
    print("Extracting random frames from the video...")
    frames = extract_random_frames(video_path, num_frames=5)

    keyword_path = './cloud_vision_keywords.csv'
    file = csv.reader(open(keyword_path, 'r'))
    keywords = set([row[0] for row in file])

    result = False
    # Step 2: Analyze each extracted frame using Cloud Vision API
    for frame in frames:
        print(f"Analyzing {frame}...")
        frame_labels = analyze_image(frame)
        for label in frame_labels:
            # Collect label description and its score
            if label.score >= 0.8:
                if label.description in keywords:
                    print("Found relevant label: " + label.description)
                    result = True

        # Delete the frame after analysis
        os.remove(frame)

    return result

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