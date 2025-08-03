import csv, cv2, os, random
from google.cloud import vision
from google.api_core.exceptions import GoogleAPICallError, PermissionDenied

google_key_path = "../keys/google_cloud_account_key.json"
vision_keyword_path = '../apidata/google_cloud_vision_keywords.csv'

print("verify_video: Setting env variables...")
if os.path.exists(google_key_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_key_path
#else:
    #TODO: raise an appropriate error
    #raise EnvironmentError("Missing GOOGLE_APPLICATION_CREDENTIALS_B64 environment variable")

def is_surf_video(video_path):
    try:
        # Step 1: Extract random frames
        print("Extracting random frames from the video...")
        frames = extract_random_frames(video_path, num_frames=10)

        file = csv.reader(open(vision_keyword_path, 'r'))
        keywords = set([row[0] for row in file])

        result = False
        # Step 2: Analyze each extracted frame using Cloud Vision API
        # Policy: Return True as soon as we detect surf content in a single frame
        for frame in frames:
            if result != True:
                print(f"  Analyzing {frame}...")
                frame_labels = analyze_image(frame)
                for label in frame_labels:
                    # Collect label description and its score
                    if label.description in keywords and label.score >= 0.8:
                        print("    Found relevant label: " + label.description)
                        result = True
                        break

            # Delete the frame after analysis
            os.remove(frame)

        return result
    except Exception as e:
        print(f"Error verifying surf video: {str(e)}")
        # Clean up any remaining frames
        if 'frames' in locals():
            for frame in frames:
                if os.path.exists(frame):
                    os.remove(frame)
        raise

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
    try:
        response = client.label_detection(image=image)
        return response.label_annotations
    except (PermissionDenied, GoogleAPICallError, Exception) as e:
        print(f"Error analyzing image: {str(e)}")
        raise