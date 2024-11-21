import os, cv2
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    # Check if the video file is part of the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    print('Received file: ' + file.filename)

    # Save the file temporarily
    temp_path = f"tmp/{file.filename}"  # or choose a path that works for you
    file.save(temp_path)
    
    # Use OpenCV to get video duration
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        return jsonify({"error": "Failed to open video file"}), 500
    
    # Calculate the video duration
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = round(total_frames / fps, 1)
    
    # Delete the temporary file after extracting metadata
    os.remove(temp_path)

    # Dummy response with hardcoded results
    result = f"3 maneuvers performed:\n- Cutback (0:03)\n- Cutback (0:09)\n- Snap (0:15)\n\nVid length: {duration} seconds"
    
    # Return hardcoded response
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
