import os
import video_content, video_overlay, inference
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
    video_path = f"/tmp/{file.filename}"  # or choose a path that works for you
    file.save(video_path)

    print("Checking whether this is a surf video...")
    is_surf = video_content.is_surf_video(video_path)
    
    if is_surf:
        model_path = "models/surf_maneuver_model_20241106_1324.pth"
        maneuvers = inference.run_inference(video_path, model_path)
        analysis = {'maneuvers': maneuvers, 'score': 8.5}
        annotated_url = video_overlay.annotate_video(video_path, analysis)

        # Return the annotated video to the client
        result = {
            "status": "success",
            "message": "Nice surfing!",
            "video_url": annotated_url
        }
    else:
        result = {
            "status": "error",
            "message": "Video does not seem to be a surf video. Please try another video."
        }
    # Delete the temporary file after extracting metadata
    os.remove(video_path)
    
    # Return hardcoded response
    return jsonify(result)

@app.route('/upload_video_hardcode', methods=['POST'])
def upload_video_hardcode():
    # Check if the video file is part of the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    print('Received file: ' + file.filename)

    # Save the file temporarily
    video_path = f"/tmp/{file.filename}"  # or choose a path that works for you
    file.save(video_path)

    print("Checking whether this is a surf video...")
    is_surf = video_content.is_surf_video(video_path)
    
    if is_surf:
        maneuvers = [
            {'name': '360', 'start_time': 3.0, 'end_time': 5.0},
            {'name': 'Snap', 'start_time': 6.0, 'end_time': 8.0},
            {'name': 'Snap', 'start_time': 10.0, 'end_time': 11.0},
            {'name': 'Cutback', 'start_time': 14.0, 'end_time': 15.0},
            {'name': 'Cutback', 'start_time': 17.0, 'end_time': 18.0},
            {'name': 'Cutback', 'start_time': 20.0, 'end_time': 21.0},
            {'name': 'Cutback', 'start_time': 23.0, 'end_time': 24.0}
        ]
        analysis = {'maneuvers': maneuvers, 'score': 8.5}
        annotated_url = video_overlay.annotate_video(video_path, analysis)

        # Return the annotated video to the client
        result = {
            "status": "success",
            "message": "Nice surfing!",
            "video_url": annotated_url
        }
    else:
        result = {
            "status": "error",
            "message": "Video does not seem to be a surf video. Please try another video."
        }
    # Delete the temporary file after extracting metadata
    os.remove(video_path)
    
    # Return hardcoded response
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))