# import inference
import os, time, json
import verify_video, modify_video
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

@app.route('/upload_video_hardcode_sse', methods=['POST'])
def upload_video_hardcode_sse():
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
    
    return Response(process_video_stream(video_path), content_type='text/event-stream')

def process_video_stream(video_path):
    print("Checking whether this is a surf video...")
    result = {
        "status": "interim",
        "message": "Checking if video is a surf video..."
    }
    yield f"data: {json.dumps(result)}\n\n"
    is_surf = verify_video.is_surf_video(video_path)

    if isinstance(is_surf, dict) and "error" in is_surf:
        print("Couldn't check if surf video, exiting...")
        result = {
            "status": "error",
            "message": is_surf["error"]
        }
        yield f"data: {json.dumps(result)}\n\n"
    elif is_surf:
        result = {
            "status": "interim",
            "message": "Analyzing ride..."
        }
        yield f"data: {json.dumps(result)}\n\n"
        time.sleep(5)
        s3_bucket_name = "wavescorevideos"
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

        result = {
            "status": "interim",
            "message": "Annotating video with analysis..."
        }
        yield f"data: {json.dumps(result)}\n\n"
        annotated_url = modify_video.annotate_video(video_path, s3_bucket_name, analysis)

        # Return the annotated video to the client
        result = {
            "status": "success",
            "message": "Nice surfing!",
            "video_url": annotated_url
        }
        yield f"data: {json.dumps(result)}\n\n"
    else:
        result = {
            "status": "error",
            "message": "Video does not seem to be a surf video. Please try another video."
        }
        yield f"data: {json.dumps(result)}\n\n"

    # Delete the temporary file after analyzing the video & returning to client
    if os.path.exists(video_path):
        os.remove(video_path)

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
    is_surf = verify_video.is_surf_video(video_path)
    
    if is_surf:
        s3_bucket_name = "wavescorevideos"
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
        annotated_url = modify_video.annotate_video(video_path, s3_bucket_name, analysis)

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
    is_surf = verify_video.is_surf_video(video_path)
    
    if is_surf:
        # model_url = "https://wavescorevideos.s3.us-east-1.amazonaws.com/surf_maneuver_model_20241106_1324.pth"
        s3_bucket_name = "wavescorevideos"
        model_filename = "surf_maneuver_model_20241106_1324.pth"

        # Run inference
        # maneuvers = inference.run_inference(video_path, s3_bucket_name, model_filename)
        maneuvers = 'TODO'

        # Annotate the user video with inference results
        analysis = {'maneuvers': maneuvers, 'score': 8.5}
        annotated_url = modify_video.annotate_video(video_path, s3_bucket_name, analysis)

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