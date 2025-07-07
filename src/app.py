import os, time, json
import verify_video, modify_video, inference, score_inference
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://www.wavescore.xyz/", "https://*.vercel.app/"])

@app.route('/upload_video_sse', methods=['POST'])
def upload_video_sse():
    # Check if the video file is part of the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    print('Received video file: ' + file.filename)
    # Save the file temporarily
    video_path = f"/tmp/{file.filename}"  # or choose a path that works for you
    print("Saving original video to: " + video_path)
    file.save(video_path)
    
    return Response(process_video_stream(video_path), content_type='text/event-stream')

def process_video_stream(video_path):
    print("Checking whether this is a surf video...")
    result = {
        "status": "interim",
        "message": "Checking if video is a surf video..."
    }
    yield f"data: {json.dumps(result)}\n\n"
    
    try:
        is_surf = verify_video.is_surf_video(video_path)
    except Exception as e:
        print(f"Error during video verification: {str(e)}")
        result = {
            "status": "server_error",
            "message": "Internal server error"
        }
        yield f"data: {json.dumps(result)}\n\n"
        return

    if is_surf:
        result = {
            "status": "interim",
            "message": "Analyzing ride..."
        }
        yield f"data: {json.dumps(result)}\n\n"
        time.sleep(2)
     
        result = {
            "status": "interim",
            "message": "Identifying maneuvers..."
        }
        yield f"data: {json.dumps(result)}\n\n"

        # Run inference
        s3_bucket_name = "wavescorevideos"
        # maneuver_model_url = "https://wavescorevideos.s3.us-east-1.amazonaws.com/surf_maneuver_model_20250518_2118.pth"
        maneuver_model = "surf_maneuver_model_20250518_2118.pth"
        try:
            maneuvers, _, _ = inference.run_inference(video_path, maneuver_model, mode='prod')
        except Exception as e:
            print(f"Error during maneuver inference: {str(e)}")
            result = {
                "status": "server_error",
                "message": "Internal server error"
            }
            yield f"data: {json.dumps(result)}\n\n"
            return

        result = {
            "status": "interim",
            "message": "Predicting score..."
        }
        yield f"data: {json.dumps(result)}\n\n"

        # Predict score
        score_model = "score_model_20250602_1643.pth"
        # score = 8.5
        try:
            scores = score_inference.run_inference([video_path], score_model, mode='prod')
            score = scores[0]
        except Exception as e:
            print(f"Error during score inference: {str(e)}")
            result = {
                "status": "server_error",
                "message": "Internal server error"
            }
            yield f"data: {json.dumps(result)}\n\n"
            return

        analysis = {'maneuvers': maneuvers, 'score': score}

        result = {
            "status": "interim",
            "message": "Annotating video with analysis..."
        }
        yield f"data: {json.dumps(result)}\n\n"
        annotated_url = modify_video.annotate_video(video_path, s3_bucket_name, analysis)

        # Return the annotated video to the client
        result = {
            "status": "success",
            "message": "Analysis complete",
            "analysis": analysis,
            "video_url": annotated_url
        }
        yield f"data: {json.dumps(result)}\n\n"
    else:
        print("Not a surf video! Exiting...")
        result = {
            "status": "user_error",
            "message": "Video does not seem to be a surf video. Please try another video."
        }
        yield f"data: {json.dumps(result)}\n\n"

    # Delete the temporary file after analyzing the video & returning to client
    print("Deleting original local video at: " + video_path)
    if os.path.exists(video_path):
        os.remove(video_path)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))