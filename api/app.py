from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    # Check if the video file is part of the request
    # if 'file' not in request.files:
    #     return jsonify({"error": "No file part"}), 400
    
    # file = request.files['file']
    # if file.filename == '':
    #     return jsonify({"error": "No selected file"}), 400

    # Dummy response with hardcoded results
    result = {
        "maneuvers": [
            {"name": "Cutback", "time": "0:03"},
            {"name": "Cutback", "time": "0:09"},
            {"name": "Snap", "time": "0:15"}
        ],
        "message": "Nice surfing!"
    }
    
    # Return hardcoded response
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
