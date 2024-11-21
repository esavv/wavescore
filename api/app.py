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

    # Dummy response with hardcoded results
    result = "3 maneuvers performed:\n- Cutback (0:03)\n- Cutback (0:09)\n- Snap (0:15)\n"
    
    # Return hardcoded response
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
