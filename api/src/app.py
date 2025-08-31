import ctypes, gc, json, os
import maneuver_inference, modify_video, score_inference, verify_video
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://www.wavescore.xyz", "https://*.vercel.app"])

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
    try:
        # Probe media info
        try:
            info = modify_video.get_media_info(video_path)
            print(
                f"Upload info: size={info['size_bytes']/1_000_000:.2f} MB, "
                f"resolution={info['width']}x{info['height']}, fps={info['fps']:.2f}, "
                f"duration={info['duration']:.2f}s, est_avg_bitrate={info['est_avg_bitrate_mbps']:.2f} Mbps"
            )
        except Exception as e:
            print(f"Failed to read media info: {e}")
            info = {'width': 0, 'height': 0, 'fps': 0.0, 'duration': 0.0, 'est_avg_bitrate_mbps': 0.0, 'size_bytes': 0}

        # Decide whether to normalize
        needs_norm = (
            info['width'] > 1280 or info['height'] > 720 or (info['fps'] and info['fps'] > 30.0) or info['size_bytes'] > 100 * 1024 * 1024 or info['est_avg_bitrate_mbps'] > 10.0
        )
        if needs_norm:
            # Notify client
            result = {
                "status": "interim",
                "message": "Compressing video..."
            }
            yield f"data: {json.dumps(result)}\n\n"

            # Normalize to 640x360 @ 30fps
            try:
                norm_path = modify_video.normalize_video(video_path, target_width=640, target_fps=30)
                # Log post-normalization info
                try:
                    nfo = modify_video.get_media_info(norm_path)
                    print(
                        f"Normalized info: size={nfo['size_bytes']/1_000_000:.2f} MB, "
                        f"resolution={nfo['width']}x{nfo['height']}, fps={nfo['fps']:.2f}, "
                        f"duration={nfo['duration']:.2f}s, est_avg_bitrate={nfo['est_avg_bitrate_mbps']:.2f} Mbps"
                    )
                except Exception as e:
                    print(f"Failed to read normalized media info: {e}")
                video_path = norm_path
            except Exception as e:
                print(f"Normalization failed: {e}")
                # Proceed with original

        print("Checking whether this is a surf video...")
        result = {
            "status": "interim",
            "message": "Checking for surf content..."
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

        print(f"Surf verification result: {is_surf}")
        if is_surf:
            result = {
                "status": "interim",
                "message": "Identifying maneuvers..."
            }
            yield f"data: {json.dumps(result)}\n\n"

            # Run inference
            s3_bucket_name = "wavescorevideos"
            # maneuver_model_url = "https://wavescorevideos.s3.us-east-1.amazonaws.com/maneuver_model_20250518_2118.pth"
            maneuver_model = "maneuver_model_20250518_2118.pth"
            print(f"Starting maneuver inference with model: {maneuver_model}")
            try:
                maneuvers, _, _ = maneuver_inference.run_inference(video_path, maneuver_model, mode='prod')
            except Exception as e:
                print(f"Error during maneuver inference: {str(e)}")
                result = {
                    "status": "server_error",
                    "message": "Internal server error"
                }
                yield f"data: {json.dumps(result)}\n\n"
                return
            print(f"Maneuver inference complete: {len(maneuvers)} maneuvers detected")

            result = {
                "status": "interim",
                "message": "Predicting score..."
            }
            yield f"data: {json.dumps(result)}\n\n"

            # Predict score
            score_model = "score_model_20250602_1643.pth"
            print(f"Starting score prediction with model: {score_model}")
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
            print(f"Score prediction complete: {score}")

            analysis = {'maneuvers': maneuvers, 'score': score}

            result = {
                "status": "interim",
                "message": "Processing results..."
            }
            yield f"data: {json.dumps(result)}\n\n"
            print("Starting video annotation and upload to S3...")
            annotated_url = modify_video.annotate_video(video_path, s3_bucket_name, analysis)
            print(f"Annotation complete, S3 URL: {annotated_url}")

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
    finally:
        # Delete the temporary file after processing, regardless of outcome
        try:
            print("Deleting original local video at: " + video_path)
            if os.path.exists(video_path):
                os.remove(video_path)
                print("Temp video deleted.")
            else:
                print("Temp video already absent.")
        except Exception as e:
            print(f"Error deleting temp video: {e}")

        # Encourage memory to be released back to the OS after heavy work
        try:
            gc.collect()
            print("Garbage collection completed.")
        except Exception:
            pass
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            print("malloc_trim executed.")
        except Exception:
            pass

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))