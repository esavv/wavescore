import boto3, csv, cv2, gc, os, subprocess
from botocore.exceptions import NoCredentialsError
from imageio_ffmpeg import get_ffmpeg_exe
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import TextClip

data_dir = '../../data'
aws_keys_path = "../keys/aws_s3_accessKeys.csv"

print("modify_video: Setting env variables...")
if os.path.exists(aws_keys_path):
    with open(aws_keys_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            os.environ['AWS_ACCESS_KEY_ID'] = row['Access key ID']
            os.environ['AWS_SECRET_ACCESS_KEY'] = row['Secret access key']
            break  # Assuming there is only one row, exit the loop after setting the variables
#else:
    #TODO: raise an appropriate error
    #raise EnvironmentError("Missing AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY environment variables")

def annotate_video(input_path, bucket_name, analysis):
    video = None
    text_clips = []
    video_url = None

    # Determine font based on platform
    font = 'Arial' if os.uname().sysname == 'Darwin' else 'DejaVuSans'

    # Write the result to a file
    input_name = os.path.splitext(os.path.basename(input_path))[0]
    input_ext = os.path.splitext(input_path)[1]
    output_name = input_name + "_annotated" + input_ext
    output_video_path = "/tmp/" + output_name

    try:
        # Ensure the main clip is closed after use
        with VideoFileClip(input_path) as clip:
            # Build overlay clips
            for maneuver in analysis['maneuvers']:
                start_time = maneuver['start_time']
                end_time = maneuver['end_time']
                text = maneuver['name']

                txt_clip = (
                    TextClip(text=text, font=font, font_size=40, color='white')
                    .with_position(('center', clip.h * 0.85))
                    .with_duration(end_time - start_time)
                    .with_start(start_time)
                )
                text_clips.append(txt_clip)

            score_text = f"Predicted Score: {analysis['score']}"
            score_clip = (
                TextClip(text=score_text, font=font, font_size=40, color='white')
                .with_position(('center', 50))
                .with_duration(2)
                .with_start(max(0, clip.duration - 2))
            )
            text_clips.append(score_clip)

            # Compose and write
            video = CompositeVideoClip([clip] + text_clips)
            print("Saving annotated video to: " + output_video_path)
            video.write_videofile(
                output_video_path,
                codec='libx264',
                remove_temp=True,
                threads=1
            )
    finally:
        # Close composite and text clips regardless of success
        try:
            if video is not None:
                video.close()
        except Exception:
            pass
        for t in text_clips:
            try:
                t.close()
            except Exception:
                pass
        # Encourage Python-side memory release for MoviePy buffers
        gc.collect()

    print('Uploading video to AWS...')
    s3 = boto3.client('s3')  # Ensure AWS credentials are set in your environment or AWS credentials file

    try:
        # Upload the file to S3
        s3.upload_file(output_video_path, bucket_name, output_name)
        print(f"File uploaded successfully! You can download it from:")
        video_url = f"https://{bucket_name}.s3.amazonaws.com/{output_name}"
        print(video_url)
        print('Deleting annotated local video at: ' + output_video_path)
        os.remove(output_video_path)
    except FileNotFoundError:
        print(f"File not found: {output_video_path}")
    except NoCredentialsError:
        print("Credentials not found or not configured correctly.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return video_url

def get_media_info(video_path):
    """Return basic media info for a given video path."""
    try:
        size_bytes = os.path.getsize(video_path)
    except Exception:
        size_bytes = 0
    width = height = total_frames = 0
    fps = 0.0
    duration = 0.0
    try:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = (total_frames / fps) if fps and total_frames > 0 else 0.0
    except Exception:
        pass
    est_avg_bitrate_mbps = ((size_bytes * 8) / 1e6 / duration) if duration > 0 else 0.0
    return {
        'size_bytes': size_bytes,
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total_frames,
        'duration': duration,
        'est_avg_bitrate_mbps': est_avg_bitrate_mbps,
    }

def normalize_video(video_path, target_width=640, target_fps=30):
    """
    Normalize a video to target resolution (keeping aspect) and fps.
    Returns the path to the normalized video. Deletes the original on success.
    """
    base, ext = os.path.splitext(video_path)
    norm_path = f"{base}_norm{ext}"
    vf = f"scale='min({target_width},iw)':'-2':flags=bicubic,fps={target_fps}"
    ffmpeg_path = get_ffmpeg_exe()
    cmd = [
        ffmpeg_path, '-y', '-i', video_path,
        '-loglevel', 'error',
        '-vf', vf,
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '128k',
        '-movflags', '+faststart',
        '-threads', '1',
        norm_path
    ]
    print("Starting normalization with ffmpeg...")
    subprocess.run(cmd, check=True)
    # Delete original to reclaim space
    try:
        os.remove(video_path)
        print(f"Deleted original heavy video: {video_path}")
    except Exception as e:
        print(f"Failed to delete original video: {e}")
    return norm_path

if __name__ == "__main__":
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
    input_path = data_dir + '/inference_vids/1Zj_jAPToxI_6_inf/1Zj_jAPToxI_6_inf.mp4'
    annotate_video(input_path, analysis)