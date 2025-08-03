from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import boto3, os, csv
from botocore.exceptions import NoCredentialsError

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
    # Load the video
    clip = VideoFileClip(input_path)

    # Create a list of text clips to overlay on the video
    text_clips = []

    # Overlay maneuvers
    if os.uname().sysname == 'Darwin':
        font='Arial'
    else:
        font='DejaVuSans'
    for maneuver in analysis['maneuvers']:
        start_time = maneuver['start_time']
        end_time = maneuver['end_time']
        text = maneuver['name']

        # Create a TextClip for each maneuver
        txt_clip = (TextClip(text=text, font=font, font_size=40, color='white')
                    .with_position(('center', clip.h * 0.85))
                    .with_duration(end_time - start_time)
                    .with_start(start_time))

        text_clips.append(txt_clip)

    # Create a TextClip for the predicted score (overlay it in the last 2 seconds)
    score_text = f"Predicted Score: {analysis['score']}"
    score_clip = (TextClip(text=score_text, font=font, font_size=40, color='white')
                  .with_position(('center', 50))
                  .with_duration(2)  # Last 2 seconds of the video
                  .with_start(clip.duration - 2))  # Start 2 seconds before the end

    text_clips.append(score_clip)

    # Overlay the text clips on the video
    video = CompositeVideoClip([clip] + text_clips)

    # Write the result to a file
    input_name = os.path.splitext(os.path.basename(input_path))[0]
    input_ext = os.path.splitext(input_path)[1]
    output_name = input_name + "_annotated" + input_ext
    output_video_path = "/tmp/" + output_name
    print("Saving annotated video to: " + output_video_path)
    video.write_videofile(output_video_path, codec='libx264')

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