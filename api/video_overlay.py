from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

def annotate_video(input_video_path, output_video_path, analysis):
    # Load the video
    clip = VideoFileClip(input_video_path)

    # Create a list of text clips to overlay on the video
    text_clips = []

    # Overlay maneuvers
    for maneuver in analysis['maneuvers']:
        start_time = maneuver['start_time']
        end_time = maneuver['end_time']
        text = maneuver['name']

        # Create a TextClip for each maneuver
        txt_clip = (TextClip(text=text, font='Arial', font_size=40, color='white')
                    .with_position(('center', clip.h * 0.85))
                    .with_duration(end_time - start_time)
                    .with_start(start_time))

        text_clips.append(txt_clip)

    # Create a TextClip for the predicted score (overlay it in the last 2 seconds)
    score_text = f"Predicted Score: {analysis['score']}"
    score_clip = (TextClip(text=score_text, font='Arial', font_size=40, color='white')
                  .with_position(('center', 50))
                  .with_duration(2)  # Last 2 seconds of the video
                  .with_start(clip.duration - 2))  # Start 2 seconds before the end

    text_clips.append(score_clip)

    # Overlay the text clips on the video
    video = CompositeVideoClip([clip] + text_clips)

    # Write the result to a file
    video.write_videofile(output_video_path, codec='libx264')

# Example usage
maneuvers = [
    {'name': '360', 'start_time': 3.0, 'end_time': 5.0},
    {'name': 'Snap', 'start_time': 6.0, 'end_time': 8.0},
    {'name': 'Snap', 'start_time': 10.0, 'end_time': 11.0},
    {'name': 'Cutback', 'start_time': 14.0, 'end_time': 15.0},
    {'name': 'Cutback', 'start_time': 17.0, 'end_time': 18.0},
    {'name': 'Cutback', 'start_time': 20.0, 'end_time': 21.0},
    {'name': 'Cutback', 'start_time': 23.0, 'end_time': 24.0}
]
score = 8.5
analysis = {'maneuvers': maneuvers, 'score': score}

input_video_path = "../data/inference_vids/1Zj_jAPToxI_6_inf/1Zj_jAPToxI_6_inf.mp4"
output_video_path = "1Zj_jAPToxI_6_annotated.mp4"
annotate_video(input_video_path, output_video_path, analysis)
