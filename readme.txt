## How to download a YouTube video using yt-dlp:
 > yt-dlp https://www.youtube.com/watch?v=1Zj_jAPToxI

## How to clip a shorter video from a longer video and save it to the appropriate directory:
   # Assumes we're in the /surfjudge/clips directory
 > ffmpeg -i ../videos/1Zj_jAPToxI.mp4 -ss 00:00:17 -to 00:00:46 -c:v libx264 -c:a aac 1Zj_jAPToxI_1.mp4

## How to convert a longer video into a sequence of clips:
 - Ensure the target video is in the /videos directory
 - Add a csv to the /clip_times directory that contains the start & end timestamps for each clip desired
 - From the main /surfjudge directory, run this command:
   > python3 src/clipify.py videos/target.mp4 clip_times/target_times.csv
 - This runs a python script that outputs the desired clips to the /clips directory
