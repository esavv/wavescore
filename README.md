# WaveScore

Upload surf videos and get your ride scored from 0 to 10 as if you're in a [surf competition](https://en.wikipedia.org/wiki/World_Surf_League#Judging[28]).

## Check it out!

Try it here: [wavescore.xyz](https://www.wavescore.xyz/)

... or [watch a demo](https://youtu.be/qYMjHPNFqr0)

## Roadmap

### Current & Upcoming Tasks
   - Web app optimization & bugfixes
   - Cleanup: Organize `src` files into subdirectories
   - Cleanup: Refactor `src` to use filepaths relative to the absolute path for the main directory
   - Migrate maneuver prediction to TCN architecture to predict sequence of maneuvers from single video
   - Streamline data labeling workflow & updating maneuver taxonomy for falls / failed moves
   - Scale training data to improve predictions
   - Scale backend infra to enable concurrency
   - Generate progressive score prediction: show user how predicted score changes as video progresses
   - Migrate data from directory system to postgres + blob storage (S3)

### Completed Milestones
   - *Aug 2025:* Replaced iOS app with web app at [wavescore.xyz](https://www.wavescore.xyz/)
   - *Jun 2025:* Trained score prediction model (architecture: transformer-based vision encoder with temporal pooling)
   - *May 2025:* Switch maneuver model architecture to 3D CNN and deploy to API to AWS
   - *Apr 2025:* Convert API to SSE to display upload progress to user
   - *Dec 2024:* Basic iOS app connected to maneuver inference API
   - *Nov 2024:* Trained maneuver prediction model and initial maneuver inference API (architecture: CNN + LSTM)
   - *Oct 2024:* Initial data pipeline and sourced sample training videos