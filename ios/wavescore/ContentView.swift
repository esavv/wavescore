//
//  ContentView.swift
//  wavescore
//
//  Created by Erik Savage on 11/8/24.
//

import SwiftUI
import PhotosUI
import AVKit

enum AppState {
    case home, loading, results
}

struct ContentView: View {
    @State private var appState: AppState = .home  // Initial state is 'home'
    @State private var isPickerPresented = false  // State to control the video picker presentation
    @State private var selectedVideo: URL?  // State to hold the selected video URL
    @State private var videoMetadata: VideoMetadata? // Store video metadata from user-uploaded surf video
    @State private var apiResponse: APIResponse?  // Store the API response
    @State private var localVideoURL: URL?
    @State private var showToast = false  // Controls whether the toast is visible
    @State private var toastMessage = ""  // The message to show in the toast
    @State private var toastDuration: Double = 2.0  // Duration for the toast to be visible
    @State private var isPlayerReady = false
    
    var body: some View {
        VStack {
            switch appState {
            case .home:
                // Show the video upload button if showResults is false
                Button(action: {
                    PHPhotoLibrary.requestAuthorization { (status) in
                        if status == .authorized {
                            print("Status is authorized")
                        } else {
                            print("Status is denied or something else")
                        }
                    }
                    isPickerPresented = true  // Show the video picker when the button is tapped
                }) {
                    Label("Upload Surf Video", systemImage: "square.and.arrow.up")
                        .padding()
                        .foregroundColor(.blue)
                }
                .sheet(isPresented: $isPickerPresented) {
                    VideoPicker(selectedVideo: $selectedVideo, videoMetadata: $videoMetadata) {
                        print("Selected Video URL: \(selectedVideo?.absoluteString ?? "No video selected")")
                        appState = .loading
                        // Make an API call
                        if let videoURL = selectedVideo {
                            print("Calling the API now...")
                            // Call the API with a video file
                            uploadVideoToAPI(videoURL: videoURL) { response in
                                // Handle the result returned by the API
                                DispatchQueue.main.async {
                                    apiResponse = response  // Set the resultText state
                                    appState = .results  // Transition to results state after receiving the response
                                    // If the API response is successful, download the video locally
                                    if let videoURLString = response?.video_url, let url = URL(string: videoURLString) {
                                        downloadVideo(from: url)
                                    } else {
                                        isPlayerReady = true
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Optional: Display the selected video URL
                if let videoURL = selectedVideo {
                    Text("Selected Video: \(videoURL.lastPathComponent)")
                }
                
            case .loading:
                // Show loading indicator
                Text("Analyzing video...")
                    .font(.headline)
                    .padding()
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle())
                    .padding()
                
            case .results:
                // Display the result text (from API response) and hardcoded "Nice surfing!"
                if isPlayerReady {
                    if let response = apiResponse {
                        if response.status == "success", let localVideoURL = localVideoURL {
                            // Play the saved video
                            VideoPlayer(player: AVPlayer(url: localVideoURL))
                                .frame(height: 300)
                                .padding()
                            Text(response.message)
                                .font(.body)
                                .padding()
                            // Add button to save video to the photo library
                            Button(action: {
                                saveVideoToPhotos(localVideoURL)
                            }) {
                                Label("Save Video", systemImage: "square.and.arrow.down")
                                    .foregroundColor(.blue)
                            }
                            .padding()
                        } else {
                            // Error case: Show only the error message
                            Text("\(response.message)")
                                .font(.body)
                                .foregroundColor(.red)
                                .padding()
                        }
                        
                    } else {
                        Text("Something went wrong. Please try again.")
                            .font(.body)
                            .foregroundColor(.red)
                            .padding()
                    }
                    Button(action: {
                        // Reset the state to allow uploading a new video
                        appState = .home
                        selectedVideo = nil
                        isPickerPresented = true  // Open video picker again
                        isPlayerReady = false
                    }) {
                        Label("Upload Another Video", systemImage: "square.and.arrow.up")
                            .foregroundColor(.blue)
                    }
                    .padding()
                } else {
                    // Optionally, show a loading indicator here
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle())
                        .padding()
                }
            }
        }
        .padding()
        .overlay(
            // Toast notification
            ToastView(message: toastMessage, show: $showToast)
        )
    }
    // Download video and save to app's temp directory
    private func downloadVideo(from url: URL) {
        let tempDirectory = FileManager.default.temporaryDirectory
        let tempFileURL = tempDirectory.appendingPathComponent(url.lastPathComponent)

        // Check if the file already exists, and delete it if necessary
        if FileManager.default.fileExists(atPath: tempFileURL.path) {
            do {
                try FileManager.default.removeItem(at: tempFileURL)
                print("Previous temp video deleted")
            } catch {
                print("Failed to delete previous temp video: \(error.localizedDescription)")
            }
        }

        URLSession.shared.downloadTask(with: url) { (location, response, error) in
            if let location = location {
                do {
                    try FileManager.default.moveItem(at: location, to: tempFileURL)
                    DispatchQueue.main.async {
                        localVideoURL = tempFileURL  // Set the local video URL to be used later
                        isPlayerReady = true  // Mark the player as ready
                    }
                } catch {
                    print("Failed to save video: \(error.localizedDescription)")
                }
            } else if let error = error {
                print("Download failed: \(error.localizedDescription)")
            }
        }.resume()
    }

    // Save the video to the photo library
    private func saveVideoToPhotos(_ url: URL) {
        PHPhotoLibrary.requestAuthorization { status in
            if status == .authorized {
                do {
                    try PHPhotoLibrary.shared().performChangesAndWait {
                        let assetChangeRequest = PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
                        if let _ = assetChangeRequest {
                            print("Video saved to library")
                        }
                        toastMessage = "Video Saved"
                        showToast = true
                    }
                } catch {
                    toastMessage = "Error saving video"
                    showToast = true
                    print("Error saving video: \(error.localizedDescription)")
                }
            } else {
                print("Photo library access denied")
            }
        }
    }
}

struct BlurView: UIViewRepresentable {
    func makeUIView(context: Context) -> UIVisualEffectView {
        let blurEffect = UIBlurEffect(style: .systemMaterial)
        let blurView = UIVisualEffectView(effect: blurEffect)
        return blurView
    }

    func updateUIView(_ uiView: UIVisualEffectView, context: Context) {
        // Nothing to update here for now
    }
}

// Custom Toast View
struct ToastView: View {
    var message: String
    @Binding var show: Bool
    
    var body: some View {
        VStack {
            if show {
                HStack {
                    VStack {
                        Image(systemName: "checkmark")
                            .foregroundColor(.black)
                            .font(.system(size: 40))  // Make checkmark larger if needed
                            
                        Text(message)
                            .font(.body)
                            .foregroundColor(.black)  // Dark gray text
                            .padding(.top, 4)  // Add space between checkmark and text
                    }
                    .padding()
                    .frame(width: 250, height: 250)  // Widen the toast and make it square-shaped
                    .background(BlurView())  // Semi-transparent light gray background
                    .cornerRadius(8)
                    .transition(.slide)
                }
                .padding(.top, 40)
                .padding(.horizontal, 20)
                .frame(maxWidth: .infinity, alignment: .top)
                .onAppear {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                        withAnimation {
                            show = false
                        }
                    }
                }
            }
        }
        .padding(.top, 40)
        .padding(.horizontal, 20)
        .frame(maxWidth: .infinity, alignment: .top)
    }
}

#Preview {
    ContentView()
}
