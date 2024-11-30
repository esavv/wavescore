//
//  ContentView.swift
//  surfjudge
//
//  Created by Erik Savage on 11/8/24.
//

import SwiftUI
import PhotosUI

enum AppState {
    case home, loading, results
}

struct ContentView: View {
    @State private var appState: AppState = .home  // Initial state is 'default'
    @State private var isPickerPresented = false  // State to control the video picker presentation
    @State private var selectedVideo: URL?  // State to hold the selected video URL
    @State private var videoMetadata: VideoMetadata? // Store video metadata from user-uploaded surf video
    @State private var resultText: String?  // Store the result from the API

    var body: some View {
        VStack {
            switch appState {
            case .home:
                // Show the video upload button if showResults is false
                Button("Upload Surf Video") {
                    PHPhotoLibrary.requestAuthorization { (status) in
                        if status == .authorized {
                            print("Status is authorized")
                        } else if status == .denied {
                            print("Status is denied")
                        } else {
                            print("Status is something else")
                        }
                    }
                    isPickerPresented = true  // Show the video picker when the button is tapped
                }
                .sheet(isPresented: $isPickerPresented) {
                    VideoPicker(selectedVideo: $selectedVideo, videoMetadata: $videoMetadata) {
                        print("Selected Video URL: \(selectedVideo?.absoluteString ?? "No video selected")")
                        appState = .loading
                        // Make an API call
                        if let videoURL = selectedVideo {
                            print("Calling the API now...")
                            // Call the API with a video file
                            uploadVideoToAPI(videoURL: videoURL) { result in
                                // Handle the result returned by the API
                                DispatchQueue.main.async {
                                    if let result = result {
                                        resultText = result  // Set the resultText state
                                    }
                                    appState = .results  // Transition to results state after receiving the response
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
            }
                
            case .results:
                // Display the result text (from API response) and hardcoded "Nice surfing!"
                if let resultText = resultText {
                    Text(resultText)  // Display the maneuvers and "3 maneuvers performed"
                        .font(.body)
                        .padding()
                    
                    Text("Nice surfing!")  // Hardcoded message in iOS app
                        .font(.subheadline)
                        .padding()
                    
                    Button("Upload Another Video") {
                        // Reset the state to allow uploading a new video
                        appState = .home
                        selectedVideo = nil
                        isPickerPresented = true  // Open video picker again
                    }
                    .padding()
                }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
