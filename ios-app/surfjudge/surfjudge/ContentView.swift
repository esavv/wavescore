//
//  ContentView.swift
//  surfjudge
//
//  Created by Erik Savage on 11/8/24.
//

import SwiftUI
import PhotosUI

struct ContentView: View {
    @State private var selectedVideo: URL?  // State to hold the selected video URL
    @State private var isPickerPresented = false  // State to control the video picker presentation
    @State private var showResults = false  // State to show results after video upload
    @State private var resultText: String?  // Store the result from the API
    @State private var videoMetadata: VideoMetadata? // Store video metadata from user-uploaded surf video

    var body: some View {
        VStack {
            if showResults {
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
                        showResults = false
                        selectedVideo = nil
                        isPickerPresented = true  // Open video picker again
                    }
                    .padding()
                }
            } else {
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
                        // Make an API call
                        if let videoURL = selectedVideo {
                            print("Calling the API now...")
                            // Call the API with a video file
                            uploadVideoToAPI(videoURL: videoURL) { result in
                                // Handle the result returned by the API
                                if let result = result {
                                    resultText = result  // Set the resultText state
                                }
                            }
                            showResults = true
                        }
                    }
                }
                
                // Optional: Display the selected video URL
                if let videoURL = selectedVideo {
                    Text("Selected Video: \(videoURL.lastPathComponent)")
                }
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}

struct APIResponse: Codable {
    let result: String
}

func uploadVideoToAPI(videoURL: URL, completion: @escaping (String?) -> Void) {
    let url = URL(string: "https://surfjudge-api-71248b819ca4.herokuapp.com/upload_video")!
//    let url = URL(string: "https://2947-70-23-3-136.ngrok-free.app/upload_video")!
//    let url = URL(string: "http://127.0.0.1:5000/upload_video")!  // Replace with your server's URL
    
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    
    // Create multipart form data body to send the video file
    let boundary = "Boundary-\(UUID().uuidString)"
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
    
    var body = Data()
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(videoURL.lastPathComponent)\"\r\n".data(using: .utf8)!)
    body.append("Content-Type: video/mp4\r\n\r\n".data(using: .utf8)!)
    do {
        let videoData = try Data(contentsOf: videoURL)
        body.append(videoData)
        body.append("\r\n".data(using: .utf8)!)
    } catch {
        print("Error reading video data: \(error.localizedDescription)")
        completion(nil)  // Call completion with nil in case of error
        return
    }
    body.append("--\(boundary)--\r\n".data(using: .utf8)!)
    request.httpBody = body

    // Make the network request
    let task = URLSession.shared.dataTask(with: request) { data, response, error in
        if let error = error {
            print("Error: \(error)")
            completion(nil)  // Call completion with nil in case of error
            return
        }
        
        guard let data = data else {
            completion(nil)  // Call completion with nil if no data returned
            return
        }
        
        // Print raw response for debugging
        if let rawResponse = String(data: data, encoding: .utf8) {
            print("Raw Response: \(rawResponse)")
        }
        
        do {
            // Parse the JSON response (e.g., return a hardcoded message from the API)
            let apiResponse = try JSONDecoder().decode(APIResponse.self, from: data)
            completion(apiResponse.result)  // Return the result via the completion handler
        } catch {
            print("Failed to decode response: \(error)")
            completion(nil)  // Call completion with nil in case of decode error
        }
    }
    task.resume()
}
