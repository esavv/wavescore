//
//  ContentView.swift
//  surfjudge
//
//  Created by Erik Savage on 11/8/24.
//

import SwiftUI
import PhotosUI
import Foundation
import AVFoundation

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
                    isPickerPresented = true  // Show the video picker when the button is tapped
                }
                .sheet(isPresented: $isPickerPresented) {
                    VideoPicker(selectedVideo: $selectedVideo, videoMetadata: $videoMetadata) {
                        print("Selected Video URL: \(selectedVideo?.absoluteString ?? "No video selected")")
                        if let videoMetadata = videoMetadata {
                            print("Video metadata synced: \(videoMetadata.syncData)")
                            print("Video metadata async: \(videoMetadata.asyncData)")
                            resultText = """
                            Duration: \(videoMetadata.asyncData.duration)
                            File Size: \(videoMetadata.syncData.fileSize)
                            Created: \(videoMetadata.syncData.created)
                            """
                            showResults = true
                        } else {
                            print("Error: videoMetadata is nil.")
                        }                        // Make an API call
                        // *** API block start ********************
//                        if let videoURL = selectedVideo {
//                            print("Video file exists: \(fileExists(at: videoURL))")
//                             // Call the dummy API with no video for hardcoded text
//                             uploadVideoToAPI() { result in
//                             // // Call the API with a video file
//                             // uploadVideoToAPI(videoURL: videoURL) { result in
//                                 // Handle the result returned by the API
//                                 if let result = result {
//                                     resultText = result  // Set the resultText state
//                                 }
//                             }
//                            showResults = true
//                        }
                        // *** API block end **********************
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

struct VideoPicker: UIViewControllerRepresentable {
    @Binding var selectedVideo: URL?  // Binding to hold the selected video URL
    @Binding var videoMetadata: VideoMetadata?  // Binding to update videoMetadata in ContentView
    var completion: (() -> Void)?  // Completion handler to dismiss the picker

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var configuration = PHPickerConfiguration()
        configuration.filter = .videos  // Only allow video selection
        let picker = PHPickerViewController(configuration: configuration)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        var parent: VideoPicker

        init(_ parent: VideoPicker) {
            self.parent = parent
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            if let result = results.first {
                if result.itemProvider.hasItemConformingToTypeIdentifier(UTType.movie.identifier) {
                    result.itemProvider.loadFileRepresentation(forTypeIdentifier: UTType.movie.identifier) { url, error in
                        if let url = url {
                            // Initialize videoMetadata if it's nil
                            if self.parent.videoMetadata == nil {
                                self.parent.videoMetadata = VideoMetadata(
                                    syncData: VideoMetadataSync(fileSize: "Unknown", created: "Unknown"),
                                    asyncData: VideoMetadataAsync(duration: "Unknown")
                                )
                            }
                            // 1. Call the sync video metadata function
                            let videoMetadataSync = inspectVideoInfoSync(videoURL: url)
                            if let videoMetadataSync = videoMetadataSync {
                                print("Sync Metadata: \(videoMetadataSync)")
                                // Update sync metadata directly
                                self.parent.videoMetadata?.syncData = videoMetadataSync
                                print("Sync Metadata in parent: \(String(describing: self.parent.videoMetadata?.syncData))")
                            } else {
                                print("Error: sync metadata is nil.")
                            }

                            // 2. Perform file move synchronously on the main thread
                            if let movedURL = moveVideoToPersistentLocation(from: url) {
                                DispatchQueue.main.async {
                                    print("Video successfully moved to: \(movedURL)")
                                    self.parent.selectedVideo = movedURL
                                }
                            } else {
                                print("Failed to move video to persistent location.")
                            }
                            
                            // 3. Call the async video metadata function with the moved video
                            Task {
                                // Perform the async video metadata inspection
                                let videoMetadataAsync = await inspectVideoInfoAsync(videoURL: self.parent.selectedVideo!)

                                // Ensure UI updates are done on the main thread
                                await MainActor.run {
                                    if let videoMetadataAsync = videoMetadataAsync {
                                        print("Async Metadata: \(videoMetadataAsync)")
                                        // Update async metadata
                                        self.parent.videoMetadata?.asyncData = videoMetadataAsync
                                        print("Async Metadata in parent: \(String(describing: self.parent.videoMetadata?.asyncData))")
                                        picker.dismiss(animated: true)
                                        self.parent.completion?()  // Call the completion handler when done
                                    } else {
                                        print("Error: async metadata is nil.")
                                    }
                                }
                            }
                        } else {
                            print("Error loading file representation: \(error?.localizedDescription ?? "Unknown error")")
                        }
                    }
                }
            } else {
                picker.dismiss(animated: true)
            }
        }
    }
}

struct VideoMetadataSync: Codable {
    let fileSize: String
    let created: String
}

struct VideoMetadataAsync: Codable {
    let duration: String
}

struct VideoMetadata: Codable {
    var syncData: VideoMetadataSync
    var asyncData: VideoMetadataAsync
}

// Function to inspect video metadata synchronously
func inspectVideoInfoSync(videoURL: URL) -> VideoMetadataSync? {
    print("Inspecting sync metadata for video: \(videoURL.path)")
    var fileSizeString: String = "Unknown size"
    var creationDateString: String = "Unknown date"
    
    // Retrieve file size using FileManager
    if let attributes = try? FileManager.default.attributesOfItem(atPath: videoURL.path),
       let fileSize = attributes[.size] as? NSNumber {
        fileSizeString = String(format: "%.2f MB", fileSize.doubleValue / 1_000_000)
        
        // Retrieve creation date
        let creationDate = attributes[.creationDate] as? Date ?? Date()
        let creationDateFormatter = DateFormatter()
        creationDateFormatter.dateStyle = .medium
        creationDateFormatter.timeStyle = .short
        creationDateString = creationDateFormatter.string(from: creationDate)
    }
    
    print("File size: \(fileSizeString), Creation Date: \(creationDateString)")
    // Return the VideoMetadataSync instance
    return VideoMetadataSync(fileSize: fileSizeString, created: creationDateString)
}

// Function to inspect video metadata asynchronously
func inspectVideoInfoAsync(videoURL: URL) async -> VideoMetadataAsync? {
    print("Inspecting async metadata for video: \(videoURL.path)")
    // Create AVAsset instance
    let asset = AVAsset(url: videoURL)
    
    var durationString: String = "Unknown duration"
    
    // Use async to load the duration as it’s now deprecated
    do {
        let duration = try await asset.load(.duration)
        let durationInSeconds = CMTimeGetSeconds(duration)
        if durationInSeconds.isFinite {
            durationString = String(format: "%.2f seconds", durationInSeconds)
        }
    } catch {
        print("Error loading duration: \(error.localizedDescription)")
    }
    
    print("Duration: \(durationString)")
    // Return a VideoMetadataAsync instance with the duration
    return VideoMetadataAsync(duration: durationString)
}

func moveVideoToPersistentLocation(from temporaryURL: URL) -> URL? {
    // Get the Documents directory URL
    let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    
    // Create a destination URL for the video
    let destinationURL = documentsDirectory.appendingPathComponent(temporaryURL.lastPathComponent)
    
    // Always delete the existing video (if any)
    do {
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
            print("Deleted existing video at \(destinationURL.path)")
        }
    } catch {
        print("Error deleting existing video file: \(error)")
        return nil  // Return nil if deletion fails
    }
    
    // Now proceed with copying the new video
    do {
        // Copy the video from the temporary URL to the Documents directory
        try FileManager.default.copyItem(at: temporaryURL, to: destinationURL)
        return destinationURL  // Return the new URL for use
    } catch {
        print("Error copying video file: \(error)")
        return nil
    }
}

func fileExists(at url: URL) -> Bool {
    return FileManager.default.fileExists(atPath: url.path)
}

struct APIResponse: Codable {
    let result: String
}

func uploadVideoToAPI(completion: @escaping (String?) -> Void) {
// func uploadVideoToAPI(videoURL: URL, completion: @escaping (String?) -> Void) {
    let url = URL(string: "https://6cd5-70-23-3-136.ngrok-free.app/upload_video")!  // Replace with your server's URL
//    let url = URL(string: "http://192.168.1.151:5000/upload_video")!  // Replace with your server's URL
//    let url = URL(string: "http://127.0.0.1:5000/upload_video")!  // Replace with your server's URL
    
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    
    // *** Dummy Body Start ****************************************
    // Remove multipart form data (we are not sending the video anymore)
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    
    // Non-Video Body: Send a dummy JSON object or empty body to the API
    let jsonBody = ["message": "Testing connection, no video file"]  // Adjust according to your API's expected structure
    do {
        let jsonData = try JSONSerialization.data(withJSONObject: jsonBody, options: [])
        request.httpBody = jsonData
        print("Request: \(request)")  // Print the request details
    } catch {
        print("Error creating JSON body: \(error.localizedDescription)")
        completion(nil)
        return
    }
    // *** Dummy Body End ******************************************

    // *** Video Body Start ****************************************
//    // Replace "Non-Video Body" above with this chunk below for video requests
//    // Create multipart form data body to send the video file
//    let boundary = "Boundary-\(UUID().uuidString)"
//    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
//    
//    var body = Data()
//    body.append("--\(boundary)\r\n".data(using: .utf8)!)
//    body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(videoURL.lastPathComponent)\"\r\n".data(using: .utf8)!)
//    body.append("Content-Type: video/mp4\r\n\r\n".data(using: .utf8)!)
//    do {
//        let videoData = try Data(contentsOf: videoURL)
//        body.append(videoData)
//        body.append("\r\n".data(using: .utf8)!)
//    } catch {
//        print("Error reading video data: \(error.localizedDescription)")
//        completion(nil)  // Call completion with nil in case of error
//        return
//    }
//    body.append("--\(boundary)--\r\n".data(using: .utf8)!)
//    request.httpBody = body
    // *** Video Body End ******************************************

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
