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
import Photos
import CoreLocation

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
//                        if let videoMetadata = videoMetadata {
//                            resultText = """
//                            Duration: \(videoMetadata.duration)
//                            File Size: \(videoMetadata.fileSize)
//                            Created: \(videoMetadata.created)
//                            Lat/Long: \(videoMetadata.latlon)
//                            """
//                            showResults = true
//                        } else {
//                            print("Error: videoMetadata is nil.")
//                        }
                        // Make an API call
                        // *** API block start ********************
                        if let videoURL = selectedVideo {
                            print("Video file exists: \(fileExists(at: videoURL))")
                            print("Calling the API now...")
//                            // Call the dummy API with no video for hardcoded text
//                            uploadVideoToAPI() { result in
                             // Call the API with a video file
                             uploadVideoToAPI(videoURL: videoURL) { result in
                                // Handle the result returned by the API
                                if let result = result {
                                    resultText = result  // Set the resultText state
                                }
                            }
                            showResults = true
                        }
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
        var configuration = PHPickerConfiguration(photoLibrary: PHPhotoLibrary.shared())
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
                                self.parent.videoMetadata = VideoMetadata(fileSize: "Unknown", created: "Unknown", duration: "Unknown", latlon: "Unknown")
                            }

                            // Get the localIdentifier of the video to fetch the PHAsset
                            let localIdentifier = result.assetIdentifier
                            if let localIdentifier = localIdentifier {
                                let assets = PHAsset.fetchAssets(withLocalIdentifiers: [localIdentifier], options: nil)
                                print("assets: \(assets)")

                                if let phAsset = assets.firstObject {
                                    print("phAsset: \(phAsset)")
                                    // 1. Extract sync metadata (duration, creation date, file size, geolocation)
                                    if let videoMetadata = inspectVideoInfo(phAsset: phAsset, videoUrl: url) {
                                        self.parent.videoMetadata = videoMetadata
                                    } else {
                                        print("Error: videoMetadata is nil.")
                                    }
                                } else {
                                    print("Error: phAsset not available.")
                                }
                            } else {
                                print("Error: localIdentifier not available.")
                            }

                            // 2. Perform file move synchronously on the main thread
                            if let movedURL = moveVideoToPersistentLocation(from: url) {
                                DispatchQueue.main.async {
                                    print("Video successfully moved to: \(movedURL)")
                                    self.parent.selectedVideo = movedURL
                                    picker.dismiss(animated: true)
                                    self.parent.completion?()  // Call the completion handler when done
                                }
                            } else {
                                print("Failed to move video to persistent location.")
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

struct VideoMetadata: Codable {
    let fileSize: String
    let created: String
    let duration: String
    let latlon: String
}

// Function to inspect video metadata synchronously
func inspectVideoInfo(phAsset: PHAsset, videoUrl: URL) -> VideoMetadata? {
    print("Inspecting sync metadata for video")
    var fileSizeString: String = "Unknown size"
    var creationDateString: String = "Unknown date"
    var durationString: String = "Unknown duration"
    var latlonString: String = "Unknown lat/long"

    // Get creation date directly from PHAsset
    if let creationDate = phAsset.creationDate {
        let creationDateFormatter = DateFormatter()
        creationDateFormatter.dateStyle = .medium
        creationDateFormatter.timeStyle = .short
        creationDateString = creationDateFormatter.string(from: creationDate)
    } else {
        print("...Error: Creation date not found")
    }

    // If needed, you can get the URL for the video using the localIdentifier
    let fileManager = FileManager.default
    do {
        let attributes = try fileManager.attributesOfItem(atPath: videoUrl.path)
        if let fileSize = attributes[.size] as? NSNumber {
            fileSizeString = String(format: "%.2f MB", fileSize.doubleValue / 1_000_000)
        }
    } catch {
        print("...Error: Unable to retrieve file size from temporary URL")
    }
    // Retrieve duration directly from PHAsset
    durationString = String(format: "%.2f seconds", phAsset.duration)

    // Retrieve geolocation data (latitude and longitude) from PHAsset
    if let location = phAsset.location {
        let latitude = String(location.coordinate.latitude)
        let longitude = String(location.coordinate.longitude)
        latlonString = latitude + ", " + longitude
    } else {
        print("...Error: phAsset location not vound")
    }
    
    print("File size: \(fileSizeString),\n Creation Date: \(creationDateString),\n Duration is: \(durationString),\n Latlon is: \(latlonString)")
    // Return the VideoMetadataSync instance
    return VideoMetadata(fileSize: fileSizeString, created: creationDateString, duration: durationString, latlon: latlonString)
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

//func uploadVideoToAPI(completion: @escaping (String?) -> Void) {
func uploadVideoToAPI(videoURL: URL, completion: @escaping (String?) -> Void) {
    let url = URL(string: "https://e21e-70-23-3-136.ngrok-free.app/upload_video")!  // Replace with your server's URL
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
    // Replace "Non-Video Body" above with this chunk below for video requests
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
