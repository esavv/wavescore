//
//  ContentView.swift
//  surfjudge
//
//  Created by Erik Savage on 11/8/24.
//

import SwiftUI
import PhotosUI
import Foundation

struct ContentView: View {
    @State private var selectedVideo: URL?  // State to hold the selected video URL
    @State private var isPickerPresented = false  // State to control the video picker presentation
    @State private var showResults = false  // State to show results after video upload
    @State private var resultText: String?  // Store the result from the API

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
                    VideoPicker(selectedVideo: $selectedVideo) {
                        // Do something with the selected video, like running inference
                        print("Selected Video URL: \(selectedVideo?.absoluteString ?? "No video selected")")
                        // Simulate results after video is uploaded
                        showResults = true
                            if let videoURL = selectedVideo {
                            uploadVideoToAPI() { result in
//                            uploadVideoToAPI(videoURL: videoURL) { result in
                                // Handle the result returned by the API
                                if let result = result {
                                    resultText = result  // Set the resultText state
                                }
                            }
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

struct VideoPicker: UIViewControllerRepresentable {
    @Binding var selectedVideo: URL?  // Binding to hold the selected video URL
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
                            DispatchQueue.main.async {
                                self.parent.selectedVideo = url
                                picker.dismiss(animated: true)
                                self.parent.completion?()  // Call the completion handler if provided
                            }
                        }
                    }
                }
            } else {
                picker.dismiss(animated: true)
            }
        }
    }
}

struct APIResponse: Codable {
    let result: String
}

func uploadVideoToAPI(completion: @escaping (String?) -> Void) {
    let url = URL(string: "https://6cd5-70-23-3-136.ngrok-free.app/upload_video")!  // Replace with your server's URL
//    let url = URL(string: "http://192.168.1.151:5000/upload_video")!  // Replace with your server's URL
//    let url = URL(string: "http://127.0.0.1:5000/upload_video")!  // Replace with your server's URL
    
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    
    // Remove multipart form data (we are not sending the video anymore)
    let boundary = "Boundary-\(UUID().uuidString)"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    
    // Send a dummy JSON object or empty body to the API
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


//func uploadVideoToAPI(videoURL: URL, completion: @escaping (String?) -> Void) {
//    let url = URL(string: "http://192.168.1.151:5000/upload_video")!  // Replace with your server's URL
//    // let url = URL(string: "http://127.0.0.1:5000/upload_video")!  // Replace with your server's URL
//    
//    var request = URLRequest(url: url)
//    request.httpMethod = "POST"
//    
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
//    
//    request.httpBody = body
//    
//    // Make the network request
//    let task = URLSession.shared.dataTask(with: request) { data, response, error in
//        if let error = error {
//            print("Error: \(error)")
//            completion(nil)  // Call completion with nil in case of error
//            return
//        }
//        guard let data = data else {
//            completion(nil)  // Call completion with nil in case of error
//            return
//        }
//        
//        do {
//            // Parse the JSON response
//            let apiResponse = try JSONDecoder().decode(APIResponse.self, from: data)
//            completion(apiResponse.result)  // Return the result via the completion handler
//        } catch {
//            print("Failed to decode response: \(error)")
//            completion(nil)  // Call completion with nil in case of decode error
//        }
//    }
//    task.resume()
//}
