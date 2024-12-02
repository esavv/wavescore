//
//  ModelAPI.swift
//  surfjudge
//
//  Created by Erik Savage on 11/29/24.
//

import SwiftUI

struct APIResponse: Codable {
    let result: String
}

func uploadVideoToAPI(videoURL: URL, completion: @escaping (String?) -> Void) {
    let url = URL(string: "https://surfjudge-api-71248b819ca4.herokuapp.com/upload_video")!
//    let url = URL(string: "https://c74e-70-23-3-136.ngrok-free.app/upload_video")!
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
