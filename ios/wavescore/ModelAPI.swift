//
//  ModelAPI.swift
//  wavescore
//
//  Created by Erik Savage on 11/29/24.
//

import SwiftUI

struct APIResponse: Codable {
    let status: String  // "success" or "error"
    let message: String // Descriptive message
    let video_url: String? // Optional URL for annotated video (only present on success)
}

func uploadVideoToAPI(videoURL: URL,
                      onProgress: @escaping (String) -> Void,
                      onComplete: @escaping (APIResponse?) -> Void) {
//     let url = URL(string: "https://surfjudge-api-71248b819ca4.herokuapp.com/upload_video_hardcode")!
    let url = URL(string: "https://9a81-2600-4041-5986-7c00-b031-b239-37f9-eb57.ngrok-free.app/upload_video_hardcode_sse")!
    
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
        print("Error reading video data: \(error)")
        onComplete(nil)
        return
    }
    body.append("--\(boundary)--\r\n".data(using: .utf8)!)
    request.httpBody = body

    // Configure a custom URLSession with extended timeout
    let config = URLSessionConfiguration.default
    config.timeoutIntervalForRequest = 180 // 3 minutes
    config.timeoutIntervalForResource = 180
    let session = URLSession(configuration: config)

    // Make the network request
    // Create a stream task manually using InputStream
    let task = session.uploadTask(with: request, from: body) { data, response, error in
        // This part may not even be needed if SSE responses arrive through the stream below
    }

    let streamTask = session.uploadTask(withStreamedRequest: request)
    streamTask.resume()

    // This is where we "intercept" the streamed response (SSE)
    session.dataTask(with: request) { data, response, error in
        let inputStream = InputStream(data: data ?? Data())

        inputStream.open()
        let bufferSize = 4096
        var buffer = [UInt8](repeating: 0, count: bufferSize)

        var partialData = ""

        while inputStream.hasBytesAvailable {
            let bytesRead = inputStream.read(&buffer, maxLength: bufferSize)
            if bytesRead > 0 {
                if let chunk = String(bytes: buffer[0..<bytesRead], encoding: .utf8) {
                    partialData += chunk

                    // Break on each event
                    let events = partialData.components(separatedBy: "\n\n")
                    for event in events.dropLast() {
                        if event.starts(with: "data: ") {
                            let jsonString = event.dropFirst(6) // remove "data: "
                            if let jsonData = jsonString.data(using: .utf8) {
                                do {
                                    let response = try JSONDecoder().decode(APIResponse.self, from: jsonData)
                                    DispatchQueue.main.async {
                                        if response.status == "interim" {
                                            onProgress(response.message)
                                        } else {
                                            onComplete(response)
                                        }
                                    }
                                } catch {
                                    print("Failed to decode streamed message: \(error)")
                                }
                            }
                        }
                    }
                    partialData = events.last ?? ""
                }
            }
        }
        inputStream.close()
    }.resume()
}

