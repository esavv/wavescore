//
//  VideoPicker.swift
//  surfjudge
//
//  Created by Erik Savage on 11/29/24.
//

import SwiftUI
import PhotosUI

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
