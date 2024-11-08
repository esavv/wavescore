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

    var body: some View {
        VStack {
//            Image(systemName: "globe")
//                .imageScale(.large)
//                .foregroundStyle(.tint)
//            Text("Hello, world!")
            Button("Upload Surf Video") {
                isPickerPresented = true  // Show the video picker when the button is tapped
            }
            .sheet(isPresented: $isPickerPresented) {
                VideoPicker(selectedVideo: $selectedVideo) {
                    // Do something with the selected video, like running inference
                    print("Selected Video URL: \(selectedVideo?.absoluteString ?? "No video selected")")
                }
            }

            // Optional: Display the selected video URL
            if let videoURL = selectedVideo {
                Text("Selected Video: \(videoURL.lastPathComponent)")
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
