import SwiftUI
import Foundation
import AVFoundation
import AppKit

// MARK: - Data Models (unchanged)
struct InsectDetection: Codable, Identifiable {
    let id = UUID()
    let frameNumber: Int
    let timestamp: Double
    let bbox: BoundingBox
    let center: Point
    let area: Double
    let velocity: Point
    let confidence: Double
    let trackId: Int?
    
    enum CodingKeys: String, CodingKey {
        case frameNumber = "frame_number"
        case timestamp, bbox, center, area, velocity, confidence
        case trackId = "track_id"
    }
}

struct BoundingBox: Codable {
    let x, y, width, height: Int
    
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let array = try container.decode([Int].self)
        guard array.count == 4 else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Expected 4 values for bbox")
        }
        x = array[0]
        y = array[1]
        width = array[2]
        height = array[3]
    }
}

struct Point: Codable {
    let x, y: Double
    
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let array = try container.decode([Double].self)
        guard array.count == 2 else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Expected 2 values for point")
        }
        x = array[0]
        y = array[1]
    }
}

struct TrackInfo: Codable, Identifiable {
    let id = UUID()
    let trackId: Int
    let detectionCount: Int
    let firstFrame: Int
    let lastFrame: Int
    let avgConfidence: Double
    let maxVelocity: Double
    let path: [Point]
    
    enum CodingKeys: String, CodingKey {
        case trackId = "track_id"
        case detectionCount = "detection_count"
        case firstFrame = "first_frame"
        case lastFrame = "last_frame"
        case avgConfidence = "avg_confidence"
        case maxVelocity = "max_velocity"
        case path
    }
}

struct DetectionResults: Codable {
    let metadata: Metadata
    let detections: [InsectDetection]
    let summary: Summary
}

struct Metadata: Codable {
    let totalDetections: Int
    let uniqueTracks: Int
    let processedFrames: Int
    
    enum CodingKeys: String, CodingKey {
        case totalDetections = "total_detections"
        case uniqueTracks = "unique_tracks"
        case processedFrames = "processed_frames"
    }
}

struct Summary: Codable {
    let totalDetections: Int
    let uniqueTracks: Int
    let framesWithDetections: Int
    let tracksInfo: [TrackInfo]
    
    enum CodingKeys: String, CodingKey {
        case totalDetections = "total_detections"
        case uniqueTracks = "unique_tracks"
        case framesWithDetections = "frames_with_detections"
        case tracksInfo = "tracks_info"
    }
}

// MARK: - Status Response Model
struct StatusResponse: Codable {
    let jobId: String
    let status: String
    let progress: Double
    let message: String
    let results: DetectionResults?
    let error: String?
    let outputFiles: [String: String]?
    let createdAt: Double
    let framesProcessed: Int
    let totalFrames: Int
    let detectionsCount: Int
    let uniqueTracks: Int
    
    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case status, progress, message, results, error
        case outputFiles = "output_files"
        case createdAt = "created_at"
        case framesProcessed = "frames_processed"
        case totalFrames = "total_frames"
        case detectionsCount = "detections_count"
        case uniqueTracks = "unique_tracks"
    }
}

// MARK: - Upload Response Model
struct UploadResponse: Codable {
    let jobId: String
    let status: String
    let message: String
    
    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case status, message
    }
}

// MARK: - API Service
class InsectDetectionAPI: ObservableObject {
    @Published var isProcessing = false
    @Published var processingProgress: Double = 0.0
    @Published var results: DetectionResults?
    @Published var error: String?
    @Published var currentJobId: String?
    @Published var processingMessage: String = ""
    @Published var framesProcessed: Int = 0
    @Published var totalFrames: Int = 0
    @Published var detectionsCount: Int = 0
    @Published var uniqueTracks: Int = 0
    
    private let baseURL = "http://127.0.0.1:5000/api"
    private var progressTimer: Timer?
    
    func processVideo(url: URL) async {
        print("Starting video processing for: \(url.path)")
        await MainActor.run {
            isProcessing = true
            processingProgress = 0.0
            error = nil
            results = nil
        }
        
        do {
            print("Uploading video...")
            let jobId = try await uploadVideo(url: url)
            print("Upload successful, job ID: \(jobId)")
            await MainActor.run {
                self.currentJobId = jobId
            }
            
            print("Monitoring progress...")
            try await monitorProgress(jobId: jobId)
            print("Processing complete, fetching results...")
            let results = try await fetchResults(jobId: jobId)
            
            await MainActor.run {
                self.results = results
                self.isProcessing = false
                self.processingProgress = 1.0
                self.processingMessage = "Processing completed successfully"
            }
            
        } catch {
            print("Error during video processing: \(error)")
            await MainActor.run {
                self.error = error.localizedDescription
                self.isProcessing = false
                self.processingProgress = 0.0
            }
        }
    }
    
    private func uploadVideo(url: URL) async throws -> String {
        let uploadURL = URL(string: "\(baseURL)/upload")!
        var request = URLRequest(url: uploadURL)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        let videoData = try Data(contentsOf: url)
        var body = Data()
        
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"video\"; filename=\"\(url.lastPathComponent)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: video/mp4\r\n\r\n".data(using: .utf8)!)
        body.append(videoData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        let (data, _) = try await URLSession.shared.data(for: request)
        let response = try JSONDecoder().decode(UploadResponse.self, from: data)
        
        return response.jobId
    }
    
    private func monitorProgress(jobId: String) async throws {
        repeat {
            let statusURL = URL(string: "\(baseURL)/status/\(jobId)")!
            let (data, _) = try await URLSession.shared.data(from: statusURL)
            let statusResponse = try JSONDecoder().decode(StatusResponse.self, from: data)
            
            await MainActor.run {
                self.processingProgress = statusResponse.progress
                self.processingMessage = statusResponse.message
                self.framesProcessed = statusResponse.framesProcessed
                self.totalFrames = statusResponse.totalFrames
                self.detectionsCount = statusResponse.detectionsCount
                self.uniqueTracks = statusResponse.uniqueTracks
            }
            
            if statusResponse.status == "completed" {
                break
            } else if statusResponse.status == "failed" {
                throw NSError(domain: "ProcessingError", code: 0, userInfo: [NSLocalizedDescriptionKey: statusResponse.error ?? "Processing failed"])
            }
            
            try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
        } while true
    }
    
    private func fetchResults(jobId: String) async throws -> DetectionResults {
        let resultsURL = URL(string: "\(baseURL)/results/\(jobId)")!
        let (data, _) = try await URLSession.shared.data(from: resultsURL)
        return try JSONDecoder().decode(DetectionResults.self, from: data)
    }
    
    func downloadFile(type: String) async throws -> URL {
        guard let jobId = currentJobId else {
            throw NSError(domain: "APIError", code: 0, userInfo: [NSLocalizedDescriptionKey: "No active job"])
        }
        
        let downloadURL = URL(string: "\(baseURL)/download/\(jobId)/\(type)")!
        let (data, _) = try await URLSession.shared.data(from: downloadURL)
        
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("download_\(type)_\(jobId)")
        try data.write(to: tempURL)
        
        return tempURL
    }
}
