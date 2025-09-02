//
//  ContentView.swift
//  pollinator_dashboard_macOS
//
//  Created by James Edmond on 8/29/25.
//
import SwiftUI

// MARK: - Main Content View
struct ContentView: View {
    @EnvironmentObject var api: InsectDetectionAPI
    @State private var selectedTrack: TrackInfo?
    
    var body: some View {
        HSplitView {
            // Sidebar
            VStack(alignment: .leading, spacing: 0) {
                SidebarHeader()
                
                Divider()
                
                if api.isProcessing {
                    ProcessingSidebar(api: api)
                        .padding()
                } else if let results = api.results {
                    ResultsSidebar(results: results, selectedTrack: $selectedTrack)
                } else {
                    WelcomeSidebar()
                        .padding()
                }
                
                Spacer()
            }
            .frame(minWidth: 250, idealWidth: 300, maxWidth: 400)
            .background(Color(NSColor.controlBackgroundColor))
            
            // Main Content
            VStack(spacing: 0) {
                if api.isProcessing {
                    ProcessingView(api: api)
                } else if let results = api.results {
                    MainResultsView(results: results, selectedTrack: selectedTrack)
                } else {
                    WelcomeView()
                }
            }
            .frame(minWidth: 600, idealWidth: 900, maxWidth: .infinity)
        }
        .frame(minWidth: 850, idealWidth: 1200, maxWidth: .infinity,
               minHeight: 600, idealHeight: 800, maxHeight: .infinity)
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button(action: toggleSidebar) {
                    Image(systemName: "sidebar.left")
                }
            }
            
            if !api.isProcessing {
                ToolbarItem(placement: .automatic) {
                    Button("Open Video") {
                        openVideo()
                    }
                }
            }
        }
    }
    
    private func toggleSidebar() {
        NSApp.keyWindow?.firstResponder?.tryToPerform(#selector(NSSplitViewController.toggleSidebar(_:)), with: nil)
    }
    
    private func openVideo() {
        let panel = NSOpenPanel()
        panel.title = "Select Video File"
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.mpeg4Movie, .quickTimeMovie, .avi]
        if panel.runModal() == .OK, let url = panel.url {
            Task {
                await api.processVideo(url: url)
            }
        }
    }
}

// MARK: - Sidebar Components
struct SidebarHeader: View {
    var body: some View {
        VStack {
            Image(systemName: "ladybug.fill")
                .font(.system(size: 30))
                .foregroundColor(.green)
                .padding(.top)
            
            Text("Insect Detection")
                .font(.headline)
                .fontWeight(.semibold)
                .padding(.bottom)
        }
        .frame(maxWidth: .infinity)
        .background(Color(NSColor.controlBackgroundColor))
    }
}

struct WelcomeSidebar: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Getting Started")
                .font(.title2)
                .fontWeight(.semibold)
            
            Text("Select a video file to analyze for flying insects.")
                .font(.body)
                .foregroundColor(.secondary)
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Supported formats:")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                HStack {
                    ForEach(["MP4", "MOV", "AVI", "MKV"], id: \.self) { format in
                        Text(format)
                            .font(.caption)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.accentColor.opacity(0.1))
                            .cornerRadius(4)
                    }
                }
            }
        }
    }
}

struct ProcessingSidebar: View {
    @ObservedObject var api: InsectDetectionAPI
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Processing Video")
                .font(.title2)
                .fontWeight(.semibold)
            
            VStack(alignment: .leading, spacing: 8) {
                ProgressView(value: api.processingProgress)
                    .progressViewStyle(LinearProgressViewStyle(tint: .accentColor))
                
                Text("\(Int(api.processingProgress * 100))% Complete")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                if !api.processingMessage.isEmpty {
                    Text(api.processingMessage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(3)
                }
            }
            
            // Live statistics
            if api.totalFrames > 0 {
                Divider()
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Progress")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    HStack {
                        Text("Frames:")
                        Spacer()
                        Text("\(api.framesProcessed) / \(api.totalFrames)")
                            .fontWeight(.medium)
                    }
                    .font(.caption)
                    
                    HStack {
                        Text("Detections:")
                        Spacer()
                        Text("\(api.detectionsCount)")
                            .fontWeight(.medium)
                            .foregroundColor(.green)
                    }
                    .font(.caption)
                    
                    HStack {
                        Text("Tracks:")
                        Spacer()
                        Text("\(api.uniqueTracks)")
                            .fontWeight(.medium)
                            .foregroundColor(.blue)
                    }
                    .font(.caption)
                }
            }
        }
    }
}

struct ResultsSidebar: View {
    let results: DetectionResults
    @Binding var selectedTrack: TrackInfo?
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Summary
            VStack(alignment: .leading, spacing: 12) {
                Text("Summary")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .padding(.horizontal)
                    .padding(.top)
                
                LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 8) {
                    SummaryCard(title: "Detections", value: "\(results.metadata.totalDetections)")
                    SummaryCard(title: "Tracks", value: "\(results.metadata.uniqueTracks)")
                    SummaryCard(title: "Frames", value: "\(results.metadata.processedFrames)")
                    SummaryCard(title: "Hit Rate", value: "\(String(format: "%.1f%%", Double(results.summary.framesWithDetections) / Double(results.metadata.processedFrames) * 100))")
                }
                .padding(.horizontal)
            }
            
            Divider()
                .padding(.vertical, 8)
            
            // Track List
            VStack(alignment: .leading, spacing: 8) {
                Text("Tracks (\(results.summary.tracksInfo.count))")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .padding(.horizontal)
                
                ScrollView {
                    LazyVStack(spacing: 4) {
                        ForEach(results.summary.tracksInfo.sorted { $0.detectionCount > $1.detectionCount }) { track in
                            TrackListItem(track: track, isSelected: selectedTrack?.trackId == track.trackId) {
                                selectedTrack = track
                            }
                        }
                    }
                    .padding(.horizontal, 8)
                }
            }
        }
    }
}

// MARK: - Main View Components
struct WelcomeView: View {
    @EnvironmentObject var api: InsectDetectionAPI
    
    var body: some View {
        VStack(spacing: 30) {
            Image(systemName: "video.badge.plus")
                .font(.system(size: 80))
                .foregroundColor(.accentColor.opacity(0.7))
            
            VStack(spacing: 12) {
                Text("Welcome to Insect Detection")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                Text("Analyze videos to detect and track flying insects with advanced computer vision.")
                    .font(.title3)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            
            if let error = api.error {
                Text("Error: \(error)")
                    .foregroundColor(.red)
                    .padding()
                    .background(Color.red.opacity(0.1))
                    .cornerRadius(8)
            }
            
            Button("Open Video File") {
                openVideo()
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(NSColor.controlBackgroundColor).opacity(0.3))
    }
    
    private func openVideo() {
        let panel = NSOpenPanel()
        panel.title = "Select Video File"
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.mpeg4Movie, .quickTimeMovie]
        
        if panel.runModal() == .OK, let url = panel.url {
            Task {
                await api.processVideo(url: url)
            }
        }
    }
}

struct ProcessingView: View {
    @ObservedObject var api: InsectDetectionAPI
    
    var body: some View {
        VStack(spacing: 40) {
            ProgressView(value: api.processingProgress)
                .progressViewStyle(CircularProgressViewStyle(tint: .accentColor))
                .scaleEffect(2.0)
            
            VStack(spacing: 16) {
                Text("Analyzing Video")
                    .font(.title)
                    .fontWeight(.semibold)
                
                Text(api.processingMessage.isEmpty ? "Processing your video..." : api.processingMessage)
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                
                Text("\(Int(api.processingProgress * 100))% Complete")
                    .font(.headline)
                    .fontWeight(.medium)
                    .foregroundColor(.accentColor)
                
                // Live processing statistics
                if api.totalFrames > 0 {
                    VStack(spacing: 12) {
                        Divider()
                            .padding(.horizontal, 40)
                        
                        HStack(spacing: 30) {
                            VStack {
                                Text("\(api.framesProcessed)")
                                    .font(.title2)
                                    .fontWeight(.bold)
                                    .foregroundColor(.primary)
                                Text("of \(api.totalFrames)")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text("Frames")
                                    .font(.caption)
                                    .fontWeight(.medium)
                            }
                            
                            VStack {
                                Text("\(api.detectionsCount)")
                                    .font(.title2)
                                    .fontWeight(.bold)
                                    .foregroundColor(.green)
                                Text("Detections")
                                    .font(.caption)
                                    .fontWeight(.medium)
                            }
                            
                            VStack {
                                Text("\(api.uniqueTracks)")
                                    .font(.title2)
                                    .fontWeight(.bold)
                                    .foregroundColor(.blue)
                                Text("Unique Tracks")
                                    .font(.caption)
                                    .fontWeight(.medium)
                            }
                        }
                        .padding(.top, 8)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(NSColor.controlBackgroundColor).opacity(0.3))
    }
}

struct MainResultsView: View {
    let results: DetectionResults
    let selectedTrack: TrackInfo?
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Detection Results")
                    .font(.title)
                    .fontWeight(.bold)
                
                Spacer()
                
                Menu("Export") {
                    Button("Annotated Video...") {
                        // Export video
                    }
                    Button("Detection Data (JSON)...") {
                        // Export JSON
                    }
                    Button("Insect Crops (ZIP)...") {
                        // Export crops
                    }
                }
                .menuStyle(.borderlessButton)
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            // Content
            if let track = selectedTrack {
                TrackDetailView(track: track)
            } else {
                OverviewView(results: results)
            }
        }
    }
}

// MARK: - Detail Components
struct SummaryCard: View {
    let title: String
    let value: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.primary)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(8)
        .background(Color(NSColor.quaternaryLabelColor))
        .cornerRadius(6)
    }
}

struct TrackListItem: View {
    let track: TrackInfo
    let isSelected: Bool
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text("Track \(track.trackId)")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        
                        Spacer()
                        
                        Text("\(track.detectionCount)")
                            .font(.caption)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1)
                            .background(Color.accentColor.opacity(0.2))
                            .cornerRadius(3)
                    }
                    
                    Text("Frames \(track.firstFrame)-\(track.lastFrame)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text("Conf: \(String(format: "%.2f", track.avgConfidence))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(isSelected ? Color.accentColor.opacity(0.2) : Color.clear)
            .cornerRadius(6)
        }
        .buttonStyle(.plain)
    }
}

struct TrackDetailView: View {
    let track: TrackInfo
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Track \(track.trackId) Details")
                        .font(.title2)
                        .fontWeight(.semibold)
                    
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: 16) {
                        DetailCard(title: "Detections", value: "\(track.detectionCount)")
                        DetailCard(title: "Duration", value: "\(track.lastFrame - track.firstFrame) frames")
                        DetailCard(title: "Avg Confidence", value: String(format: "%.3f", track.avgConfidence))
                        DetailCard(title: "Max Velocity", value: String(format: "%.1f px/f", track.maxVelocity))
                        DetailCard(title: "First Frame", value: "\(track.firstFrame)")
                        DetailCard(title: "Last Frame", value: "\(track.lastFrame)")
                    }
                }
                
                VStack(alignment: .leading, spacing: 12) {
                    Text("Flight Path")
                        .font(.title3)
                        .fontWeight(.semibold)
                    
                    FlightPathView(path: track.path)
                        .frame(height: 300)
                        .background(Color(NSColor.controlBackgroundColor))
                        .cornerRadius(8)
                }
            }
            .padding()
        }
    }
}

struct DetailCard: View {
    let title: String
    let value: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
                .textCase(.uppercase)
                .tracking(0.5)
            
            Text(value)
                .font(.title2)
                .fontWeight(.semibold)
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
    }
}

struct FlightPathView: View {
    let path: [Point]
    
    var body: some View {
        GeometryReader { geometry in
            if !path.isEmpty {
                let minX = path.map(\.x).min() ?? 0
                let maxX = path.map(\.x).max() ?? 1
                let minY = path.map(\.y).min() ?? 0
                let maxY = path.map(\.y).max() ?? 1
                
                let scaleX = geometry.size.width / (maxX - minX)
                let scaleY = geometry.size.height / (maxY - minY)
                
                ZStack {
                    // Path line
                    Path { path in
                        if let first = self.path.first {
                            let startX = (first.x - minX) * scaleX
                            let startY = geometry.size.height - ((first.y - minY) * scaleY)
                            path.move(to: CGPoint(x: startX, y: startY))
                            
                            for point in self.path.dropFirst() {
                                let x = (point.x - minX) * scaleX
                                let y = geometry.size.height - ((point.y - minY) * scaleY)
                                path.addLine(to: CGPoint(x: x, y: y))
                            }
                        }
                    }
                    .stroke(Color.accentColor, lineWidth: 2)
                    
                    // Points
                    ForEach(Array(path.enumerated()), id: \.offset) { index, point in
                        let x = (point.x - minX) * scaleX
                        let y = geometry.size.height - ((point.y - minY) * scaleY)
                        
                        Circle()
                            .fill(index == 0 ? Color.green : (index == path.count - 1 ? Color.red : Color.accentColor))
                            .frame(width: 6, height: 6)
                            .position(x: x, y: y)
                    }
                }
            } else {
                Text("No path data available")
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }
}

struct OverviewView: View {
    let results: DetectionResults
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                Text("Analysis Overview")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Text("Select a track from the sidebar to view detailed information about individual insect trajectories.")
                    .font(.body)
                    .foregroundColor(.secondary)
                
                LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 16) {
                    OverviewCard(
                        title: "Total Detections",
                        value: "\(results.metadata.totalDetections)",
                        description: "Individual insect detections across all frames"
                    )
                    
                    OverviewCard(
                        title: "Unique Tracks",
                        value: "\(results.metadata.uniqueTracks)",
                        description: "Distinct insects identified and tracked"
                    )
                    
                    OverviewCard(
                        title: "Processing Coverage",
                        value: "\(String(format: "%.1f%%", Double(results.summary.framesWithDetections) / Double(results.metadata.processedFrames) * 100))",
                        description: "Percentage of frames containing detected insects"
                    )
                    
                    OverviewCard(
                        title: "Frames Analyzed",
                        value: "\(results.metadata.processedFrames)",
                        description: "Total video frames processed"
                    )
                }
            }
            .padding()
        }
    }
}

struct OverviewCard: View {
    let title: String
    let value: String
    let description: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                    .fontWeight(.semibold)
                
                Text(value)
                    .font(.title)
                    .fontWeight(.bold)
                    .foregroundColor(.accentColor)
            }
            
            Text(description)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(2)
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(12)
    }
}
