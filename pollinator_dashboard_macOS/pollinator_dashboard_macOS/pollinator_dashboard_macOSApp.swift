//
//  pollinator_dashboard_macOSApp.swift
//  pollinator_dashboard_macOS
//
//  Created by James Edmond on 8/29/25.
//

import SwiftUI

@main
struct InsectDetectionApp: App {
    @StateObject private var api = InsectDetectionAPI()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(api)
        }
        .windowResizability(.contentSize)
        .windowStyle(.titleBar)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open Video...") {
                    openVideoFile()
                }
                .keyboardShortcut("o")
            }
            
            CommandGroup(after: .importExport) {
                if api.results != nil {
                    Divider()
                    
                    Menu("Export Results") {
                        Button("Export Annotated Video...") {
                            exportFile(type: "video")
                        }
                        
                        Button("Export Detection Data...") {
                            exportFile(type: "json")
                        }
                        
                        Button("Export Insect Crops...") {
                            exportFile(type: "crops")
                        }
                    }
                }
            }
        }
    }
    
    private func openVideoFile() {
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
    
    private func exportFile(type: String) {
        Task {
            do {
                let tempURL = try await api.downloadFile(type: type)
                
                await MainActor.run {
                    let panel = NSSavePanel()
                    panel.title = "Export \(type.capitalized)"
                    panel.nameFieldStringValue = "insect_detection_\(type)"
                    
                    if panel.runModal() == .OK, let saveURL = panel.url {
                        do {
                            _ = try FileManager.default.replaceItem(at: saveURL, withItemAt: tempURL, backupItemName: nil, options: [], resultingItemURL: nil)
                        } catch {
                            print("Export failed: \(error)")
                        }
                    }
                }
            } catch {
                await MainActor.run {
                    api.error = error.localizedDescription
                }
            }
        }
    }
}
