import SwiftUI

struct RootView: View {
    @StateObject private var logManager = MeasurementLogManager()
    @State private var showWoundDetector = false
    @State private var showWoundMeasurement = false
    @State private var showLogs = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                Text("Woundcorder")
                    .font(.largeTitle).bold()

                Button("Wound Analysis") { showWoundDetector = true }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)

                Button("Wound Measurement") { showWoundMeasurement = true }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                
                Button("View Logs") { showLogs = true }
                    .buttonStyle(.bordered)

                Spacer()
            }
            .padding()
            .navigationTitle("Home")
        }
        .fullScreenCover(isPresented: $showWoundDetector) {
            ARCameraTabView()
                .environmentObject(logManager)
                .toolbar(.hidden, for: .tabBar)
                .statusBarHidden(true)
        }
        .fullScreenCover(isPresented: $showWoundMeasurement) {
            ContentView()
                .toolbar(.hidden, for: .tabBar)
                .statusBarHidden(true)
        }
        .sheet(isPresented: $showLogs) {
            MeasurementLogListView()
                .environmentObject(logManager)
        }
    }
}

#Preview {
    RootView()
}
