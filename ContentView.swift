import SwiftUI

struct ContentView: View {
    @StateObject private var metricsManager = MetricsManager()
    @State private var userPoints: [SIMD3<Float>] = []
    @State private var selectedDepthPoint: SIMD3<Float>?
    @Environment(\.dismiss) private var dismiss
    @AppStorage("showLegacyControls") private var showLegacyControls: Bool = false
    @State private var isDetecting: Bool = false
    @State private var showAnalysis = false

    var body: some View {
        Group {
            if #available(iOS 16.0, *) {
                NavigationStack {
                    content
                        .navigationTitle("Wound Measurement")
                        .toolbar(.hidden, for: .tabBar)
                        .toolbar(.hidden, for: .navigationBar)
                        .toolbar {
                            ToolbarItem(placement: .cancellationAction) {
                                Button("Close") { dismiss() }
                            }
                        }
                }
            } else {
                NavigationView {
                    content
                        .navigationTitle("Wound Measurement")
                        .toolbar {
                            ToolbarItem(placement: .cancellationAction) {
                                Button("Close") { dismiss() }
                            }
                        }
                }
                .navigationViewStyle(StackNavigationViewStyle())
            }

               Button {
                   dismiss()
               } label: {
                   Image(systemName: "xmark")
                       .font(.headline)
                       .padding(10)
                       .background(.ultraThinMaterial, in: Circle())
               }
               .padding(.trailing, 16)
        }
        .statusBarHidden(true)
        .fullScreenCover(isPresented: $showAnalysis) {
            ARCameraTabView()
                .statusBarHidden(true)
        }
    }

    @ViewBuilder
    private var content: some View {
        ZStack {
            if #available(iOS 17, *) {
                ARViewContainer(
                    userPoints: $userPoints,
                    metricsManager: metricsManager,
                    selectedDepthPoint: $selectedDepthPoint,
                    onMeasurementState: { state in
                        // Be defensive about enum cases defined in ARViewController.MeasurementState
                        let description = String(describing: state).lowercased()
                        if description.contains("detect") {
                            isDetecting = true
                        } else if description.contains("complete") {
                            isDetecting = false
                        } else if description.contains("idle") || description.contains("ready") || description.contains("reset") {
                            isDetecting = false
                        } else {
                            // Fallback for unknown states
                            isDetecting = false
                        }
                    }
                )
                .onChange(of: userPoints, initial: false) { oldValue, newValue in
                    if newValue.count >= 3 {
                        isDetecting = false
                    }
                }
                .ignoresSafeArea()
                .onAppear {
                    // Always request overlays hidden when the AR view appears
                    NotificationCenter.default.post(name: NSNotification.Name("ARHideOverlays"), object: nil)
                }
                .onReceive(NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)) { _ in
                    NotificationCenter.default.post(name: NSNotification.Name("ARHideOverlays"), object: nil)
                }
            } else {
                ARViewContainer(
                    userPoints: $userPoints,
                    metricsManager: metricsManager,
                    selectedDepthPoint: $selectedDepthPoint,
                    onMeasurementState: { state in
                        // Be defensive about enum cases defined in ARViewController.MeasurementState
                        let description = String(describing: state).lowercased()
                        if description.contains("detect") {
                            isDetecting = true
                        } else if description.contains("complete") {
                            isDetecting = false
                        } else if description.contains("idle") || description.contains("ready") || description.contains("reset") {
                            isDetecting = false
                        } else {
                            // Fallback for unknown states
                            isDetecting = false
                        }
                    }
                )
                .onChange(of: userPoints) {
                    if userPoints.count >= 3 {
                        isDetecting = false
                    }
                }
                .ignoresSafeArea()
                .onAppear {
                    // Always request overlays hidden when the AR view appears
                    NotificationCenter.default.post(name: NSNotification.Name("ARHideOverlays"), object: nil)
                }
                .onReceive(NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)) { _ in
                    NotificationCenter.default.post(name: NSNotification.Name("ARHideOverlays"), object: nil)
                }
            }

            VStack {
                Spacer()

                MetricsView(metrics: metricsManager.metrics)
                    .padding()

                HStack {
                    Button {
                        showAnalysis = true
                    } label: {
                        Label("Wound Analysis", systemImage: "scope")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.accentColor)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }

                    Button {
                        NotificationCenter.default.post(name: NSNotification.Name("StartGuidedWoundMeasurement"), object: nil)
                    } label: {
                        Label("Wound Measurement", systemImage: "ruler")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.accentColor.opacity(0.8))
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                }
                .padding([.leading, .trailing, .bottom], 20)

                HStack(spacing: 12) {
                    Button {
                        NotificationCenter.default.post(name: NSNotification.Name("StartManualOutline"), object: nil)
                    } label: {
                        Label("Start Outline", systemImage: "pencil.tip")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.orange)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }

                    Button {
                        NotificationCenter.default.post(name: NSNotification.Name("FinishManualOutline"), object: nil)
                    } label: {
                        Label("Finish", systemImage: "checkmark.circle")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }

                    Button {
                        NotificationCenter.default.post(name: NSNotification.Name("ClearManualOutline"), object: nil)
                    } label: {
                        Label("Clear", systemImage: "trash")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.red)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                }
                .padding([.leading, .trailing], 20)

                if showLegacyControls {
                    HStack {
                        Button(action: resetScene) {
                            Text("Reset")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.red)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }

                        Button(action: selectDepthPoint) {
                            Text("Select Depth Point")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.green)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }

                        Button(action: exportScene) {
                            Text("Export")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                    }
                    .padding([.leading, .trailing, .bottom], 20)
                }
            }
        }

        
    }

    private func resetScene() {
        userPoints.removeAll()
        selectedDepthPoint = nil
        metricsManager.resetMetrics()
        isDetecting = false
        NotificationCenter.default.post(name: NSNotification.Name("ResetARScene"), object: nil)
    }

    private func selectDepthPoint() {
        NotificationCenter.default.post(name: NSNotification.Name("SelectDepthPoint"), object: nil)
    }

    private func exportScene() {
        NotificationCenter.default.post(name: NSNotification.Name("ExportARScene"), object: nil)
    }
}


struct MetricsView: View {
    let metrics: (perimeter: Float, area: Float, volume: Float)

    var body: some View {
        VStack(alignment: .leading) {
            Text("Wound Measurements")
                .font(.headline)
                .padding(.bottom, 4)
            Text("Perimeter: \(metrics.perimeter, specifier: "%.2f") cm")
            Text("Area: \(metrics.area, specifier: "%.2f") cm²")
            Text("Volume: \(metrics.volume, specifier: "%.2f") cm³")
        }
        .padding()
        .background(Color.black.opacity(0.6))
        .cornerRadius(10)
        .foregroundColor(.white)
    }
}

struct ControlButtons: View {
    let onReset: () -> Void
    let onExport: () -> Void

    var body: some View {
        HStack {
            Button(action: onReset) {
                Text("Reset")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.red)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }

            Button(action: onExport) {
                Text("Export")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
        }
    }
}

final class MetricsManager: ObservableObject {
    @Published var perimeter: Float = 0
    @Published var area: Float = 0
    @Published var volume: Float = 0

    var metrics: (perimeter: Float, area: Float, volume: Float) {
        (perimeter: perimeter, area: area, volume: volume)
    }

    func updateMetrics(perimeter: Float, area: Float, volume: Float) {
        self.perimeter = perimeter
        self.area = area
        self.volume = volume
    }

    func resetMetrics() {
        perimeter = 0
        area = 0
        volume = 0
    }
}

