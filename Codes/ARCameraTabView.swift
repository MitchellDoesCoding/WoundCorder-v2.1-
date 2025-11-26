import SwiftUI
import ARKit
import SceneKit
import Photos
import PhotosUI
import Vision
import CoreImage

struct ARCameraTabView: View {
    @EnvironmentObject private var logManager: MeasurementLogManager
    @Environment(\.dismiss) private var dismiss
    @State private var showPicker = false
    @State private var selectedItem: PhotosPickerItem? = nil
    @State private var analysisText: String? = nil
    @State private var pendingImage: UIImage? = nil
    @State private var snapshotRequestID: Int = 0
    @State private var showComorbiditySheet = false
    @State private var selectedComorbidities: ComorbidityChecklist? = nil
    @State private var isAnalyzing = false
    @AppStorage("woundServerBaseURL") private var baseURL: String = ""
    @State private var showSettings = false
    @State private var formattedAnalysis: AttributedString? = nil
    @State private var autoEdgeDetection = false
    @AppStorage("showLegacyControls") private var showLegacyControls: Bool = false

    @State private var latestAnalysisSummary: AttributedString? = nil
    
    @State private var analyzeStartTime: Date? = nil
    @State private var analysisElapsed: TimeInterval = 0
    @State private var analysisTimer: Timer? = nil

    private var detector: WoundDetector {
        #if targetEnvironment(simulator)
        let base = baseURL.isEmpty ? "http://127.0.0.1:3000" : baseURL
        #else
        let base = baseURL.isEmpty ? "https://dae3083997de.ngrok-free.app" : baseURL // ACUTALYL USRl
        #endif
        let endpoint = URL(string: "\(base)/api/analyze-wound")!
        return WoundDetector(endpoint: endpoint, session: .shared)
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            ARCameraControllerHost(snapshotRequestID: $snapshotRequestID, onSnapshot: handleSnapshot)
                .ignoresSafeArea()

            HStack(spacing: 16) {
                if showLegacyControls {
                    Button {
                        showPicker = true
                    } label: {
                        Label("Upload", systemImage: "square.and.arrow.up.on.square")
                            .padding(.horizontal, 14).padding(.vertical, 10)
                            .background(.ultraThinMaterial, in: Capsule())
                    }

                    Button {
                        showComorbiditySheet = true
                    } label: {
                        Label("Shutter", systemImage: "camera.shutter.button")
                            .padding(.horizontal, 14).padding(.vertical, 10)
                            .background(.ultraThinMaterial, in: Capsule())
                    }
                    .sheet(isPresented: $showComorbiditySheet) {
                        ComorbidityChecklistSheet(
                            onUse: { checklist in
                                selectedComorbidities = checklist
                                snapshotRequestID &+= 1 // now trigger the actual capture
                            },
                            onSkip: {
                                selectedComorbidities = nil
                                snapshotRequestID &+= 1 // trigger capture without metadata
                            }
                        )
                    }

                    Button {
                        showSettings = true
                    } label: {
                        Label("Settings", systemImage: "gearshape")
                            .padding(.horizontal, 14).padding(.vertical, 10)
                            .background(.ultraThinMaterial, in: Capsule())
                    }
                    .sheet(isPresented: $showSettings) {
                        SettingsView()
                    }
                }

                Button {
                    NotificationCenter.default.post(name: NSNotification.Name("RunWoundAnalysisNow"), object: nil)
                } label: {
                    Label("Wound Analysis", systemImage: "scope")
                        .padding(.horizontal, 14).padding(.vertical, 10)
                        .background(.ultraThinMaterial, in: Capsule())
                }

                Button {
                    showPicker = true
                } label: {
                    Label("Upload Image", systemImage: "photo.on.rectangle")
                        .padding(.horizontal, 14).padding(.vertical, 10)
                        .background(.ultraThinMaterial, in: Capsule())
                }
            }
            .padding(.bottom, 24)
            
            VStack {
                HStack {
                    Spacer()
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark")
                            .font(.headline)
                            .padding(10)
                            .background(.ultraThinMaterial, in: Circle())
                    }
                    .padding(.top, 16)
                    .padding(.trailing, 16)
                }
                Spacer()
            }
            
            VStack(alignment: .leading, spacing: 8) {
                if let summary = latestAnalysisSummary {
                    Text(summary)
                        .font(.footnote)
                        .padding(10)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                }
                Spacer()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding([.top, .leading], 16)
            .allowsHitTesting(false)
        }
        .overlay(alignment: .center) {
            if isAnalyzing {
                ProgressView("Analyzing… (\(String(format: "%.1f", analysisElapsed))s)")
                    .padding()
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
        }
        .photosPicker(isPresented: $showPicker, selection: $selectedItem)
        .onChange(of: selectedItem) { _, newItem in
            guard let newItem = newItem else { return }
            Task {
                if let data = try? await newItem.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    await analyze(image)
                }
            }
            
        }
        .fullScreenCover(
            item: Binding(
                get: { formattedAnalysis.map { AnalysisResult(content: $0) } },
                set: { _ in
                    analysisText = nil
                    formattedAnalysis = nil
                }
            )
        ) { item in
            AnalysisResultView(content: item.content) { title in
                logManager.add(title: title.isEmpty ? "Analysis" : title, perimeter: 0, area: 0, volume: 0)
            }
        }
    }

    private func handleSnapshot(_ image: UIImage) {
        Task { await analyze(image) }
    }

    private func analyze(_ image: UIImage) async {
        isAnalyzing = true
        self.analyzeStartTime = Date()
        self.analysisElapsed = 0
        self.analysisTimer?.invalidate()
        self.analysisTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            if let start = self.analyzeStartTime {
                self.analysisElapsed = Date().timeIntervalSince(start)
            }
        }
        defer {
            isAnalyzing = false
            self.analysisTimer?.invalidate()
            self.analysisTimer = nil
        }

        do {
            let t0 = CFAbsoluteTimeGetCurrent()
            let reduced = image.scaled(maxDimension: 1024)
            let t1 = CFAbsoluteTimeGetCurrent()
            let metadataDict = selectedComorbidities?.asDictionary()
            let result = try await detector.analyze(image: reduced, metadata: metadataDict)
            let t2 = CFAbsoluteTimeGetCurrent()
            print(String(format: "Timing: scale=%.2fs, network+server=%.2fs, total=%.2fs", t1 - t0, t2 - t1, t2 - t0))
            await MainActor.run {
                let raw = String(describing: result)
                self.analysisText = raw
                self.formattedAnalysis = SummaryFormatter.formatMarkdown(raw)
                self.latestAnalysisSummary = self.formattedAnalysis
            }
        } catch {
            await MainActor.run {
                let message: String
                if let detectorError = error as? WoundDetector.DetectorError {
                    switch detectorError {
                    case .server(let serverMessage):
                        message = "Analysis failed: \(serverMessage)"
                    case .encodingFailed:
                        message = "Analysis failed: Could not encode image."
                    case .badResponse:
                        message = "Analysis failed: Bad response from server."
                    }
                } else if let urlError = error as? URLError {
                    message = "Network error (\(urlError.code.rawValue)): \(urlError.localizedDescription)"
                } else {
                    message = "Analysis failed: \(error.localizedDescription)"
                }
                self.analysisText = message
                self.formattedAnalysis = SummaryFormatter.formatParagraphs(message)
                self.latestAnalysisSummary = self.formattedAnalysis
            }
        }
    }
}

extension UIImage {
    func scaled(maxDimension: CGFloat) -> UIImage {
        let size = self.size
        let maxSide = max(size.width, size.height)
        guard maxSide > maxDimension else { return self }
        let scale = maxDimension / maxSide
        let newSize = CGSize(width: size.width * scale, height: size.height * scale)
        UIGraphicsBeginImageContextWithOptions(newSize, true, 1.0)
        defer { UIGraphicsEndImageContext() }
        self.draw(in: CGRect(origin: .zero, size: newSize))
        return UIGraphicsGetImageFromCurrentImageContext() ?? self
    }
}

private struct AnalysisResult: Identifiable {
    let id = UUID()
    let content: AttributedString
}

private struct AnalysisResultView: View {
    enum CardStyle {
        case none
        case rounded(radius: CGFloat)
    
    }

    let content: AttributedString
    let onSave: (String) -> Void
    var cardStyle: CardStyle = .rounded(radius: 16) // default

    @Environment(\.dismiss) private var dismiss
    @State private var title: String = ""

    var body: some View {
        NavigationView {
            ScrollView {
                let textView = Text(content)
                    .font(.title3)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()

                switch cardStyle {
                case .none:
                    textView
                        .padding(.horizontal)
                        .padding(.top)
                case .rounded(let radius):
                    textView
                        .frame(maxWidth: 700)
                        .padding(.horizontal)
                        .padding(.top)
                        .background(Color(.secondarySystemBackground))
                        .clipShape(RoundedRectangle(cornerRadius: radius, style: .continuous))
                    
                }
            }
            .navigationTitle("Analysis")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
                ToolbarItem(placement: .principal) {
                    TextField("Title", text: $title)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 240)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save to Log") {
                        onSave(title)
                        dismiss()
                    }
                }
            }
            
        }
    }
}

private struct ARCameraControllerHost: UIViewControllerRepresentable {
    @Binding var snapshotRequestID: Int
    let onSnapshot: (UIImage) -> Void

    func makeUIViewController(context: Context) -> ARCameraViewController {
        let vc = ARCameraViewController()
        vc.onSnapshot = onSnapshot
        return vc
    }

    func updateUIViewController(_ uiViewController: ARCameraViewController, context: Context) {
        // When snapshotRequestID changes, trigger a snapshot.
        if context.coordinator.lastHandledRequestID != snapshotRequestID {
            context.coordinator.lastHandledRequestID = snapshotRequestID
            uiViewController.takeSnapshot()
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var lastHandledRequestID: Int = 0
    }
}

final class ARCameraViewController: UIViewController, ARSCNViewDelegate {
    private let sceneView = ARSCNView()
    var onSnapshot: ((UIImage) -> Void)?

    private var isSimpleMode: Bool = false

    override func viewDidLoad() {
        super.viewDidLoad()
        setupScene()
        
        NotificationCenter.default.addObserver(self, selector: #selector(hideOverlays), name: NSNotification.Name("ARHideOverlays"), object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(showOverlays), name: NSNotification.Name("ARShowOverlays"), object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(enableSimpleMode), name: NSNotification.Name("ARSimpleModeEnabled"), object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(disableSimpleMode), name: NSNotification.Name("ARSimpleModeDisabled"), object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(simpleDetectNow), name: NSNotification.Name("ARSimpleDetectNow"), object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(simpleDetectNow), name: NSNotification.Name("RunWoundAnalysisNow"), object: nil)
        hideOverlays()
        
        sceneView.showsStatistics = false
        sceneView.overlaySKScene = nil
        sceneView.automaticallyUpdatesLighting = false
        sceneView.scene.background.contents = nil
        sceneView.technique = nil

        // Disable user interaction and any gesture-based measurement
        sceneView.gestureRecognizers?.forEach { sceneView.removeGestureRecognizer($0) }
        if let recognizers = sceneView.gestureRecognizers, !recognizers.isEmpty {
            print("Unexpected gesture recognizers still present in viewDidLoad: ", recognizers)
        } else {
            print("No gesture recognizers on sceneView in viewDidLoad.")
        }
        sceneView.isUserInteractionEnabled = false
        // Clear any camera filters/post-processing that could draw outlines
        sceneView.pointOfView?.camera?.wantsHDR = false
        sceneView.pointOfView?.camera?.wantsExposureAdaptation = false
        sceneView.pointOfView?.camera?.motionBlurIntensity = 0
        sceneView.pointOfView?.camera?.screenSpaceAmbientOcclusionIntensity = 0
        sceneView.pointOfView?.camera?.colorFringeIntensity = 0
        sceneView.pointOfView?.camera?.bloomIntensity = 0
        sceneView.pointOfView?.camera?.vignettingIntensity = 0
        sceneView.pointOfView?.camera?.grainIntensity = 0
        sceneView.pointOfView?.camera?.contrast = 1.0
        sceneView.pointOfView?.camera?.saturation = 1.0
        sceneView.pointOfView?.filters = nil
        sceneView.antialiasingMode = .none
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        startSession()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sceneView.scene.rootNode.childNodes.forEach { $0.removeFromParentNode() }
        sceneView.debugOptions = []
    }

    private func setupScene() {
        view.addSubview(sceneView)
        sceneView.frame = view.bounds
        sceneView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        sceneView.delegate = self
        sceneView.scene = SCNScene()
        sceneView.debugOptions = []
    }

    private func startSession() {
        let config = ARWorldTrackingConfiguration()
        if isSimpleMode {
            // Keep it minimal in simple mode
            config.planeDetection = []
            if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) || ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification) {
                config.sceneReconstruction = []
            }
            config.environmentTexturing = .none
        } else {
            // Non-simple: still keep visuals minimal; you can re-enable as needed
            config.planeDetection = []
            if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) || ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification) {
                config.sceneReconstruction = []
            }
            config.environmentTexturing = .none
        }
        sceneView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        // Ensure no residual nodes/overlays
        sceneView.debugOptions = []
        sceneView.scene.rootNode.childNodes.forEach { $0.removeFromParentNode() }
        sceneView.technique = nil
        sceneView.pointOfView?.filters = nil
        sceneView.gestureRecognizers?.forEach { sceneView.removeGestureRecognizer($0) }
        if let recognizers = sceneView.gestureRecognizers, !recognizers.isEmpty {
            print("Unexpected gesture recognizers still present in startSession: ", recognizers)
        } else {
            print("No gesture recognizers on sceneView in startSession.")
        }
        sceneView.isUserInteractionEnabled = false
    }

    private func setupSnapshotButton() {
        let button: UIButton
        if #available(iOS 15.0, *) {
            var config = UIButton.Configuration.filled()
            config.baseBackgroundColor = UIColor.black.withAlphaComponent(0.3)
            config.baseForegroundColor = .white
            config.cornerStyle = .capsule
            config.image = UIImage(systemName: "circle.fill")
            config.contentInsets = NSDirectionalEdgeInsets(top: 12, leading: 12, bottom: 12, trailing: 12)

            button = UIButton(configuration: config, primaryAction: nil)
        } else {
            button = UIButton(type: .system)
            button.setImage(UIImage(systemName: "circle.fill"), for: .normal)
            button.tintColor = .white
            button.backgroundColor = UIColor.black.withAlphaComponent(0.3)
            button.layer.cornerRadius = 28
            button.contentEdgeInsets = UIEdgeInsets(top: 12, left: 12, bottom: 12, right: 12)
        }
        button.translatesAutoresizingMaskIntoConstraints = false
        button.addTarget(self, action: #selector(takeSnapshot), for: .touchUpInside)

        view.addSubview(button)
        NSLayoutConstraint.activate([
            button.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            button.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -24)
        ])
    }

    @objc func takeSnapshot() {
        let image = sceneView.snapshot()
        self.onSnapshot?(image)
        PHPhotoLibrary.requestAuthorization { status in
            guard status == .authorized || status == .limited else { return }
            UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
        }
    }

    @objc private func hideOverlays() {
        // Remove any existing visualization nodes and disable debug visuals
        sceneView.debugOptions = []
        sceneView.scene.rootNode.childNodes.forEach { $0.removeFromParentNode() }
        sceneView.technique = nil
        sceneView.pointOfView?.filters = nil
    }

    @objc private func showOverlays() {
        // Optional: enable minimal debug options if desired, leaving it empty to keep clean UI
        sceneView.debugOptions = []
    }

    @objc private func enableSimpleMode() {
        isSimpleMode = true
        startSession()
        hideOverlays()
    }
    @objc private func disableSimpleMode() {
        isSimpleMode = false
        startSession()
    }

    @objc private func simpleDetectNow() {
        self.takeSnapshot()
    }
}

