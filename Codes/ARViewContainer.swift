import SwiftUI
import ARKit
import SceneKit
import Foundation
import ZIPFoundation
import Vision
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreML

// Placeholder measurement state typealias to keep API flexible without coupling to AR-specific enums.
public typealias AnyMeasurementState = Any

/// A top-level SwiftUI placeholder AR container so other files (e.g., ContentView) can reference it as `ARViewContainer`.
/// Replace the body implementation with a real ARKit/RealityKit integration when available.
struct ARViewContainer: View {
    @Binding var userPoints: [SIMD3<Float>]
    let metricsManager: MetricsManager
    @Binding var selectedDepthPoint: SIMD3<Float>?

    // Use a loosely-typed closure so ContentView can pass whatever state enum it owns
    // (e.g., ARViewController.MeasurementState) without this file needing that type.
    let onMeasurementState: (AnyMeasurementState) -> Void

    init(
        userPoints: Binding<[SIMD3<Float>]>,
        metricsManager: MetricsManager,
        selectedDepthPoint: Binding<SIMD3<Float>?>,
        onMeasurementState: @escaping (AnyMeasurementState) -> Void
    ) {
        self._userPoints = userPoints
        self.metricsManager = metricsManager
        self._selectedDepthPoint = selectedDepthPoint
        self.onMeasurementState = onMeasurementState
    }

    var body: some View {
        ARContainerRepresentable(
            userPoints: $userPoints,
            metricsManager: metricsManager,
            selectedDepthPoint: $selectedDepthPoint,
            onMeasurementState: onMeasurementState
        )
        .ignoresSafeArea()
    }
}

private struct ARContainerRepresentable: UIViewControllerRepresentable {
    @Binding var userPoints: [SIMD3<Float>]
    let metricsManager: MetricsManager
    @Binding var selectedDepthPoint: SIMD3<Float>?
    let onMeasurementState: (AnyMeasurementState) -> Void

    func makeUIViewController(context: Context) -> ARViewController {
        let vc = ARViewController()
        // Bind the SwiftUI points array to the controller
        vc.userPoints = Binding(get: { self.userPoints }, set: { self.userPoints = $0 })
        // Metrics callback to update the manager
        vc.onMetricsCalculated = { perimeter, area, volume in
            DispatchQueue.main.async {
                self.metricsManager.updateMetrics(perimeter: perimeter, area: area, volume: volume)
            }
        }
        // Forward measurement state to SwiftUI as loosely-typed AnyMeasurementState
        vc.onMeasurementState = { state in
            self.onMeasurementState(state)
        }
        // Initialize depth point
        vc.selectedDepthPoint = selectedDepthPoint
        return vc
    }

    func updateUIViewController(_ uiViewController: ARViewController, context: Context) {
        // Keep binding and callbacks current in case dependencies change
        uiViewController.userPoints = Binding(get: { self.userPoints }, set: { self.userPoints = $0 })
        uiViewController.onMetricsCalculated = { perimeter, area, volume in
            DispatchQueue.main.async {
                self.metricsManager.updateMetrics(perimeter: perimeter, area: area, volume: volume)
            }
        }
        uiViewController.onMeasurementState = { state in
            self.onMeasurementState(state)
        }
        // Push latest selected depth point to the controller
        uiViewController.selectedDepthPoint = selectedDepthPoint
    }
}

#if DEBUG
#Preview {
    ARViewContainer(
        userPoints: .constant([]),
        metricsManager: MetricsManager(),
        selectedDepthPoint: .constant(nil),
        onMeasurementState: { _ in }
    )
}
#endif

// Disambiguate LiDARVolumeCalculator across possible modules
// Prefer WoundCorder when the compile flag USE_WOUNDCORDER_VOLUME is set.
#if canImport(WoundCorder)
#if USE_WOUNDCORDER_VOLUME
import WoundCorder
private typealias VolumeCalculatorType = WoundCorder.LiDARVolumeCalculator
#else
// If the flag is not set, fall back to the local project LiDARVolumeCalculator
private typealias VolumeCalculatorType = LiDARVolumeCalculator
#endif
#else
// If WoundCorder is not available, use the local project LiDARVolumeCalculator
private typealias VolumeCalculatorType = LiDARVolumeCalculator
#endif

final class ARViewController: UIViewController, ARSCNViewDelegate, ARSessionDelegate {
    private let sceneView = ARSCNView()
    // Startup timestamp for grace period
    private let meshBuildStartTime: CFTimeInterval = CACurrentMediaTime()

    #if DEBUG
    private func debugLog(_ msg: String) {
        print("[MeshDebug] \(msg)")
    }
    #endif
    
    // Single declaration:
    private var isSelectingDepthPoint: Bool = false
    private var isManualOutlineMode: Bool = false
    
    // Guided outline flow controls
    private var guidedActive: Bool = false
    private var guidedMaxPoints: Int = 40
    private var guidedMinPointsToAuto: Int = 8
    private var guidedInactivitySeconds: TimeInterval = 2.0
    private var guidedAutoFinishTimer: Timer?
    private var lastGuidedPointTime: CFTimeInterval = 0
    
    // Single LiDAR volume calculator:
    private let volumeCalculator: VolumeCalculatorType = VolumeCalculatorType()
    
    /// All anchors from LiDAR mesh, if needed
    var scannedMeshAnchors: [ARMeshAnchor] = []

    // For user-tapped points and lines:
    private var pointNodes: [SCNNode] = []
    private var lineNode: SCNNode?

    // For filled wound polygon and depth visualization
    private var filledPolygonNode: SCNNode?
    private var depthPointNode: SCNNode?

    // Track mesh nodes per anchor to render LiDAR mesh
    private var meshNodesByAnchorID: [UUID: SCNNode] = [:]

    // Rendering options
    private var renderMeshAsWireframe: Bool = false
    private var colorMeshByClassification: Bool = false

    // Throttling
    private var lastMeshUpdateTime: TimeInterval = 0
    private var meshUpdateInterval: TimeInterval = 0.2 // seconds

    // Core Image and Vision
    private let ciContext = CIContext()
    private let edgeFilter = CIFilter.edges()
    private var edgeOverlayView: UIImageView = UIImageView()
    
    // HUD label for prompts
    private var hudLabel: UILabel = UILabel()
    
    // New boolean property for disabling HUD prompts globally
    private var hudPromptsEnabled: Bool = true

    // Vision / Core ML segmentation scaffolding
    private var segmentationRequest: VNCoreMLRequest?
    private var lastSegmentationTime: TimeInterval = 0
    private var segmentationInterval: TimeInterval = 1.0 // seconds (changed from 0.5 to 1.0)
    private var isRunningSegmentation = false

    // New boolean flag and thresholds for auto segmentation and contour processing
    private var isAutoSegmentationEnabled = true
    private let contourDownsampleStep = 6
    private let rdpEpsilon: CGFloat = 4.0
    private let contourChangeThreshold: CGFloat = 16.0 // pixels total change (changed from 20.0)

    // Added flags for manual overlays and refining state:
    private var showManualPointOverlays = false
    private var isRefiningWound = false
    private var isApplyingSegmentationResult = false

    // Refine flow controls
    private var refinePassesRemaining: Int = 0
    private let maxRefinePasses: Int = 4
    private var stablePolygonCount: Int = 0
    private let requiredStableChecks: Int = 2

    // Debounce / throttle helpers
    private var metricsDebounceWorkItem: DispatchWorkItem?
    private var lastMetricsUpdateTime: TimeInterval = 0
    private let metricsMinInterval: TimeInterval = 0.2 // 5 Hz

    private var lastPolygonUpdateTime: TimeInterval = 0
    private let polygonMinInterval: TimeInterval = 0.25 // 250 ms

    // Change tracking
    private var lastBoundaryHash: Int = 0
    private var lastDepthPoint: SIMD3<Float>? = nil
    private var lastPolygonScreenSample: [CGPoint] = []
    
    // Added new throttling property for display link frame count
    private var displayLinkFrameCounter: Int = 0
    
    // Added new property for tracking settle window after starting refine
    private var trackingSettleUntil: CFTimeInterval = 0

    // Track whether AR session is currently running to avoid double-run
    private var sessionRunning: Bool = false

    // New flag to avoid multiple auto-saves of current scan
    private var didAutoSaveCurrentScan: Bool = false

    // Keep display link strongly so it doesn't disappear mid-flight.
    private var displayLink: CADisplayLink?
    // Dedicated queue for Vision segmentation
    private let segmentationQueue = DispatchQueue(label: "ai.woundcorder.segmentation")

    deinit {
        displayLink?.invalidate()
        NotificationCenter.default.removeObserver(self)
    }

    // Bound property from SwiftUI, used to store the boundary polygon
    var userPoints: Binding<[SIMD3<Float>]>!

    // Callback for passing perimeter/area/volume back to SwiftUI
    var onMetricsCalculated: ((Float, Float, Float) -> Void)?
    var onMeasurementState: ((MeasurementState) -> Void)?

    // Additional callback to SwiftUI for UX overlay
    struct MeasurementState {
        enum Confidence { case low, medium, high }
        let area: Float?
        let volume: Float?
        let isPreliminary: Bool
        let confidence: Confidence
    }
    
    private struct MeshSnapshot {
        let transform: simd_float4x4
        let vertexData: Data
        let vertexCount: Int
        let vertexStride: Int
        let vertexOffset: Int
        let indexData: Data
        let faceCount: Int
        let indexCountPerPrimitive: Int
    }

    /// The “depth point” from SwiftUI side
    var selectedDepthPoint: SIMD3<Float>? {
        didSet {
            if let point = selectedDepthPoint {
                // Tell the volumeCalculator this is our userSelectedPoint
                volumeCalculator.userSelectedPoint = point
                print("Updated wound depth point: \(point)")
                // Update depth point visualization
                updateDepthPointNode()
                updateDepthIndicatorLine()
            }
        }
    }

    private var depthLineNode: SCNNode?
    
    // In-AR metrics label
    private var metricsTextNode: SCNNode?

    // Settings & configuration
    private let manualOverlaysDefaultsKey = "ManualOverlaysEnabled"
    private var refineDuration: TimeInterval = 8.0

    // Tunable HUD messages
    private var hudMessageMoveCloser = "Move closer and scan around the wound to capture depth"
    private var hudMessageMeasurementsReady = "Measurements ready"
    private var hudMessageRefine = "  Wound analysis in progress. Move around to refine.  "

    // MARK: - Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()
        setupARScene()
        setupGestures()
        setupObservers()
        setupSegmentationPipeline()
        loadSettings()
    }

    // MARK: - AR Setup

    private func setupARScene() {
        sceneView.delegate = self
        sceneView.session.delegate = self
        sceneView.scene = SCNScene()
        
        sceneView.autoenablesDefaultLighting = true
        sceneView.automaticallyUpdatesLighting = true

        view.addSubview(sceneView)
        sceneView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            sceneView.topAnchor.constraint(equalTo: view.topAnchor),
            sceneView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            sceneView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            sceneView.trailingAnchor.constraint(equalTo: view.trailingAnchor)
        ])

        // Edge overlay view
        edgeOverlayView.translatesAutoresizingMaskIntoConstraints = false
        edgeOverlayView.backgroundColor = .clear
        edgeOverlayView.isUserInteractionEnabled = false
        view.addSubview(edgeOverlayView)
        NSLayoutConstraint.activate([
            edgeOverlayView.topAnchor.constraint(equalTo: view.topAnchor),
            edgeOverlayView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            edgeOverlayView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            edgeOverlayView.trailingAnchor.constraint(equalTo: view.trailingAnchor)
        ])

        // HUD label setup
        hudLabel.textAlignment = .center
        hudLabel.textColor = .white
        hudLabel.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        hudLabel.font = UIFont.systemFont(ofSize: 15, weight: .semibold)
        hudLabel.layer.cornerRadius = 10
        hudLabel.layer.masksToBounds = true
        hudLabel.alpha = 0
        hudLabel.numberOfLines = 2
        hudLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(hudLabel)
        NSLayoutConstraint.activate([
            hudLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            hudLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 8),
            hudLabel.widthAnchor.constraint(lessThanOrEqualToConstant: 320)
        ])

        // ---- AR configuration with capability checks ----
        guard ARWorldTrackingConfiguration.isSupported else {
            showHUDPrompt("AR World Tracking not supported on this device")
            return
        }
        // Avoid enabling an already-enabled session
        guard !sessionRunning else { return }

        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal]

        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification) {
            config.sceneReconstruction = .meshWithClassification
        } else if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }

        sceneView.session.run(config)
        sessionRunning = true

        // Start display link for edge overlay updates and keep strong reference
        displayLink = CADisplayLink(target: self, selector: #selector(updateEdgeOverlay))
        displayLink?.add(to: .main, forMode: .common)
    }

    // MARK: - Gesture Setup

    /// Returns a world-space point for a screen location by preferring SceneKit mesh hits, then AR raycasts.
    private func worldPointFromScreen(_ screenPoint: CGPoint) -> SIMD3<Float>? {
        // 1) Try SceneKit hit test against current scene geometry (including LiDAR mesh nodes)
        let hits = sceneView.hitTest(screenPoint, options: [SCNHitTestOption.firstFoundOnly: true])
        if let hit = hits.first {
            let p = hit.worldCoordinates
            if p.x.isFinite && p.y.isFinite && p.z.isFinite { return SIMD3<Float>(p.x, p.y, p.z) }
        }
        // 2) Prefer AR raycast against existing plane geometry (avoids going through objects)
        if let query = sceneView.raycastQuery(from: screenPoint, allowing: .existingPlaneGeometry, alignment: .any),
           let result = sceneView.session.raycast(query).first {
            let t = result.worldTransform
            return SIMD3<Float>(t.columns.3.x, t.columns.3.y, t.columns.3.z)
        }
        // 3) Fallback to infinite plane (more permissive)
        if let query = sceneView.raycastQuery(from: screenPoint, allowing: .existingPlaneInfinite, alignment: .any),
           let result = sceneView.session.raycast(query).first {
            let t = result.worldTransform
            return SIMD3<Float>(t.columns.3.x, t.columns.3.y, t.columns.3.z)
        }
        // 4) Last resort: estimated plane
        if let query = sceneView.raycastQuery(from: screenPoint, allowing: .estimatedPlane, alignment: .any),
           let result = sceneView.session.raycast(query).first {
            let t = result.worldTransform
            return SIMD3<Float>(t.columns.3.x, t.columns.3.y, t.columns.3.z)
        }
        return nil
    }

    private func setupGestures() {
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        sceneView.addGestureRecognizer(tapGesture)

        let pan = UIPanGestureRecognizer(target: self, action: #selector(handleGuidedPan(_:)))
        pan.minimumNumberOfTouches = 1
        pan.maximumNumberOfTouches = 1
        sceneView.addGestureRecognizer(pan)
    }

    @objc private func handleTap(_ gesture: UITapGestureRecognizer) {
        if !isManualOutlineMode { return }

        // The user taps once, but we do multiple quick raycasts and average them
        let location = gesture.location(in: sceneView)
        
        // We'll run, say, 10 samples over ~0.3 seconds
        let samplesCount = 10
        var collectedPoints: [SIMD3<Float>] = []
        let sampleInterval = 0.03 // 30ms between each sample
        var sampleIndex = 0

        Timer.scheduledTimer(withTimeInterval: sampleInterval, repeats: true) { timer in
            sampleIndex += 1
            
            if let point = self.worldPointFromScreen(location) {
                collectedPoints.append(point)
            }

            // Once we've gathered all
            if sampleIndex >= samplesCount {
                timer.invalidate()
                let finalPoint = self.averagePoint(collectedPoints)
                self.handleFinalTapPoint(finalPoint)
            }
        }
    }

    @objc private func handleGuidedPan(_ gesture: UIPanGestureRecognizer) {
        guard guidedActive && isManualOutlineMode else { return }
        let location = gesture.location(in: sceneView)

        switch gesture.state {
        case .began, .changed:
            // Sample points along the drag path, but don't overload: add every ~10 px
            if let last = projectToScreen(userPoints.wrappedValue).last {
                let dx = location.x - last.x
                let dy = location.y - last.y
                let dist = sqrt(dx*dx + dy*dy)
                if dist < 10 { return }
            }
            if let p = worldPointFromScreen(location) {
                userPoints.wrappedValue.append(p)
                addPointToScene(at: p)
                updateFilledPolygon()
                updateMetrics()
                lastGuidedPointTime = CACurrentMediaTime()
                if userPoints.wrappedValue.count >= guidedMaxPoints {
                    finishManualOutline()
                }
            }
        case .ended, .cancelled, .failed:
            // If user stopped drawing and we have enough points, auto-finish soon
            lastGuidedPointTime = CACurrentMediaTime()
        default:
            break
        }
    }

    /// Single-use function that does just **one** raycast and returns a 3D point (if found).
    private func singleRaycast3D(at screenPoint: CGPoint) -> SIMD3<Float>? {
        return worldPointFromScreen(screenPoint)
    }

    /// Helper to compute the average of a list of 3D points
    private func averagePoint(_ points: [SIMD3<Float>]) -> SIMD3<Float> {
        guard !points.isEmpty else { return SIMD3<Float>(0,0,0) }
        var sum = SIMD3<Float>(0,0,0)
        for p in points { sum += p }
        return sum / Float(points.count)
    }

    private func handleFinalTapPoint(_ point: SIMD3<Float>) {
        // Always treat taps as ROI boundary points in manual mode
        userPoints.wrappedValue.append(point)
        addPointToScene(at: point)
        if showManualPointOverlays {
            updateFilledPolygon()
        }
        // Auto-select depth as centroid of ROI when possible
        if userPoints.wrappedValue.count >= 3 {
            if let deep = deepestPointInROI(boundary: userPoints.wrappedValue) {
                selectedDepthPoint = deep
            } else {
                selectedDepthPoint = centroid(of: userPoints.wrappedValue)
            }
            updateDepthPointNode()
            updateDepthIndicatorLine()
        }
        updateMetrics()
        
        // Guided flow auto-finish logic
        lastGuidedPointTime = CACurrentMediaTime()
        if guidedActive && userPoints.wrappedValue.count >= guidedMaxPoints {
            finishManualOutline()
        }
    }


    // MARK: - Observers

    private func setupObservers() {
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(resetARScene),
                                               name: NSNotification.Name("ResetARScene"),
                                               object: nil)

        NotificationCenter.default.addObserver(self,
                                               selector: #selector(exportARScene),
                                               name: NSNotification.Name("ExportARScene"),
                                               object: nil)

        // Removed observer for SelectDepthPoint per instructions
        
        NotificationCenter.default.addObserver(self, selector: #selector(toggleWireframe), name: NSNotification.Name("ToggleWireframe"), object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(toggleClassificationColors), name: NSNotification.Name("ToggleClassificationColors"), object: nil)
        
        // Added observers for auto segmentation toggle
        NotificationCenter.default.addObserver(self, selector: #selector(enableAutoSegmentation), name: NSNotification.Name("EnableAutoSegmentation"), object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(disableAutoSegmentation), name: NSNotification.Name("DisableAutoSegmentation"), object: nil)
        
        // Added observer for immediate wound detection run (renamed)
        NotificationCenter.default.addObserver(self, selector: #selector(runWoundAnalysisNow), name: NSNotification.Name("RunWoundAnalysisNow"), object: nil)
        
        // Observer for manual overlays toggle from Settings UI
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(handleManualOverlaysToggle(_:)),
                                               name: NSNotification.Name("ManualOverlaysChanged"),
                                               object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(startMeasureFlow), name: NSNotification.Name("StartMeasureFlow"), object: nil)

        NotificationCenter.default.addObserver(self, selector: #selector(startManualOutline), name: NSNotification.Name("StartManualOutline"), object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(finishManualOutline), name: NSNotification.Name("FinishManualOutline"), object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(clearManualOutline), name: NSNotification.Name("ClearManualOutline"), object: nil)
        
        // Added observers to disable/enable HUD prompts globally
        NotificationCenter.default.addObserver(self, selector: #selector(disableHUDPrompts), name: NSNotification.Name("DisableHUDPrompts"), object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(enableHUDPrompts), name: NSNotification.Name("EnableHUDPrompts"), object: nil)
        
        // Added new observer for guided wound measurement flow
        NotificationCenter.default.addObserver(self, selector: #selector(startGuidedWoundMeasurement), name: NSNotification.Name("StartGuidedWoundMeasurement"), object: nil)
    }
    
    @objc private func enableAutoSegmentation() { isAutoSegmentationEnabled = true }
    @objc private func disableAutoSegmentation() { isAutoSegmentationEnabled = false }
    
    @objc private func disableHUDPrompts() { hudPromptsEnabled = false; hideHUDPrompt() }
    @objc private func enableHUDPrompts() { hudPromptsEnabled = true }
    
    @objc private func handleManualOverlaysToggle(_ note: Notification) {
        // Expecting userInfo["enabled"] as Bool
        if let enabled = note.userInfo?["enabled"] as? Bool {
            showManualPointOverlays = enabled
            UserDefaults.standard.set(enabled, forKey: manualOverlaysDefaultsKey)
            // Update existing visuals: remove or add overlays accordingly
            if !enabled {
                // Remove current manual overlays
                for n in pointNodes { n.removeFromParentNode() }
                pointNodes.removeAll()
                lineNode?.removeFromParentNode()
                lineNode = nil
            } else {
                // Rebuild overlays from current userPoints
                for p in userPoints.wrappedValue {
                    let sphereNode = SCNNode(geometry: SCNSphere(radius: 0.005))
                    sphereNode.geometry?.firstMaterial?.diffuse.contents = UIColor.red
                    sphereNode.position = SCNVector3(p)
                    sceneView.scene.rootNode.addChildNode(sphereNode)
                    pointNodes.append(sphereNode)
                }
                autoConnectPoints()
            }
        }
    }

    // MARK: - Settings

    private func loadSettings() {
        // Load persisted flag for manual overlays
        let enabled = UserDefaults.standard.bool(forKey: manualOverlaysDefaultsKey)
        showManualPointOverlays = enabled
    }

    // MARK: - HUD Prompt Methods
    
    private func showHUDPrompt(_ message: String, duration: TimeInterval = 2.0) {
        guard hudPromptsEnabled else { return }
        hudLabel.text = "  \(message)  "
        UIView.animate(withDuration: 0.25) { self.hudLabel.alpha = 1 }
        DispatchQueue.main.asyncAfter(deadline: .now() + duration) {
            UIView.animate(withDuration: 0.25) { self.hudLabel.alpha = 0 }
        }
    }

    private func showRefinePrompt() {
        guard hudPromptsEnabled else { return }
        hudLabel.text = hudMessageRefine
        UIView.animate(withDuration: 0.25) { self.hudLabel.alpha = 1 }
    }

    private func hideHUDPrompt() {
        guard hudPromptsEnabled else { return }
        UIView.animate(withDuration: 0.25) { self.hudLabel.alpha = 0 }
    }

    @objc private func runWoundAnalysisNow() {
        lastSegmentationTime = 0

        // Show immediate detecting feedback and temporarily hide manual overlays
        showHUDPrompt("Running Wound Analysis…", duration: 1.0)
        // Temporarily disable manual overlays to avoid visual conflict during auto-detect
        let previousManualOverlayState = showManualPointOverlays
        showManualPointOverlays = false
        for n in pointNodes { n.removeFromParentNode() }
        pointNodes.removeAll()
        lineNode?.removeFromParentNode()
        lineNode = nil

        isAutoSegmentationEnabled = true
        isRefiningWound = true
        
        // Tracking settle window after starting refine
        let trackingSettleDuration: TimeInterval = 1.0
        self.segmentationInterval = 1.5
        self.trackingSettleUntil = CACurrentMediaTime() + trackingSettleDuration

        showRefinePrompt()
        // Auto-disable after configurable duration
        DispatchQueue.main.asyncAfter(deadline: .now() + refineDuration) { [weak self] in
            guard let self = self else { return }
            self.isAutoSegmentationEnabled = false
            self.isRefiningWound = false

            // Restore previous manual overlay preference after refine window
            self.showManualPointOverlays = previousManualOverlayState

            self.segmentationInterval = 1.0
            self.hideHUDPrompt()
            self.updateMetrics() // ensure latest metrics pushed
        }
    }
    @objc private func startMeasureFlow() {
        // Clear previous and kick a quick pass + refine
        runWoundAnalysisNow()
        refinePassesRemaining = maxRefinePasses
        stablePolygonCount = 0
    }

    @objc private func startManualOutline() {
        isManualOutlineMode = true
        showManualPointOverlays = true
        isAutoSegmentationEnabled = false
        isRefiningWound = false
        showHUDPrompt("Tap around the wound to outline it")
    }

    @objc private func finishManualOutline() {
        isManualOutlineMode = false
        isAutoSegmentationEnabled = false
        updateFilledPolygon()
        updateMetrics()
        showHUDPrompt("Manual outline saved")
        guidedActive = false
        guidedAutoFinishTimer?.invalidate(); guidedAutoFinishTimer = nil
        runGuidedSegmentation()
    }

    @objc private func clearManualOutline() {
        userPoints.wrappedValue.removeAll()
        for n in pointNodes { n.removeFromParentNode() }
        pointNodes.removeAll()
        lineNode?.removeFromParentNode()
        lineNode = nil
        filledPolygonNode?.removeFromParentNode()
        filledPolygonNode = nil
        updateMetrics()
        showHUDPrompt("Outline cleared")
        guidedActive = false
        guidedAutoFinishTimer?.invalidate(); guidedAutoFinishTimer = nil
    }
    
    @objc private func startGuidedWoundMeasurement() {
        // Reset overlays and state
        userPoints.wrappedValue.removeAll()
        for n in pointNodes { n.removeFromParentNode() }
        pointNodes.removeAll()
        lineNode?.removeFromParentNode(); lineNode = nil
        filledPolygonNode?.removeFromParentNode(); filledPolygonNode = nil
        depthPointNode?.removeFromParentNode(); depthPointNode = nil
        depthLineNode?.removeFromParentNode(); depthLineNode = nil
        selectedDepthPoint = nil

        // Enable manual outline mode with visible overlays
        isManualOutlineMode = true
        showManualPointOverlays = true
        isAutoSegmentationEnabled = false
        isRefiningWound = false
        guidedActive = true

        // Show overlays and prompt
        NotificationCenter.default.post(name: NSNotification.Name("ARShowOverlays"), object: nil)
        showHUDPrompt("Draw a rough circle around the wound. Tap Finish when done.")

        // Start inactivity-based auto-finish timer
        lastGuidedPointTime = CACurrentMediaTime()
        guidedAutoFinishTimer?.invalidate()
        guidedAutoFinishTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] t in
            guard let self = self else { t.invalidate(); return }
            let now = CACurrentMediaTime()
            let count = self.userPoints.wrappedValue.count
            if count >= self.guidedMaxPoints {
                t.invalidate()
                self.finishManualOutline()
                return
            }
            if count >= self.guidedMinPointsToAuto && (now - self.lastGuidedPointTime) >= self.guidedInactivitySeconds {
                t.invalidate()
                self.finishManualOutline()
            }
        }
    }

    private func runGuidedSegmentation() {
        guard let frame = sceneView.session.currentFrame else { return }
        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let extent = ciImage.extent
        guard let cgImage = ciContext.createCGImage(ciImage, from: extent) else { return }
        // Orient for UI usage
        let iface: UIInterfaceOrientation = {
            if let ws = self.view.window?.windowScene {
                if #available(iOS 18.0, *) { return ws.effectiveGeometry.interfaceOrientation } else { return ws.interfaceOrientation }
            }
            return .portrait
        }()
        let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: iface.toUIImageOrientationForBackCamera())

        // Replaced original block with Vision + Core ML integration:

        // Run Core ML via Vision using the bundled WoundSegmentation.mlmodelc
        guard
            let modelURL = Bundle.main.url(forResource: "WoundSegmentation", withExtension: "mlmodelc"),
            let mlModel = try? MLModel(contentsOf: modelURL),
            let vnModel = try? VNCoreMLModel(for: mlModel)
        else {
            DispatchQueue.main.async {
                self.showHUDPrompt("Using manual outline (model unavailable)")
                self.updateFilledPolygon(); self.updateMetrics()
            }
            return
        }

        let request = VNCoreMLRequest(model: vnModel) { [weak self] req, _ in
            guard let self = self else { return }
            // Expect a segmentation mask as a pixel buffer
            guard let obs = req.results?.first as? VNPixelBufferObservation else { return }
            let pb = obs.pixelBuffer
            let ciMask = CIImage(cvPixelBuffer: pb)
            guard let cgMask = self.ciContext.createCGImage(ciMask, from: ciMask.extent) else { return }

            // Constrain mask to user ROI in screen space
            let roiScreen = self.projectToScreen(self.userPoints.wrappedValue)
            guard roiScreen.count >= 3 else { return }

            let viewSize = self.sceneView.bounds.size
            UIGraphicsBeginImageContextWithOptions(viewSize, false, 1.0)
            guard let ctx = UIGraphicsGetCurrentContext() else { UIGraphicsEndImageContext(); return }
            // Draw mask scaled to view size
            ctx.interpolationQuality = .low
            ctx.draw(cgMask, in: CGRect(origin: .zero, size: viewSize))
            // Create ROI path in view coordinates
            let path = UIBezierPath()
            if let first = roiScreen.first { path.move(to: first) }
            for p in roiScreen.dropFirst() { path.addLine(to: p) }
            path.close()
            // Clip to ROI and redraw mask to keep only ROI region
            ctx.addPath(path.cgPath); ctx.clip()
            ctx.draw(cgMask, in: CGRect(origin: .zero, size: viewSize))
            let constrained = UIGraphicsGetImageFromCurrentImageContext()?.cgImage
            UIGraphicsEndImageContext()
            guard let constrainedMask = constrained else { return }

            // Detect contours on constrained mask
            let handler = VNImageRequestHandler(cgImage: constrainedMask, options: [:])
            let contoursReq = VNDetectContoursRequest()
            contoursReq.contrastAdjustment = 1.0
            contoursReq.detectsDarkOnLight = true
            do {
                try handler.perform([contoursReq])
            } catch {
                return
            }
            guard let observation = contoursReq.results?.first as? VNContoursObservation else { return }
            var bestContour: VNContour?; var bestCount = 0
            for i in 0..<observation.contourCount {
                if let c = try? observation.contour(at: i), c.pointCount > bestCount { bestCount = c.pointCount; bestContour = c }
            }
            guard let contour = bestContour else { return }
            let normalizedPoints = contour.normalizedPoints
            var screenPts: [CGPoint] = []
            for p in normalizedPoints {
                let x = CGFloat(p.x) * viewSize.width
                let y = (1 - CGFloat(p.y)) * viewSize.height
                screenPts.append(CGPoint(x: x, y: y))
            }
            let ds = self.downsample(points: screenPts, step: self.contourDownsampleStep)
            let smooth = self.rdp(ds, epsilon: self.rdpEpsilon)
            let capped = Array(smooth.prefix(120))

            var new3D: [SIMD3<Float>] = []
            for sp in capped {
                if let wp = self.worldPointFromScreen(sp) {
                    new3D.append(wp)
                }
            }
            DispatchQueue.main.async {
                if new3D.count >= 3 {
                    self.userPoints.wrappedValue = new3D
                    if let deep = self.deepestPointInROI(boundary: new3D) { self.selectedDepthPoint = deep } else { self.selectedDepthPoint = self.centroid(of: new3D) }
                    self.updateDepthPointNode(); self.updateDepthIndicatorLine(); self.updateFilledPolygon(); self.updateMetrics()
                    self.showHUDPrompt(self.hudMessageMeasurementsReady)
                }
            }
        }
        request.imageCropAndScaleOption = .scaleFill

        // Perform on the captured image CGImage
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            // If Vision fails, fall back to manual outline
            DispatchQueue.main.async {
                self.showHUDPrompt("Using manual outline (Vision failed)")
                self.updateFilledPolygon(); self.updateMetrics()
            }
        }
    }

    // MARK: - Raycast

    private func performRaycast(at location: CGPoint) -> simd_float4x4? {
        guard let query = sceneView.raycastQuery(from: location, allowing: .estimatedPlane, alignment: .horizontal),
              let result = sceneView.session.raycast(query).first else {
            return nil
        }
        return result.worldTransform
    }

    // MARK: - Adding/Connecting Points

    private func addPointToScene(at point: SIMD3<Float>) {
        // Always track logical points, but hide visual overlays when disabled
        if showManualPointOverlays {
            let sphereNode = SCNNode(geometry: SCNSphere(radius: 0.005))
            sphereNode.geometry?.firstMaterial?.diffuse.contents = UIColor.red
            sphereNode.position = SCNVector3(point)
            sceneView.scene.rootNode.addChildNode(sphereNode)
            pointNodes.append(sphereNode)
        }
        autoConnectPoints()
    }

    private func autoConnectPoints() {
        lineNode?.removeFromParentNode()
        guard userPoints.wrappedValue.count > 1 else { return }
        guard showManualPointOverlays else { return }

        let vertices = userPoints.wrappedValue.map { SCNVector3($0.x, $0.y, $0.z) }
        let indices = (0..<vertices.count).flatMap { [Int32($0), Int32(($0 + 1) % vertices.count)] }

        let vertexSource = SCNGeometrySource(vertices: vertices)
        let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.size)

        let element = SCNGeometryElement(
            data: indexData,
            primitiveType: .line,
            primitiveCount: indices.count / 2,
            bytesPerIndex: MemoryLayout<Int32>.size
        )

        let geometry = SCNGeometry(sources: [vertexSource], elements: [element])
        geometry.materials.first?.diffuse.contents = UIColor.green
        lineNode = SCNNode(geometry: geometry)

        sceneView.scene.rootNode.addChildNode(lineNode!)
    }

    // MARK: - Wound Highlight (Filled Polygon)

    private func updateFilledPolygon() {
        filledPolygonNode?.removeFromParentNode()

        // Throttle updates
        let now = CACurrentMediaTime()
        // During refine, allow slightly faster updates
        let minInterval = isRefiningWound ? 0.2 : polygonMinInterval
        if now - lastPolygonUpdateTime < minInterval { return }

        let points3D = userPoints.wrappedValue
        let sampleScreen = projectToScreen(points3D)
        // Compare a small screen-space sample to detect meaningful change
        if !lastPolygonScreenSample.isEmpty {
            let dist = polygonScreenDistance(sampleScreen, lastPolygonScreenSample)
            if dist < contourChangeThreshold { return }
        }
        lastPolygonScreenSample = sampleScreen
        lastPolygonUpdateTime = now

        let points = userPoints.wrappedValue
        // Simple guard to avoid heavy triangulation for very large polygons frequently
        if points.count > 200 { return }
        guard points.count >= 3 else { return }

        // Triangulate a simple polygon via ear clipping (assumes points are roughly coplanar and ordered)
        let triangles = triangulatePolygon(points)
        guard !triangles.isEmpty else { return }

        // Build SCNGeometry from triangles
        var vertices: [SCNVector3] = []
        var indices: [Int32] = []
        for tri in triangles {
            let base = Int32(vertices.count)
            vertices.append(SCNVector3(tri.0))
            vertices.append(SCNVector3(tri.1))
            vertices.append(SCNVector3(tri.2))
            indices.append(contentsOf: [base, base + 1, base + 2])
        }

        let vertexSource = SCNGeometrySource(vertices: vertices)
        let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.size)
        let element = SCNGeometryElement(data: indexData,
                                         primitiveType: .triangles,
                                         primitiveCount: indices.count / 3,
                                         bytesPerIndex: MemoryLayout<Int32>.size)
        let geometry = SCNGeometry(sources: [vertexSource], elements: [element])
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.green.withAlphaComponent(0.35)
        material.emission.contents = UIColor.green.withAlphaComponent(0.15)
        material.isDoubleSided = true
        geometry.materials = [material]

        let node = SCNNode(geometry: geometry)
        sceneView.scene.rootNode.addChildNode(node)
        filledPolygonNode = node

        // Refresh metrics label position and text with last known values (will be finalized on next metrics update)
        updateMetricsTextNode(perimeter: nil, area: nil, volume: nil)
    }

    // Basic ear clipping triangulation for 3D points by projecting to best-fit plane
    private func triangulatePolygon(_ points: [SIMD3<Float>]) -> [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] {
        guard points.count >= 3 else { return [] }

        // Compute best-fit plane normal using Newell's method
        let normal = polygonNormal(points)
        // Build a local 2D basis (u, v) on the plane
        let (u, v) = planeBasis(from: normal)
        let origin = points[0]

        // Project 3D to 2D
        var pts2D: [CGPoint] = points.map { p in
            let d = p - origin
            let x = CGFloat(simd_dot(d, u))
            let y = CGFloat(simd_dot(d, v))
            return CGPoint(x: x, y: y)
        }

        // Maintain index mapping to original 3D points
        var indices = Array(0..<points.count)
        var tris: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = []

        func area(_ a: CGPoint, _ b: CGPoint, _ c: CGPoint) -> CGFloat {
            return 0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x))
        }

        func isCCW() -> Bool {
            var sum: CGFloat = 0
            for i in 0..<pts2D.count {
                let a = pts2D[i]
                let b = pts2D[(i + 1) % pts2D.count]
                sum += (b.x - a.x) * (b.y + a.y)
            }
            return sum < 0
        }

        // Ensure CCW orientation
        if !isCCW() {
            pts2D.reverse()
            indices.reverse()
        }

        var guardCounter = 0
        while indices.count >= 3 && guardCounter < 1000 {
            guardCounter += 1
            var earFound = false
            for i in 0..<indices.count {
                let i0 = (i - 1 + indices.count) % indices.count
                let i1 = i
                let i2 = (i + 1) % indices.count

                let a = pts2D[i0]
                let b = pts2D[i1]
                let c = pts2D[i2]

                // Check convexity
                if area(a, b, c) <= 0 { continue }

                // Check no other point inside the triangle
                var contains = false
                for j in 0..<indices.count where j != i0 && j != i1 && j != i2 {
                    if pointInTriangle(pts2D[j], a, b, c) {
                        contains = true
                        break
                    }
                }
                if contains { continue }

                // It's an ear; add triangle in 3D
                let pa = points[indices[i0]]
                let pb = points[indices[i1]]
                let pc = points[indices[i2]]
                tris.append((pa, pb, pc))

                // Clip ear
                pts2D.remove(at: i1)
                indices.remove(at: i1)
                earFound = true
                break
            }
            if !earFound { break }
        }
        return tris
    }

    private func polygonNormal(_ points: [SIMD3<Float>]) -> SIMD3<Float> {
        var n = SIMD3<Float>(0,0,0)
        for i in 0..<points.count {
            let p0 = points[i]
            let p1 = points[(i + 1) % points.count]
            n.x += (p0.y - p1.y) * (p0.z + p1.z)
            n.y += (p0.z - p1.z) * (p0.x + p1.x)
            n.z += (p0.x - p1.x) * (p0.y + p1.y)
        }
        let len = max(1e-6, simd_length(n))
        return n / len
    }

    private func planeBasis(from normal: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) {
        let n = simd_normalize(normal)
        let up = abs(n.y) < 0.9 ? SIMD3<Float>(0,1,0) : SIMD3<Float>(1,0,0)
        let u = simd_normalize(simd_cross(up, n))
        let v = simd_normalize(simd_cross(n, u))
        return (u, v)
    }

    private func pointInTriangle(_ p: CGPoint, _ a: CGPoint, _ b: CGPoint, _ c: CGPoint) -> Bool {
        // Barycentric technique
        let v0 = CGPoint(x: c.x - a.x, y: c.y - a.y)
        let v1 = CGPoint(x: b.x - a.x, y: b.y - a.y)
        let v2 = CGPoint(x: p.x - a.x, y: p.y - a.y)

        let dot00 = v0.x * v0.x + v0.y * v0.y
        let dot01 = v0.x * v1.x + v0.y * v1.y
        let dot02 = v0.x * v2.x + v0.y * v2.y
        let dot11 = v1.x * v1.x + v1.y * v1.y
        let dot12 = v1.x * v2.x + v1.y * v2.y

        let invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        let u = (dot11 * dot02 - dot01 * dot12) * invDenom
        let v = (dot00 * dot12 - dot01 * dot02) * invDenom
        return u >= 0 && v >= 0 && (u + v) <= 1
    }

    // MARK: - Depth Point Visualization

    private func updateDepthPointNode() {
        depthPointNode?.removeFromParentNode()
        guard let p = selectedDepthPoint else { return }
        let node = SCNNode(geometry: SCNSphere(radius: 0.007))
        node.geometry?.firstMaterial?.diffuse.contents = UIColor.blue
        node.position = SCNVector3(p)
        sceneView.scene.rootNode.addChildNode(node)
        depthPointNode = node
    }

    private func updateDepthIndicatorLine() {
        depthLineNode?.removeFromParentNode()
        let boundary = userPoints.wrappedValue
        guard boundary.count >= 3 else { return }
        guard let depth = selectedDepthPoint else { return }
        // Use polygon plane to find projection of depth point
        let n = polygonNormal(boundary)
        let p0 = boundary[0]
        // Project depth onto plane
        let v = depth - p0
        let distance = simd_dot(v, n)
        let projected = depth - distance * n

        // Build a thin cylinder between projected and depth
        let start = SCNVector3(projected)
        let end = SCNVector3(depth)
        let radius: CGFloat = isRefiningWound ? 0.0025 : 0.0015
        let node = lineNodeBetween(start: start, end: end, radius: radius, color: .blue)
        sceneView.scene.rootNode.addChildNode(node)
        depthLineNode = node
    }

    private func lineNodeBetween(start: SCNVector3, end: SCNVector3, radius: CGFloat, color: UIColor) -> SCNNode {
        let vector = SCNVector3(end.x - start.x, end.y - start.y, end.z - start.z)
        let height = CGFloat(sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z))
        guard height > 0 else { return SCNNode() }
        let cylinder = SCNCylinder(radius: radius, height: height)
        cylinder.firstMaterial?.diffuse.contents = color
        let node = SCNNode(geometry: cylinder)
        node.position = SCNVector3((start.x + end.x)/2, (start.y + end.y)/2, (start.z + end.z)/2)
        node.eulerAngles = SCNVector3Make(Float.pi/2, 0, 0)
        // Align cylinder to the vector direction
        let dir = SCNVector3(vector.x / Float(height), vector.y / Float(height), vector.z / Float(height))
        node.look(at: SCNVector3(node.position.x + dir.x, node.position.y + dir.y, node.position.z + dir.z))
        return node
    }

    // MARK: - LiDAR Mesh Visualization

    private func makeSnapshot(from anchor: ARMeshAnchor) -> MeshSnapshot? {
        let geometry = anchor.geometry
        // Basic presence checks
        guard geometry.vertices.count > 0, geometry.faces.count > 0 else { return nil }
        // Copy vertex buffer into Data using count/stride/offset
        let vDesc = geometry.vertices
        let vCount = vDesc.count
        guard vCount > 0 else { return nil }
        let vBuffer = vDesc.buffer
        let vOffset = vDesc.offset
        let vStride = vDesc.stride
        let vLength = vBuffer.length
        let neededVBytes = vOffset + (vCount - 1) * vStride + 3 * MemoryLayout<Float>.size
        guard vLength >= neededVBytes else { return nil }
        // Copy the entire region that we will read from later
        let vData = Data(bytesNoCopy: vBuffer.contents(), count: vLength, deallocator: .none)

        // Faces
        let fDesc = geometry.faces
        let faceCount = fDesc.count
        let indexCountPerPrimitive = fDesc.indexCountPerPrimitive
        guard indexCountPerPrimitive == 3, faceCount > 0 else { return nil }
        let totalIndexCount = faceCount * indexCountPerPrimitive
        let iBuffer = fDesc.buffer
        let iLength = iBuffer.length
        let neededIBytes = totalIndexCount * MemoryLayout<UInt32>.size
        guard iLength >= neededIBytes else { return nil }
        let iData = Data(bytesNoCopy: iBuffer.contents(), count: neededIBytes, deallocator: .none)

        return MeshSnapshot(
            transform: anchor.transform,
            vertexData: vData,
            vertexCount: vCount,
            vertexStride: vStride,
            vertexOffset: vOffset,
            indexData: iData,
            faceCount: faceCount,
            indexCountPerPrimitive: indexCountPerPrimitive
        )
    }

    private func updateMeshNode(for anchor: ARMeshAnchor) {
        // Avoid rebuilding geometry during refine window to reduce contention
        if isRefiningWound { 
            #if DEBUG
            if let frame = sceneView.session.currentFrame {
                debugLog("updateMeshNode: refining=\(isRefiningWound) tracking=\(frame.camera.trackingState)")
            } else {
                debugLog("updateMeshNode: no currentFrame")
            }
            #endif
            return 
        }
        
        #if DEBUG
        if let frame = sceneView.session.currentFrame {
            debugLog("updateMeshNode: refining=\(isRefiningWound) tracking=\(frame.camera.trackingState)")
        } else {
            debugLog("updateMeshNode: no currentFrame")
        }
        #endif
        
        // Skip when tracking is not normal or during settle window
        if let frame = sceneView.session.currentFrame {
            switch frame.camera.trackingState {
            case .normal: break
            default:
                return
            }
        }
        if CACurrentMediaTime() < trackingSettleUntil { 
            #if DEBUG
            debugLog("updateMeshNode: passed tracking and settle gates")
            #endif
            return 
        }
        #if DEBUG
        debugLog("updateMeshNode: passed tracking and settle gates")
        #endif

        // Skip if geometry is empty/transient
        // Use snapshot to avoid lifetime issues
        guard let snapshot = makeSnapshot(from: anchor) else { return }
        #if DEBUG
        debugLog("updateMeshNode: snapshot vCount=\(snapshot.vertexCount) faceCount=\(snapshot.faceCount)")
        #endif
        
        // Startup grace period to avoid early transient meshes
        let elapsed = CACurrentMediaTime() - meshBuildStartTime
        if elapsed < 2.0 {
            #if DEBUG
            debugLog("Skipping mesh build due to startup grace period (elapsed=\(elapsed))")
            #endif
            return
        }
        // Skip if camera is very close (< 15 cm or 25 cm during refine) to reduce instability
        let camDist = distanceCameraTo(anchor: anchor)
        let minDist: Float = isRefiningWound ? 0.25 : 0.15
        if camDist < minDist {
            #if DEBUG
            debugLog("Skipping mesh build due to close camera: dist=\(camDist)m min=\(minDist)m")
            #endif
            return
        }

        // Remove existing node for this anchor if any
        if let existing = meshNodesByAnchorID[anchor.identifier] {
            existing.removeFromParentNode()
            meshNodesByAnchorID.removeValue(forKey: anchor.identifier)
        }

        let node = SCNNode()
        if let geom = scnGeometry(from: snapshot) {
            #if DEBUG
            debugLog("updateMeshNode: built geom sources=\(geom.sources.count) elements=\(geom.elements.count)")
            #endif
            
            guard !geom.sources.isEmpty, !geom.elements.isEmpty else {
                #if DEBUG
                debugLog("updateMeshNode: geometry has empty sources/elements; skipping")
                #endif
                return
            }
            
            let material = SCNMaterial()
            material.isDoubleSided = true
            
            if colorMeshByClassification {
                // Classification by face is not available on this SDK; fall back to a neutral color
                material.diffuse.contents = UIColor.gray.withAlphaComponent(0.3)
            } else {
                material.diffuse.contents = UIColor.gray.withAlphaComponent(0.3)
            }
            
            if renderMeshAsWireframe {
                material.fillMode = .lines
            }
            geom.firstMaterial = material
            
            #if DEBUG
            if let first = geom.elements.first {
                debugLog("updateMeshNode: first primitiveCount=\(first.primitiveCount)")
            }
            #endif
            
            if let first = geom.elements.first, first.primitiveCount > 0 {
                if Thread.isMainThread {
                    node.geometry = geom
                } else {
                    DispatchQueue.main.async { node.geometry = geom }
                }
            } else {
                #if DEBUG
                debugLog("updateMeshNode: primitiveCount is zero; skipping")
                #endif
                return
            }
        } else {
            // Skip adding node if geometry is invalid/transient
            return
        }

        let addNodeBlock = {
            #if DEBUG
            self.debugLog("updateMeshNode: adding node for anchor \(anchor.identifier)")
            #endif
            self.sceneView.scene.rootNode.addChildNode(node)
            self.meshNodesByAnchorID[anchor.identifier] = node
            #if DEBUG
            self.debugLog("updateMeshNode: node added for anchor \(anchor.identifier)")
            #endif
        }
        if Thread.isMainThread {
            addNodeBlock()
        } else {
            DispatchQueue.main.async { addNodeBlock() }
        }
    }

    private func removeMeshNode(for anchor: ARMeshAnchor) {
        if let node = meshNodesByAnchorID[anchor.identifier] {
            node.removeFromParentNode()
            meshNodesByAnchorID.removeValue(forKey: anchor.identifier)
        }
    }

    private func scnGeometry(from snapshot: MeshSnapshot) -> SCNGeometry? {
        // Access counts and buffers from snapshot
        let vertexCount = snapshot.vertexCount
        guard vertexCount > 0 else { return nil }

        // Early bail on tiny/transient meshes to avoid crashes during reconstruction
        if vertexCount < 50 {
            #if DEBUG
            debugLog("Skipping tiny mesh: vertexCount=\(vertexCount)")
            #endif
            return nil
        }

        // Transform to world coordinates using snapshot.transform, reading from snapshot.vertexData
        var worldVertices: [SIMD3<Float>] = []
        worldVertices.reserveCapacity(vertexCount)
        let transform = snapshot.transform
        let vData = snapshot.vertexData
        let vOffset = snapshot.vertexOffset
        let vStride = snapshot.vertexStride
        let neededVertexBytes = vOffset + (vertexCount - 1) * vStride + 3 * MemoryLayout<Float>.size
        guard vData.count >= neededVertexBytes else { return nil }
        vData.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            for i in 0..<vertexCount {
                let baseIdx = vOffset + i * vStride
                let px = base.advanced(by: baseIdx)
                let x = px.load(as: Float.self)
                let y = px.advanced(by: MemoryLayout<Float>.size).load(as: Float.self)
                let z = px.advanced(by: 2 * MemoryLayout<Float>.size).load(as: Float.self)
                let local = SIMD3<Float>(x, y, z)
                let wp4 = transform * SIMD4<Float>(local.x, local.y, local.z, 1)
                worldVertices.append(SIMD3<Float>(wp4.x, wp4.y, wp4.z))
            }
        }

        // Validate vertices are finite
        guard worldVertices.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else {
            #if DEBUG
            debugLog("Non-finite vertex detected; skipping geometry")
            #endif
            return nil
        }

        // Build SCNGeometrySource from worldVertices
        let scnVertices = worldVertices.map { SCNVector3($0.x, $0.y, $0.z) }
        guard !scnVertices.isEmpty else { return nil }
        let vSource = SCNGeometrySource(vertices: scnVertices)

        // Faces from snapshot
        let faceCount = snapshot.faceCount
        let indexCountPerPrimitive = snapshot.indexCountPerPrimitive
        guard indexCountPerPrimitive == 3 else { return nil }
        if faceCount < 20 {
            #if DEBUG
            debugLog("Skipping tiny mesh faces: faceCount=\(faceCount)")
            #endif
            return nil
        }
        let totalIndexCount = faceCount * indexCountPerPrimitive
        guard totalIndexCount > 0 else { return nil }

        let iRaw = snapshot.indexData
        guard iRaw.count == totalIndexCount * MemoryLayout<UInt32>.size else { return nil }

        // Read indices and filter degenerates
        var indices: [Int32] = []
        indices.reserveCapacity(totalIndexCount)
        iRaw.withUnsafeBytes { raw in
            let u32 = raw.bindMemory(to: UInt32.self)
            guard u32.count == totalIndexCount else { return }
            for i in stride(from: 0, to: totalIndexCount, by: 3) {
                indices.append(Int32(u32[i + 0]))
                indices.append(Int32(u32[i + 1]))
                indices.append(Int32(u32[i + 2]))
            }
        }
        var filtered: [Int32] = []
        filtered.reserveCapacity(indices.count)
        for i in stride(from: 0, to: indices.count, by: 3) {
            let ia = Int(indices[i])
            let ib = Int(indices[i + 1])
            let ic = Int(indices[i + 2])
            if ia == ib || ib == ic || ia == ic { continue }
            guard ia >= 0 && ib >= 0 && ic >= 0 && ia < worldVertices.count && ib < worldVertices.count && ic < worldVertices.count else { continue }
            let a = worldVertices[ia]
            let b = worldVertices[ib]
            let c = worldVertices[ic]
            let ab = a - b
            let ac = a - c
            let cross = simd_cross(ab, ac)
            let area2 = simd_length(cross)
            if area2 < 1e-8 { continue }
            filtered.append(indices[i])
            filtered.append(indices[i + 1])
            filtered.append(indices[i + 2])
        }
        guard !filtered.isEmpty, filtered.count % 3 == 0 else {
            #if DEBUG
            debugLog("Filtered indices empty or not multiple of 3; skipping geometry")
            #endif
            return nil
        }

        let indexData = Data(bytes: filtered, count: filtered.count * MemoryLayout<Int32>.size)
        let element = SCNGeometryElement(
            data: indexData,
            primitiveType: .triangles,
            primitiveCount: filtered.count / 3,
            bytesPerIndex: MemoryLayout<Int32>.size
        )
        let scnGeom = SCNGeometry(sources: [vSource], elements: [element])
        guard let firstElement = scnGeom.elements.first, firstElement.primitiveCount > 0 else { return nil }
        return scnGeom
    }

    // MARK: - Metrics

    @objc private func updateMetrics() {
        // Only compute if boundary or depth changed and obey debounce
        let boundary = userPoints.wrappedValue
        let boundaryHash = boundary.reduce(into: Hasher()) { hasher, p in
            hasher.combine(p.x)
            hasher.combine(p.y)
            hasher.combine(p.z)
        }.finalize()

        let depthChanged = (lastDepthPoint != selectedDepthPoint)
        let boundaryChanged = (boundaryHash != lastBoundaryHash)
        guard boundaryChanged || depthChanged else { return }

        lastBoundaryHash = boundaryHash
        lastDepthPoint = selectedDepthPoint

        metricsDebounceWorkItem?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self = self else { return }
            let now = CACurrentMediaTime()
            if now - self.lastMetricsUpdateTime < self.metricsMinInterval { return }
            self.lastMetricsUpdateTime = now

            if self.scannedMeshAnchors.isEmpty {
                // Preliminary: compute 2D area from boundary plane
                let boundary2DArea = self.computePlanarArea(self.userPoints.wrappedValue)
                self.onMetricsCalculated?(0, boundary2DArea ?? 0, 0)
                let state = MeasurementState(area: boundary2DArea, volume: nil, isPreliminary: true, confidence: .low)
                self.onMeasurementState?(state)
                DispatchQueue.main.async {
                    self.updateMetricsTextNode(perimeter: nil, area: boundary2DArea, volume: nil)
                }
                return
            }

            let cameraPosition = self.getCameraPosition()
            self.volumeCalculator.scannedMeshAnchors = self.scannedMeshAnchors
            self.volumeCalculator.woundBoundaryPoints = boundary
            if let depthPoint = self.selectedDepthPoint {
                self.volumeCalculator.userSelectedPoint = depthPoint
            }
            self.volumeCalculator.calculateMetrics(
                from: self.sceneView.session,
                cameraPosition: cameraPosition
            ) { [weak self] perimeter, area, volume in
                guard let self = self else { return }
                self.onMetricsCalculated?(perimeter, area, volume)
                DispatchQueue.main.async {
                    self.updateMetricsTextNode(perimeter: perimeter, area: area, volume: volume)
                    let conf = self.computeConfidence(boundary: self.userPoints.wrappedValue)
                    let state = MeasurementState(area: area, volume: volume, isPreliminary: false, confidence: conf)
                    self.onMeasurementState?(state)
                    if volume > 0, !self.didAutoSaveCurrentScan {
                        self.didAutoSaveCurrentScan = true
                        self.saveCurrentOutlineAndMesh()
                    }
                }
            }
        }
        metricsDebounceWorkItem = work
        DispatchQueue.main.asyncAfter(deadline: .now() + metricsMinInterval, execute: work)
    }

    private func getCameraPosition() -> SIMD3<Float> {
        guard let frame = sceneView.session.currentFrame else {
            print("Error: No active AR frame.")
            return SIMD3<Float>(0,0,0)
        }
        let cam = frame.camera.transform.columns.3
        return SIMD3<Float>(cam.x, cam.y, cam.z)
    }
    private func distanceCameraTo(anchor: ARMeshAnchor) -> Float {
        let cam = getCameraPosition()
        let t = anchor.transform.columns.3
        let a = SIMD3<Float>(t.x, t.y, t.z)
        return simd_length(a - cam)
    }

    // MARK: - ARSCNViewDelegate

    func renderer(_ renderer: SCNSceneRenderer,
                  didAdd node: SCNNode,
                  for anchor: ARAnchor) {
        guard let meshAnchor = anchor as? ARMeshAnchor else { return }
        // Skip during refine to reduce churn
        if isRefiningWound { return }
        // Require normal tracking
        if let frame = sceneView.session.currentFrame {
            switch frame.camera.trackingState {
            case .normal: break
            default: return
            }
        }
        scannedMeshAnchors.append(meshAnchor)
        DispatchQueue.main.async {
            self.updateMeshNode(for: meshAnchor)
            self.updateMetrics()
        }
    }
    func renderer(_ renderer: SCNSceneRenderer,
                  didUpdate node: SCNNode,
                  for anchor: ARAnchor) {
        guard let meshAnchor = anchor as? ARMeshAnchor else { return }
        if let frame = sceneView.session.currentFrame {
            switch frame.camera.trackingState {
            case .normal: break
            default: return
            }
        }
        if let index = scannedMeshAnchors.firstIndex(where: { $0.identifier == meshAnchor.identifier }) {
            scannedMeshAnchors[index] = meshAnchor
            let now = CACurrentMediaTime()
            if now - lastMeshUpdateTime > meshUpdateInterval {
                lastMeshUpdateTime = now
                if !isRefiningWound {
                    if meshAnchor.geometry.vertices.count > 0 && meshAnchor.geometry.faces.count > 0 {
                        updateMeshNode(for: meshAnchor)
                    }
                }
                updateMetrics()
            }
        }
    }
    
    func renderer(_ renderer: SCNSceneRenderer,
                  didRemove node: SCNNode,
                  for anchor: ARAnchor) {
        guard let meshAnchor = anchor as? ARMeshAnchor else { return }
        if let index = scannedMeshAnchors.firstIndex(where: { $0.identifier == meshAnchor.identifier }) {
            scannedMeshAnchors.remove(at: index)
            removeMeshNode(for: meshAnchor)
            updateMetrics()
        }
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession,
                 didFailWithError error: Error) {
        print("AR Session failed: \(error)")
    }

    func sessionWasInterrupted(_ session: ARSession) {
        print("AR Session interrupted")
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        print("AR Session interruption ended")
        updateMetrics()
    }

    func session(_ session: ARSession,
                 didUpdate anchors: [ARAnchor]) {
        var meshesUpdated = false
        for anchor in anchors {
            if let meshAnchor = anchor as? ARMeshAnchor {
                if let index = scannedMeshAnchors.firstIndex(where: { $0.identifier == meshAnchor.identifier }) {
                    scannedMeshAnchors[index] = meshAnchor
                } else {
                    scannedMeshAnchors.append(meshAnchor)
                }
                meshesUpdated = true
            }
        }
        
        if meshesUpdated {
            let now = CACurrentMediaTime()
            if now - lastMeshUpdateTime > meshUpdateInterval {
                lastMeshUpdateTime = now
                updateMetrics()
            }
        }
    }

    // MARK: - Reset & Export

    @objc private func resetARScene() {
        print("Resetting AR Scene...")

        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal]
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification) {
            config.sceneReconstruction = .meshWithClassification
        } else if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }

        scannedMeshAnchors.removeAll()
        userPoints.wrappedValue.removeAll()
        selectedDepthPoint = nil

        sceneView.scene.rootNode.enumerateChildNodes { node, _ in
            node.removeFromParentNode()
        }
        meshNodesByAnchorID.removeAll()
        filledPolygonNode = nil
        depthPointNode = nil

        metricsTextNode?.removeFromParentNode()
        metricsTextNode = nil

        // Reset auto-save flag on reset
        didAutoSaveCurrentScan = false

        // Mark session as not running so we can safely restart
        sessionRunning = false

        sceneView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        sessionRunning = true
    }

    @objc private func exportARScene() {
        let fileManager = FileManager.default
        guard let documentsPath = fileManager.urls(for: .documentDirectory,
                                                   in: .userDomainMask).first else {
            print("Failed to get documents directory")
            return
        }

        let exportDirectory = documentsPath.appendingPathComponent("ARExport")
        try? fileManager.createDirectory(at: exportDirectory,
                                         withIntermediateDirectories: true)

        let usdzURL = exportDirectory.appendingPathComponent("WoundScan.usdz")
        let plyURL  = exportDirectory.appendingPathComponent("WoundScan.ply")

        exportMeshToPLY(to: plyURL)

        sceneView.scene.write(to: usdzURL,
                              options: nil,
                              delegate: nil) { _, error, _ in
            if let error = error {
                print("Export failed: \(error.localizedDescription)")
            } else {
                print("USDZ export succeeded: \(usdzURL)")

                let zipURL = documentsPath.appendingPathComponent("WoundScanExport.zip")
                self.createZipFile(from: exportDirectory, to: zipURL) { success in
                    if success {
                        DispatchQueue.main.async {
                            self.presentShareSheet(with: zipURL)
                        }
                    }
                }
            }
        }
    }

    private func exportMeshToPLY(to url: URL) {
        var plyData = "ply\nformat ascii 1.0\n"
        var vertices: [SIMD3<Float>] = []
        var exportFaces: [(Int, Int, Int)] = []
        var vertexCount = 0
        var faceCount = 0

        for anchor in scannedMeshAnchors {
            let geometry = anchor.geometry

            // Skip anchors with empty/transient geometry
            guard geometry.vertices.count > 0, geometry.faces.count > 0 else { continue }

            let transform = anchor.transform

            let vDesc = geometry.vertices
            let vCount = vDesc.count
            guard vCount > 0 else { continue }
            let vBuffer = vDesc.buffer
            let vOffset = vDesc.offset
            let vStride = vDesc.stride
            let vLength = vBuffer.length
            let neededVBytes = vOffset + (vCount - 1) * vStride + 3 * MemoryLayout<Float>.size
            guard vLength >= neededVBytes else { continue }

            let vBase = vBuffer.contents()
            var transformedVerts: [SIMD3<Float>] = []
            transformedVerts.reserveCapacity(vCount)
            for i in 0..<vCount {
                let base = vOffset + i * vStride
                let px = vBase.advanced(by: base)
                let py = px.advanced(by: MemoryLayout<Float>.size)
                let pz = py.advanced(by: MemoryLayout<Float>.size)
                let x = px.load(as: Float.self)
                let y = py.load(as: Float.self)
                let z = pz.load(as: Float.self)
                let wp = transform * SIMD4<Float>(x, y, z, 1)
                transformedVerts.append(SIMD3<Float>(wp.x, wp.y, wp.z))
            }
            vertices.append(contentsOf: transformedVerts)

            // Validate newly added vertices are finite; if not, remove them and skip this anchor
            let startIndex = vertices.count - transformedVerts.count
            let endIndex = vertices.count
            let finite = vertices[startIndex..<endIndex].allSatisfy { v in v.x.isFinite && v.y.isFinite && v.z.isFinite }
            if !finite {
                vertices.removeLast(transformedVerts.count)
                continue
            }

            let meshFaces = geometry.faces
            let indexCountPerPrimitive = meshFaces.indexCountPerPrimitive
            let totalIndexCount = meshFaces.count * indexCountPerPrimitive
            guard indexCountPerPrimitive == 3, totalIndexCount > 0 else { continue }
            let indicesBuffer = meshFaces.buffer
            let indexBufferLength = indicesBuffer.length
            let neededIndexBytes = totalIndexCount * MemoryLayout<UInt32>.size
            guard indexBufferLength >= neededIndexBytes else { continue }

            let iData = Data(bytesNoCopy: indicesBuffer.contents(), count: neededIndexBytes, deallocator: .none)
            iData.withUnsafeBytes { raw in
                let u32 = raw.bindMemory(to: UInt32.self)
                guard u32.count >= totalIndexCount else { return }
                for i in stride(from: 0, to: totalIndexCount, by: 3) {
                    let a = Int(u32[i])
                    let b = Int(u32[i + 1])
                    let c = Int(u32[i + 2])
                    // Skip duplicate indices
                    if a == b || b == c || a == c { continue }
                    // Compute global indices
                    let ia = a + vertexCount
                    let ib = b + vertexCount
                    let ic = c + vertexCount
                    // Bounds check against accumulated vertices
                    if ia < 0 || ib < 0 || ic < 0 || ia >= vertices.count || ib >= vertices.count || ic >= vertices.count { continue }
                    // Area check to drop degenerate triangles
                    let va = vertices[ia]
                    let vb = vertices[ib]
                    let vc = vertices[ic]
                    let ab = SIMD3<Float>(va.x - vb.x, va.y - vb.y, va.z - vb.z)
                    let ac = SIMD3<Float>(va.x - vc.x, va.y - vc.y, va.z - vc.z)
                    let cross = simd_cross(ab, ac)
                    let area2 = simd_length(cross)
                    if area2 < 1e-8 { continue }
                    exportFaces.append((ia, ib, ic))
                }
            }

            vertexCount += geometry.vertices.count
            faceCount += meshFaces.count
        }

        // PLY header
        plyData += "element vertex \(vertexCount)\n"
        plyData += "property float x\nproperty float y\nproperty float z\n"
        plyData += "element face \(faceCount)\n"
        plyData += "property list uchar int vertex_indices\n"
        plyData += "end_header\n"

        for v in vertices {
            plyData += String(format: "%.6f %.6f %.6f\n", v.x, v.y, v.z)
        }

        for f in exportFaces {
            plyData += "3 \(f.0) \(f.1) \(f.2)\n"
        }

        do {
            try plyData.write(to: url, atomically: true, encoding: .utf8)
            print("PLY export succeeded: \(url)")
        } catch {
            print("PLY export failed: \(error.localizedDescription)")
        }
    }

    func createZipFile(from directory: URL, to zipURL: URL, completion: @escaping (Bool) -> Void) {
        do {
            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: zipURL.path) {
                try fileManager.removeItem(at: zipURL)
            }
            try fileManager.zipItem(at: directory, to: zipURL)
            print("ZIP file created at: \(zipURL)")
            completion(true)
        } catch {
            print("Failed to create ZIP file: \(error.localizedDescription)")
            completion(false)
        }
    }

    private func presentShareSheet(with url: URL) {
        let activityViewController = UIActivityViewController(
            activityItems: [url],
            applicationActivities: nil
        )
        activityViewController.popoverPresentationController?.sourceView = self.view

        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let rootVC = windowScene.windows.first?.rootViewController {
            rootVC.present(activityViewController, animated: true)
        }
    }

    // MARK: - Edge Overlay Update

    @objc private func updateEdgeOverlay() {
        guard let frame = sceneView.session.currentFrame else { return }
        // Skip overlay updates if the overlay view is not in a window (not visible)
        guard edgeOverlayView.window != nil else { return }

        // Throttle edge overlay to every 3rd frame to reduce load
        displayLinkFrameCounter += 1
        if displayLinkFrameCounter % 3 != 0 { return }
        
        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        // Apply edge filter (tune intensity via inputIntensity)
        edgeFilter.inputImage = ciImage
        edgeFilter.intensity = 5.0
        guard let edged = edgeFilter.outputImage else { return }

        // Convert to UIImage with orientation correction
        let extent = edged.extent
        if let cgImage = ciContext.createCGImage(edged, from: extent) {
            let iface: UIInterfaceOrientation = {
                if let ws = self.view.window?.windowScene {
                    if #available(iOS 18.0, *) {
                        return ws.effectiveGeometry.interfaceOrientation
                    } else {
                        return ws.interfaceOrientation
                    }
                }
                return .portrait
            }()
            let imgOrientation = iface.toUIImageOrientationForBackCamera()
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: imgOrientation)
            edgeOverlayView.image = uiImage
            edgeOverlayView.alpha = 0.6
            edgeOverlayView.contentMode = .scaleAspectFill
        }

        // Show HUD if tracking is poor
        if let frame = sceneView.session.currentFrame {
            switch frame.camera.trackingState {
            case .normal:
                break
            default:
                showHUDPrompt("Move slower / improve lighting / point at textured surfaces", duration: 1.0)
            }
        }

        // Contextual prompts
        if scannedMeshAnchors.isEmpty && userPoints.wrappedValue.count >= 3 {
            showHUDPrompt(hudMessageMoveCloser)
        } else if userPoints.wrappedValue.count < 3 {
            // No polygon yet; no specific prompt here
        }

        // Gate segmentation on good tracking and after settle window
        if let frame = sceneView.session.currentFrame {
            switch frame.camera.trackingState {
            case .normal: break
            default: return
            }
        }
        if CACurrentMediaTime() < trackingSettleUntil { return }

        // Throttled segmentation, gated by isAutoSegmentationEnabled and refine passes
        let now = CACurrentMediaTime()
        if isAutoSegmentationEnabled, !isRunningSegmentation {
            if refinePassesRemaining > 0 || !isRefiningWound {
                if now - lastSegmentationTime > segmentationInterval {
                    lastSegmentationTime = now
                    runSegmentation(on: pixelBuffer)
                    if isRefiningWound { refinePassesRemaining = max(0, refinePassesRemaining - 1) }
                }
            }
        }
    }

    // MARK: - Vision Segmentation Pipeline

    private func setupSegmentationPipeline() {
        guard
            let modelURL = Bundle.main.url(forResource: "WoundSegmentation", withExtension: "mlmodelc"),
            let compiledModel = try? MLModel(contentsOf: modelURL),
            let vnModel = try? VNCoreMLModel(for: compiledModel)
        else {
            segmentationRequest = nil
            #if DEBUG
            print("Segmentation model not found/loaded; running without it.")
            #endif
            return
        }
        let request = VNCoreMLRequest(model: vnModel) { [weak self] request, _ in
            guard let self = self else { return }
            defer { DispatchQueue.main.async { self.isRunningSegmentation = false } }
            if let result = request.results?.first as? VNPixelBufferObservation {
                DispatchQueue.main.async {
                    self.processSegmentationMask(result.pixelBuffer)
                }
            }
        }
        request.imageCropAndScaleOption = .scaleFill
        segmentationRequest = request
    }

    private func runSegmentation(on pixelBuffer: CVPixelBuffer) {
        guard isAutoSegmentationEnabled, let request = segmentationRequest else { return }
        if isRunningSegmentation { return }
        isRunningSegmentation = true

        segmentationQueue.async { [weak self] in
            guard let self = self else { return }
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let targetWidth: CGFloat = 320
            let scale = targetWidth / max(1, ciImage.extent.width)
            let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            let rect = CGRect(origin: .zero, size: CGSize(width: scaledImage.extent.width, height: scaledImage.extent.height))
            guard let cgScaled = self.ciContext.createCGImage(scaledImage, from: rect) else {
                DispatchQueue.main.async { self.isRunningSegmentation = false }
                return
            }
            let handler = VNImageRequestHandler(cgImage: cgScaled, options: [:])
            do {
                try handler.perform([request])
            } catch {
                #if DEBUG
                print("Vision perform error:", error)
                #endif
                DispatchQueue.main.async { self.isRunningSegmentation = false }
            }
        }
    }

    private func processSegmentationMask(_ mask: CVPixelBuffer) {
        // Convert mask to CGImage
        let ci = CIImage(cvPixelBuffer: mask)
        let context = ciContext
        guard let cgMask = context.createCGImage(ci, from: ci.extent) else { return }
        // Removed unused variables width and height here as per instructions

        // Create bitmap data
        guard let dataProvider = cgMask.dataProvider,
              let data = dataProvider.data as Data? else { return }
        // Threshold and find a simple contour along the center row as a placeholder
        // For a full solution, implement marching squares or use Vision contours (VNDetectContoursRequest)
        // Here we use VNDetectContoursRequest for simplicity
        let contoursRequest = VNDetectContoursRequest()
        contoursRequest.contrastAdjustment = 1.0
        contoursRequest.detectsDarkOnLight = true
        let handler = VNImageRequestHandler(cgImage: cgMask, options: [:])
        do {
            try handler.perform([contoursRequest])
            guard let observation = contoursRequest.results?.first as? VNContoursObservation else {
                return
            }
            // Choose the largest contour
            var bestContour: VNContour?
            var bestCount = 0
            for i in 0..<observation.contourCount {
                if let contour = try? observation.contour(at: i) {
                    if contour.pointCount > bestCount {
                        bestCount = contour.pointCount
                        bestContour = contour
                    }
                }
            }
            guard let contour = bestContour else { return }

            // Convert normalized points to screen points
            let normalizedPoints = contour.normalizedPoints
            // Build screen points with smoothing (downsample + RDP)
            var rawScreen: [CGPoint] = []
            let size = self.sceneView.bounds.size
            for p in normalizedPoints {
                // VNContours are normalized to image coordinates with origin bottom-left; adjust if needed
                let x = CGFloat(p.x) * size.width
                let y = (1 - CGFloat(p.y)) * size.height
                rawScreen.append(CGPoint(x: x, y: y))
            }
            let ds = downsample(points: rawScreen, step: contourDownsampleStep)
            let smooth = rdp(ds, epsilon: rdpEpsilon)

            // Cap to maximum number of points
            let cappedSmooth = Array(smooth.prefix(80))

            let newScreen = cappedSmooth
            let distForStability = self.polygonScreenDistance(newScreen, self.lastPolygonScreenSample)
            
            // Prevent overlapping application of segmentation results
            if self.isApplyingSegmentationResult { return }
            self.isApplyingSegmentationResult = true
            
            if distForStability < self.contourChangeThreshold {
                self.stablePolygonCount += 1
            } else {
                self.stablePolygonCount = 0
            }
            let isStableEnough = self.stablePolygonCount >= self.requiredStableChecks

            // Optional: compare with previous polygon to avoid flicker
            var shouldUpdate = true
            if let existing = self.userPoints?.wrappedValue, existing.count >= 3 {
                // crude approach: always allow update (detailed reprojection is heavy)
                shouldUpdate = true
            }

            // Proceed to raycast smooth points
            var new3DPoints: [SIMD3<Float>] = []
            for sp in cappedSmooth {
                if let wp = self.worldPointFromScreen(sp) {
                    new3DPoints.append(wp)
                }
            }

            DispatchQueue.main.async {
                defer { self.isApplyingSegmentationResult = false }
                if shouldUpdate, new3DPoints.count >= 3, isStableEnough {
                    // Switch HUD from detecting to refine once polygon is found
                    self.showRefinePrompt()

                    self.lastPolygonScreenSample = newScreen
                    self.userPoints.wrappedValue = new3DPoints
                    self.updateFilledPolygon()

                    if self.scannedMeshAnchors.isEmpty {
                        self.showHUDPrompt(self.hudMessageMoveCloser)
                    }

                    // Auto-select depth point at centroid of polygon for indicator
                    if let deep = self.deepestPointInROI(boundary: new3DPoints) {
                        self.selectedDepthPoint = deep
                    } else {
                        self.selectedDepthPoint = self.centroid(of: new3DPoints)
                    }
                    self.updateDepthPointNode()
                    self.updateDepthIndicatorLine()
                    self.updateMetrics()

                    // If we have any mesh and metrics likely updated, notify user
                    if !self.scannedMeshAnchors.isEmpty {
                        self.showHUDPrompt(self.hudMessageMeasurementsReady)
                    } else {
                        // keep refine prompt visible during refine window
                    }
                }
            }
        } catch {
            return
        }
    }
    
    // MARK: - Toggle Wireframe & Classification Colors

    @objc private func toggleWireframe() {
        renderMeshAsWireframe.toggle()
        // Re-apply materials to existing mesh nodes
        for (id, node) in meshNodesByAnchorID {
            if let idx = scannedMeshAnchors.firstIndex(where: { $0.identifier == id }),
               let geom = node.geometry {
                let anchor = scannedMeshAnchors[idx]
                let material = SCNMaterial()
                material.isDoubleSided = true
                material.diffuse.contents = UIColor.gray.withAlphaComponent(0.3)
                if renderMeshAsWireframe { material.fillMode = .lines }
                geom.firstMaterial = material
            }
        }
    }

    @objc private func toggleClassificationColors() {
        colorMeshByClassification.toggle()
        // Rebuild materials for anchors
        for (id, _) in meshNodesByAnchorID {
            if let idx = scannedMeshAnchors.firstIndex(where: { $0.identifier == id }) {
                updateMeshNode(for: scannedMeshAnchors[idx])
            }
        }
    }
    
    // MARK: - Helper functions for smoothing and RDP simplification
    
    private func downsample(points: [CGPoint], step: Int) -> [CGPoint] {
        guard step > 1 else { return points }
        return points.enumerated().compactMap { idx, p in idx % step == 0 ? p : nil }
    }

    private func perpendicularDistance(_ p: CGPoint, _ a: CGPoint, _ b: CGPoint) -> CGFloat {
        if a == b { return hypot(p.x - a.x, p.y - a.y) }
        let num = abs((b.y - a.y) * p.x - (b.x - a.x) * p.y + b.x * a.y - b.y * a.x)
        let den = hypot(b.y - a.y, b.x - a.x)
        return num / den
    }

    private func rdp(_ points: [CGPoint], epsilon: CGFloat) -> [CGPoint] {
        guard points.count > 2 else { return points }
        var dmax: CGFloat = 0
        var index = 0
        let end = points.count - 1
        for i in 1..<end {
            let d = perpendicularDistance(points[i], points[0], points[end])
            if d > dmax { dmax = d; index = i }
        }
        if dmax > epsilon {
            let rec1 = rdp(Array(points[0...index]), epsilon: epsilon)
            let rec2 = rdp(Array(points[index...end]), epsilon: epsilon)
            return Array(rec1.dropLast()) + rec2
        } else {
            return [points.first!, points.last!]
        }
    }

    private func centroid(of points: [SIMD3<Float>]) -> SIMD3<Float> {
        guard !points.isEmpty else { return SIMD3<Float>(0,0,0) }
        var sum = SIMD3<Float>(0,0,0)
        for p in points { sum += p }
        return sum / Float(points.count)
    }
    
    private func deepestPointInROI(boundary: [SIMD3<Float>]) -> SIMD3<Float>? {
        guard boundary.count >= 3 else { return nil }
        // Define wound plane
        let n = polygonNormal(boundary)
        let p0 = boundary[0]
        // Build a simple bounding box in 2D plane coords to quickly test inclusion
        let (u, v) = planeBasis(from: n)
        func isInside(_ p: SIMD3<Float>) -> Bool {
            // Project p into plane basis and use winding check against projected polygon
            let origin = p0
            let pts2D: [CGPoint] = boundary.map { bp in
                let d = bp - origin
                let x = CGFloat(simd_dot(d, u))
                let y = CGFloat(simd_dot(d, v))
                return CGPoint(x: x, y: y)
            }
            let pd: SIMD3<Float> = p - origin
            let px = CGFloat(simd_dot(pd, u))
            let py = CGFloat(simd_dot(pd, v))
            let test = CGPoint(x: px, y: py)
            // Point-in-polygon (ray casting)
            var inside = false
            var j = pts2D.count - 1
            for i in 0..<pts2D.count {
                let pi = pts2D[i]
                let pj = pts2D[j]
                let intersect = ((pi.y > test.y) != (pj.y > test.y)) &&
                                (test.x < (pj.x - pi.x) * (test.y - pi.y) / max(1e-6, (pj.y - pi.y)) + pi.x)
                if intersect { inside.toggle() }
                j = i
            }
            return inside
        }
        // Iterate mesh anchors' vertices, transformed to world, find point inside ROI with max distance below plane
        var bestPoint: SIMD3<Float>? = nil
        var maxDepth: Float = -Float.greatestFiniteMagnitude
        for anchor in scannedMeshAnchors {
            let geom = anchor.geometry
            guard geom.vertices.count > 0 else { continue }
            let vDesc = geom.vertices
            let vCount = vDesc.count
            let vBuffer = vDesc.buffer
            let vOffset = vDesc.offset
            let vStride = vDesc.stride
            let transform = anchor.transform
            // Removed line: vBuffer.didAccessValue() // no-op; placeholder to avoid warnings
            // Walk vertices safely
            if vCount == 0 { continue }
            let neededBytes = vOffset + (vCount - 1) * vStride + 3 * MemoryLayout<Float>.size
            if vBuffer.length < neededBytes { continue }
            if let base = vBuffer.contents() as UnsafeMutableRawPointer? {
                for i in 0..<vCount {
                    let baseIdx = vOffset + i * vStride
                    let px = base.advanced(by: baseIdx)
                    let x = px.load(as: Float.self)
                    let y = px.advanced(by: MemoryLayout<Float>.size).load(as: Float.self)
                    let z = px.advanced(by: 2 * MemoryLayout<Float>.size).load(as: Float.self)
                    let local = SIMD3<Float>(x, y, z)
                    let wp4 = transform * SIMD4<Float>(local.x, local.y, local.z, 1)
                    let world = SIMD3<Float>(wp4.x, wp4.y, wp4.z)
                    if !world.x.isFinite || !world.y.isFinite || !world.z.isFinite { continue }
                    if !isInside(world) { continue }
                    // Signed distance from plane (positive above, negative below depending on n)
                    let dist = simd_dot(world - p0, n)
                    // We want the greatest magnitude below plane (most negative)
                    if dist < maxDepth {
                        maxDepth = dist
                        bestPoint = world
                    }
                }
            }
        }
        return bestPoint
    }

    private func computePlanarArea(_ points: [SIMD3<Float>]) -> Float? {
        guard points.count >= 3 else { return nil }
        let normal = polygonNormal(points)
        let (u, v) = planeBasis(from: normal)
        let origin = points[0]
        let pts2D: [CGPoint] = points.map { p in
            let d = p - origin
            let x = CGFloat(simd_dot(d, u))
            let y = CGFloat(simd_dot(d, v))
            return CGPoint(x: x, y: y)
        }
        // Shoelace formula
        var area: CGFloat = 0
        for i in 0..<pts2D.count {
            let j = (i + 1) % pts2D.count
            area += pts2D[i].x * pts2D[j].y - pts2D[j].x * pts2D[i].y
        }
        return Float(abs(area) * 0.5) * 10000.0 // m^2 to cm^2 assuming input in meters
    }

    /// Projects 3D points to 2D screen points
    private func projectToScreen(_ points: [SIMD3<Float>]) -> [CGPoint] {
        guard let pov = sceneView.pointOfView else { return [] }
        let projected = points.compactMap { p -> CGPoint? in
            let scnPos = SCNVector3(p.x, p.y, p.z)
            let projectedVec = sceneView.projectPoint(scnPos)
            if projectedVec.z < 1 { // in front of camera
                return CGPoint(x: CGFloat(projectedVec.x), y: CGFloat(projectedVec.y))
            } else {
                return nil
            }
        }
        return projected
    }

    private func polygonScreenDistance(_ a: [CGPoint], _ b: [CGPoint]) -> CGFloat {
        // crude measure: sum of distances between corresponding points after aligning counts
        guard !a.isEmpty, !b.isEmpty else { return .greatestFiniteMagnitude }
        let n = min(a.count, b.count)
        var total: CGFloat = 0
        for i in 0..<n { total += hypot(a[i].x - b[i].x, a[i].y - b[i].y) }
        return total / CGFloat(n)
    }
    
    // MARK: - Added computeConfidence function
    
    private func computeConfidence(boundary: [SIMD3<Float>]) -> MeasurementState.Confidence {
        let meshCount = scannedMeshAnchors.count
        let currentSample = projectToScreen(boundary)
        let stability = polygonScreenDistance(lastPolygonScreenSample, currentSample)
        if meshCount > 12 && stability < 6 { return .high }
        if meshCount > 4 && stability < 12 { return .medium }
        return .low
    }
    
    // MARK: - In-AR Metrics Text

    private func updateMetricsTextNode(perimeter: Float?, area: Float?, volume: Float?) {
        // Remove if no polygon
        guard userPoints.wrappedValue.count >= 3 else {
            metricsTextNode?.removeFromParentNode()
            metricsTextNode = nil
            return
        }
        // Compute centroid in world space
        let c = centroid(of: userPoints.wrappedValue)
        let text = SCNText(string: formattedMetricsText(area: area, volume: volume), extrusionDepth: 0.001)
        text.font = UIFont.systemFont(ofSize: 0.02, weight: .semibold)
        text.firstMaterial?.diffuse.contents = UIColor.white
        text.firstMaterial?.isDoubleSided = true
        let node = SCNNode(geometry: text)
        // Scale down since SCNText uses meter-based sizing
        let scale: Float = 0.01
        node.scale = SCNVector3(scale, scale, scale)
        // Offset slightly above the centroid
        let offset = SIMD3<Float>(0, 0.01, 0)
        node.position = SCNVector3(c + offset)
        // Face the camera
        if let cam = sceneView.pointOfView {
            node.constraints = [SCNBillboardConstraint()] // always face camera
        }
        // Replace existing
        metricsTextNode?.removeFromParentNode()
        sceneView.scene.rootNode.addChildNode(node)
        metricsTextNode = node
    }

    private func formattedMetricsText(area: Float?, volume: Float?) -> String {
        let areaStr: String
        if let a = area { areaStr = String(format: "Area: %.2f cm²", a) } else { areaStr = "Area: --" }
        let volStr: String
        if let v = volume { volStr = String(format: "Vol: %.2f cm³", v) } else { volStr = "Vol: --" }
        let prelimSuffix = (area != nil && volume == nil) ? " (prelim)" : ""
        return "\(areaStr)  |  \(volStr)\(prelimSuffix)"
    }
    
    // MARK: - Save Current Outline and Mesh
    
    private func saveCurrentOutlineAndMesh() {
        let fileManager = FileManager.default
        guard let documentsPath = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let exportDirectory = documentsPath.appendingPathComponent("ARExport", isDirectory: true)
        try? fileManager.createDirectory(at: exportDirectory, withIntermediateDirectories: true)
        // Save outline points as JSON
        let outlineURL = exportDirectory.appendingPathComponent("WoundOutline.json")
        let outline = userPoints.wrappedValue.map { ["x": $0.x, "y": $0.y, "z": $0.z] }
        if let json = try? JSONSerialization.data(withJSONObject: outline, options: [.prettyPrinted]) {
            try? json.write(to: outlineURL)
        }
        // Save mesh to PLY
        let plyURL = exportDirectory.appendingPathComponent("WoundScan.ply")
        self.exportMeshToPLY(to: plyURL)
    }
}

private extension UIInterfaceOrientation {
    func toUIImageOrientationForBackCamera() -> UIImage.Orientation {
        switch self {
        case .portrait:           return .right
        case .portraitUpsideDown: return .left
        case .landscapeLeft:      return .up
        case .landscapeRight:     return .down
        default:                  return .right
        }
    }
}

