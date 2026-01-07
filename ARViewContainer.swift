//
//  ARViewContainer.swift
//  SwiftUI wrapper for ARViewController providing ARKit integration.
//

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

enum OutlinePhase { case top, bottom, done, none }

/// A top-level SwiftUI placeholder AR container so other files (e.g., ContentView) can reference it as `ARViewContainer`.
/// Replace the body implementation with a real ARKit/RealityKit integration when available.
struct ARViewContainer: View {
    @Binding var topOutlinePoints: [SIMD3<Float>]
    @Binding var bottomOutlinePoints: [SIMD3<Float>]
    @State private var outlinePhase: OutlinePhase
    let metricsManager: MetricsManager
    @Binding var selectedDepthPoint: SIMD3<Float>?

    // Use a loosely-typed closure so ContentView can pass whatever state enum it owns
    // (e.g., ARViewController.MeasurementState) without this file needing that type.
    let onMeasurementState: (AnyMeasurementState) -> Void

    init(
        topOutlinePoints: Binding<[SIMD3<Float>]>,
        bottomOutlinePoints: Binding<[SIMD3<Float>]>,
        metricsManager: MetricsManager,
        selectedDepthPoint: Binding<SIMD3<Float>?>,
        onMeasurementState: @escaping (AnyMeasurementState) -> Void
    ) {
        self._topOutlinePoints = topOutlinePoints
        self._bottomOutlinePoints = bottomOutlinePoints
        self.metricsManager = metricsManager
        self._selectedDepthPoint = selectedDepthPoint
        self.onMeasurementState = onMeasurementState
        self._outlinePhase = State(initialValue: .none)
    }

    var body: some View {
        ARViewControllerHost(topOutlinePoints: $topOutlinePoints,
                             bottomOutlinePoints: $bottomOutlinePoints,
                             outlinePhase: $outlinePhase,
                             metricsManager: metricsManager,
                             selectedDepthPoint: $selectedDepthPoint,
                             onMeasurementState: onMeasurementState)
            .ignoresSafeArea()
    }
}

@MainActor
private struct ARViewControllerHost: UIViewControllerRepresentable {
    @Binding var topOutlinePoints: [SIMD3<Float>]
    @Binding var bottomOutlinePoints: [SIMD3<Float>]
    @Binding var outlinePhase: OutlinePhase
    let metricsManager: MetricsManager
    @Binding var selectedDepthPoint: SIMD3<Float>?
    let onMeasurementState: (AnyMeasurementState) -> Void

    func makeUIViewController(context: Context) -> ARViewController {
        let vc = ARViewController()
        // Bind the SwiftUI points arrays to the controller
        vc.topOutlinePoints = Binding(get: { self.topOutlinePoints }, set: { self.topOutlinePoints = $0 })
        vc.bottomOutlinePoints = Binding(get: { self.bottomOutlinePoints }, set: { self.bottomOutlinePoints = $0 })
        vc.outlinePhase = Binding(get: { self.outlinePhase }, set: { self.outlinePhase = $0 })

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
        // Keep bindings and callbacks current in case dependencies change
        uiViewController.topOutlinePoints = Binding(get: { self.topOutlinePoints }, set: { self.topOutlinePoints = $0 })
        uiViewController.bottomOutlinePoints = Binding(get: { self.bottomOutlinePoints }, set: { self.bottomOutlinePoints = $0 })
        uiViewController.outlinePhase = Binding(get: { self.outlinePhase }, set: { self.outlinePhase = $0 })

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
        topOutlinePoints: .constant([]),
        bottomOutlinePoints: .constant([]),
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

    // SceneKit category masks (used to keep mesh hit-tests and overlays separate)
    private enum NodeCategory {
        static let meshOcclusion: Int = 1 << 1
        static let overlays: Int = 1 << 2
    }

    // Depth-only occlusion mesh nodes to keep AR content from visually "passing through" real-world geometry
    private var meshOcclusionNodesByAnchorID: [UUID: SCNNode] = [:]
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
    private var placementStrictness: String = "strict"

    // Guided outline flow controls
    private var guidedActive: Bool = false
    private var guidedMaxPoints: Int = 600
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
    private var meshUpdateInterval: TimeInterval = 0.02 // seconds

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
    private var segmentationInterval: TimeInterval = 1.0 // seconds
    private var isRunningSegmentation = false

    // New boolean flag and thresholds for auto segmentation and contour processing
    private var isAutoSegmentationEnabled = true
    private let contourDownsampleStep = 6
    private let rdpEpsilon: CGFloat = 3.0
    private let contourChangeThreshold: CGFloat = 24.0 // pixels total change

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
    private var lastProcessedBoundaryHash: Int = 0
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

    // Continuous scanning control
    private var continuousScanningEnabled: Bool = true

    // Expected units from LiDAR volume calculator callback. If your calculator returns SI base (m, m², m³), set these to false to convert to cm-based units.
    private let calculatorUnits = (perimeterIsCM: true, areaIsCM2: true, volumeIsCM3: true)

    // Track last computed metrics and whether to announce when finishing manual outline
    private var lastComputedArea: Float? = nil
    private var lastComputedVolume: Float? = nil
    private var shouldAnnounceFinalMetrics: Bool = false

    // Keep display link strongly so it does not disappear mid-flight.
    private var displayLink: CADisplayLink?
    // Dedicated queue for Vision segmentation
    private let segmentationQueue = DispatchQueue(label: "ai.woundcorder.segmentation")
    private let meshUpdateQueue = DispatchQueue(label: "ai.woundcorder.meshUpdate", qos: .userInitiated)

    // Last tracking HUD shown timestamp to prevent repeated messages
    private var lastTrackingHUDShownAt: CFTimeInterval = 0

    deinit {
        displayLink?.invalidate()
        NotificationCenter.default.removeObserver(self)
    }

    // Bound properties from SwiftUI for outline points and phase
    var topOutlinePoints: Binding<[SIMD3<Float>]>!
    var bottomOutlinePoints: Binding<[SIMD3<Float>]>!
    var outlinePhase: Binding<OutlinePhase>!

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

    // Add struct VolumeQualityFlags with new property calibrationUncertainty
    struct VolumeQualityFlags {
        var contourMismatch: Bool = false
        var incompleteOutline: Bool = false
        var bottomLargerThanTop: Bool = false
        var calibrationUncertainty: Bool = false
    }

    private struct MeshSnapshot {
        let anchorID: UUID
        let faceCount: Int
        let indexCountPerPrimitive: Int
        let faceBytesPerIndex: Int

        let vertexCount: Int
        let vertexStride: Int
        let vertexOffset: Int
        let vertexData: Data
        let indexData: Data
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

    // Settings and configuration
    private let manualOverlaysDefaultsKey = "ManualOverlaysEnabled"
    private var refineDuration: TimeInterval = 8.0

    // Tunable HUD messages
    private var hudMessageMoveCloser = "Move closer and scan around the wound to capture depth"
    private var hudMessageMeasurementsReady = "Measurements ready"
    private var hudMessageRefine = "  Wound analysis in progress. Move around to refine.  "

    // MARK: - Lofted volume helpers (top/bottom outlines -> 3D volume)

    private func resampleClosedPolyline(_ pts: [SIMD3<Float>], targetCount: Int) -> [SIMD3<Float>] {
        guard pts.count >= 3, targetCount >= 3 else { return pts }
        // build cumulative lengths around closed loop
        var loop = pts
        if loop.first != loop.last { loop.append(loop.first!) }
        var lengths: [Float] = [0]
        var total: Float = 0
        for i in 0..<(loop.count - 1) {
            let d = simd_length(loop[i+1] - loop[i])
            total += d
            lengths.append(total)
        }
        guard total > 0 else { return Array(pts.prefix(targetCount)) }
        // sample equally spaced
        var out: [SIMD3<Float>] = []
        out.reserveCapacity(targetCount)
        for k in 0..<targetCount {
            let t = (Float(k) / Float(targetCount)) * total
            // find segment
            var idx = 0
            while idx < lengths.count - 1 && lengths[idx+1] < t { idx += 1 }
            let t0 = lengths[idx]
            let t1 = lengths[idx+1]
            let segT = (t1 - t0) > 0 ? (t - t0) / (t1 - t0) : 0
            let p = simd_mix(loop[idx], loop[idx+1], SIMD3<Float>(repeating: segT))
            out.append(p)
        }
        return out
    }

    /// Resamples two closed contours to the same point count, then:
    ///  - ensures consistent winding (so "top" and "bottom" normals agree)
    ///  - aligns the start index via cyclic shift (minimizes correspondence error)
    /// This dramatically reduces volume spikes when the user starts tracing the two outlines at different locations.
    private func alignedResampledLoops(top: [SIMD3<Float>], bottom: [SIMD3<Float>], targetCount: Int) -> (t: [SIMD3<Float>], b: [SIMD3<Float>])? {
        guard top.count >= 3, bottom.count >= 3 else { return nil }

        let N = max(32, min(targetCount, 256))
        var t = resampleClosedPolyline(top, targetCount: N)
        var b = resampleClosedPolyline(bottom, targetCount: N)
        guard t.count >= 3, b.count >= 3 else { return nil }

        // Normalize winding
        let nt = polygonNormal(t)
        let nb = polygonNormal(b)
        if simd_dot(nt, nb) < 0 {
            b.reverse()
        }

        // Align starts by cyclic shift to minimize RMS distance between corresponding points
        func bestShift(reference: [SIMD3<Float>], candidate: [SIMD3<Float>]) -> Int {
            let n = min(reference.count, candidate.count)
            guard n > 0 else { return 0 }
            var best = 0
            var bestScore: Float = .greatestFiniteMagnitude
            for s in 0..<n {
                var score: Float = 0
                for i in 0..<n {
                    let d = reference[i] - candidate[(i + s) % n]
                    score += simd_length_squared(d)
                }
                if score < bestScore {
                    bestScore = score
                    best = s
                }
            }
            return best
        }

        let shift = bestShift(reference: t, candidate: b)
        if shift != 0 {
            b = Array(b[shift...]) + Array(b[..<shift])
        }

        return (t, b)
    }


    private func signedVolumeOfTriangle(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) -> Float {
        // 1/6 * dot(a, cross(b, c))
        return simd_dot(a, simd_cross(b, c)) / 6.0
    }

    private func loftedVolume(top: [SIMD3<Float>], bottom: [SIMD3<Float>]) -> Float? {
        guard let loops = alignedResampledLoops(top: top, bottom: bottom, targetCount: max(top.count, bottom.count)) else { return nil }
        let t = loops.t
        let b = loops.b
        let N = min(t.count, b.count)
        guard N >= 3 else { return nil }

var vol: Float = 0
        // Side faces as two triangles per quad
        for i in 0..<N {
            let i0 = i
            let i1 = (i + 1) % N
            let t0 = t[i0], t1 = t[i1]
            let b0 = b[i0], b1 = b[i1]
            // triangles: (t0, t1, b0) and (b0, t1, b1)
            vol += signedVolumeOfTriangle(t0, b0, t1)
            vol += signedVolumeOfTriangle(t1, b0, b1)
        }
        // Caps: triangulate top and bottom and add signed volumes
        let topTris = triangulatePolygon(t)
        for tri in topTris { vol += signedVolumeOfTriangle(tri.0, tri.1, tri.2) }
        let bottomTris = triangulatePolygon(b)
        // Reverse winding for bottom cap so normals point outward consistently
        for tri in bottomTris { vol += signedVolumeOfTriangle(tri.2, tri.1, tri.0) }
        return abs(vol)
    }

    // Robust loft metrics in millimeters, with flags
    private func robustLoftMetricsMM(top: [SIMD3<Float>], bottom: [SIMD3<Float>]) -> (topAreaMM2: Float?, bottomAreaMM2: Float?, volumeMM3: Float?, flags: VolumeQualityFlags) {
        var flags = VolumeQualityFlags(contourMismatch: false, incompleteOutline: false, bottomLargerThanTop: false, calibrationUncertainty: false)
        guard top.count >= 3, bottom.count >= 3 else {
            flags.incompleteOutline = true
            return (nil, nil, nil, flags)
        }
        guard let loops = alignedResampledLoops(top: top, bottom: bottom, targetCount: max(top.count, bottom.count)) else {
            flags.incompleteOutline = true
            return (nil, nil, nil, flags)
        }
        let t = loops.t
        let b = loops.b
        let N = min(t.count, b.count)

        // Convert meters to millimeters
        let scale: Float = 1000.0
        let tMM = t.map { $0 * scale }
        let bMM = b.map { $0 * scale }

        func planarAreaMM2(_ pts: [SIMD3<Float>]) -> Float {
            guard pts.count >= 3 else { return 0 }
            let normal = polygonNormal(pts)
            let (u, v) = planeBasis(from: normal)
            let origin = pts[0]
            let pts2D: [CGPoint] = pts.map { p in
                let d = p - origin
                let x = CGFloat(simd_dot(d, u))
                let y = CGFloat(simd_dot(d, v))
                return CGPoint(x: x, y: y)
            }
            var area: CGFloat = 0
            for i in 0..<pts2D.count {
                let j = (i + 1) % pts2D.count
                area += pts2D[i].x * pts2D[j].y - pts2D[j].x * pts2D[i].y
            }
            return Float(abs(area) * 0.5)
        }

        let topAreaMM2 = planarAreaMM2(tMM)
        let bottomAreaMM2 = planarAreaMM2(bMM)

        // Check if bottom area larger than top, flag it
        if bottomAreaMM2 > topAreaMM2 {
            flags.bottomLargerThanTop = true
        }

        // Check contour mismatch: if distances between top and bottom points are large
        let maxDistThresholdMM: Float = 100.0 // example threshold 10 cm
        var maxDist: Float = 0
        for i in 0..<N {
            let d = simd_length(tMM[i] - bMM[i])
            if d > maxDist { maxDist = d }
        }
        if maxDist > maxDistThresholdMM {
            flags.contourMismatch = true
        }

        // Compute volume in mm^3
        var volMM3: Float = 0
        for i in 0..<N {
            let i0 = i
            let i1 = (i + 1) % N
            let t0 = tMM[i0], t1 = tMM[i1]
            let b0 = bMM[i0], b1 = bMM[i1]
            volMM3 += signedVolumeOfTriangle(t0, b0, t1)
            volMM3 += signedVolumeOfTriangle(t1, b0, b1)
        }
        let topTris = triangulatePolygon(tMM)
        for tri in topTris { volMM3 += signedVolumeOfTriangle(tri.0, tri.1, tri.2) }
        let bottomTris = triangulatePolygon(bMM)
        for tri in bottomTris { volMM3 += signedVolumeOfTriangle(tri.2, tri.1, tri.0) }
        volMM3 = abs(volMM3)

        return (topAreaMM2, bottomAreaMM2, volMM3, flags)
    }

    // MARK: - Metric Validation Helpers
    private func validateAndConvertVolume(_ volume: Float) -> Float? {
        guard volume.isFinite, volume >= 0 else { return nil }

        // All internal volume computations are intended to be in cm^3.
        // Some legacy paths may still return m^3; those values are *tiny* (typically < 1e-3),
        // so we convert them using a conservative heuristic.
        var vCM3 = volume
        if vCM3 > 0, vCM3 < 0.001 {
            vCM3 *= 1_000_000.0
        }

        // Conservative sanity bound; typical wounds should be well below this
        guard vCM3.isFinite, vCM3 < 10_000 else { return nil }
        return vCM3
    }

    // MARK: - Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()
        setupARScene()
        setupGestures()
        setupObservers()
        setupSegmentationPipeline()
    }

    // MARK: - AR Setup

    private func setupARScene() {
        sceneView.delegate = self
        sceneView.session.delegate = self
        sceneView.scene = SCNScene()

        sceneView.autoenablesDefaultLighting = true
        sceneView.automaticallyUpdatesLighting = true
        sceneView.preferredFramesPerSecond = 60

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
        edgeOverlayView.isHidden = true
        edgeOverlayView.alpha = 0.0
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

        // AR configuration with capability checks
        guard ARWorldTrackingConfiguration.isSupported else {
            showHUDPrompt("AR World Tracking not supported on this device")
            return
        }
        // Avoid enabling an already-enabled session
        guard !sessionRunning else { return }

        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]

        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification) {
            config.sceneReconstruction = .meshWithClassification
        } else if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }

        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth) {
            // Not required, but enabling depth semantics can improve depth stability on supported devices
        }
        if #available(iOS 12.0, *) {
            config.environmentTexturing = .automatic
        }

        sceneView.session.run(config)
        sessionRunning = true

        continuousScanningEnabled = true

        // Start display link for edge overlay updates and keep strong reference
        displayLink = CADisplayLink(target: self, selector: #selector(updateEdgeOverlay))
        displayLink?.add(to: .main, forMode: .common)
    }

    // MARK: - Gesture Setup

    /// Returns a world-space point for a screen location by preferring SceneKit mesh hits, then AR raycasts.
    /// - Parameters:
    ///   - screenPoint: The CGPoint on screen.
    ///   - requireMeshHit: If true, only returns a point if SceneKit hit test hits a mesh node.
    /// - Returns: The 3D world coordinate or nil.
        /// Returns a world-space point for a screen location by preferring LiDAR mesh hits (depth-only occlusion nodes),
    /// then falling back to AR raycasts.
    /// - Parameters:
    ///   - screenPoint: The CGPoint on screen.
    ///   - requireMeshHit: If true, only returns a point if a mesh hit is found.
    /// - Returns: The 3D world coordinate or nil.
    private func worldPointFromSceneDepth(_ screenPoint: CGPoint) -> SIMD3<Float>? {
        guard let frame = sceneView.session.currentFrame else { return nil }
        let depthData = frame.sceneDepth ?? frame.smoothedSceneDepth
        guard let depthMap = depthData?.depthMap else { return nil }

        let viewSize = sceneView.bounds.size
        guard viewSize.width > 0, viewSize.height > 0 else { return nil }

        // View point -> normalized view coords
        let normalizedView = CGPoint(x: screenPoint.x / viewSize.width, y: screenPoint.y / viewSize.height)

        // Normalized view -> normalized camera image coords
        let orientation = view.window?.windowScene?.interfaceOrientation ?? .portrait
        let displayToCamera = frame.displayTransform(for: orientation, viewportSize: viewSize).inverted()
        let cameraNorm = normalizedView.applying(displayToCamera)

        let w = CVPixelBufferGetWidth(depthMap)
        let h = CVPixelBufferGetHeight(depthMap)

        let px = Int((cameraNorm.x * CGFloat(w)).rounded(.toNearestOrAwayFromZero))
        let py = Int((cameraNorm.y * CGFloat(h)).rounded(.toNearestOrAwayFromZero))
        guard px >= 0, px < w, py >= 0, py < h else { return nil }

        // Optional confidence gate: reject low-confidence depth.
        if let conf = depthData?.confidenceMap {
            let cw = CVPixelBufferGetWidth(conf)
            let ch = CVPixelBufferGetHeight(conf)
            if cw == w, ch == h {
                CVPixelBufferLockBaseAddress(conf, .readOnly)
                defer { CVPixelBufferUnlockBaseAddress(conf, .readOnly) }
                if let base = CVPixelBufferGetBaseAddress(conf) {
                    let row = base.advanced(by: py * CVPixelBufferGetBytesPerRow(conf))
                    let c = row.load(fromByteOffset: px, as: UInt8.self)
                    if c == 0 { return nil } // 0 = low confidence
                }
            }
        }

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depthMap) else { return nil }

        let row = base.advanced(by: py * CVPixelBufferGetBytesPerRow(depthMap))
        let depthM = row.load(fromByteOffset: px * MemoryLayout<Float>.size, as: Float.self)
        guard depthM.isFinite, depthM > 0.02, depthM < 5.0 else { return nil }

        // Scale intrinsics (defined in camera image space) down to depth map resolution.
        let imageRes = frame.camera.imageResolution
        let sx = Float(w) / Float(imageRes.width)
        let sy = Float(h) / Float(imageRes.height)

        let intr = frame.camera.intrinsics
        let fx = intr.columns.0.x * sx
        let fy = intr.columns.1.y * sy
        let cx = intr.columns.2.x * sx
        let cy = intr.columns.2.y * sy

        // Camera space (ARKit camera looks along -Z)
        let xCam = (Float(px) - cx) / fx * depthM
        let yCam = (cy - Float(py)) / fy * depthM
        let zCam = -depthM

        let camPoint = SIMD4<Float>(xCam, yCam, zCam, 1.0)
        let worldPoint = frame.camera.transform * camPoint
        return SIMD3<Float>(worldPoint.x, worldPoint.y, worldPoint.z)
    }

func worldPointFromScreen(_ screenPoint: CGPoint, requireMeshHit: Bool = false) -> SIMD3<Float>? {
        // 1) Prefer reconstructed mesh hits (best for snapping to real surfaces)
        let meshHitOptions: [SCNHitTestOption: Any] = [
            .searchMode: SCNHitTestSearchMode.closest.rawValue,
            .backFaceCulling: false,
            .ignoreHiddenNodes: false,
            .categoryBitMask: NodeCategory.meshOcclusion
        ]
        if let hit = sceneView.hitTest(screenPoint, options: meshHitOptions).first {
            let p = hit.worldCoordinates
            if p.x.isFinite && p.y.isFinite && p.z.isFinite {
                return SIMD3<Float>(p.x, p.y, p.z)
            }
        }

        // If we explicitly require mesh contact, do not fall back to estimated planes.
        if let depthPoint = worldPointFromSceneDepth(screenPoint) { return depthPoint }
        if requireMeshHit { return nil }

        // 2) Prefer AR raycast against existing plane geometry
        if let query = sceneView.raycastQuery(from: screenPoint, allowing: .existingPlaneGeometry, alignment: .any),
           let result = sceneView.session.raycast(query).first {
            let t = result.worldTransform
            return SIMD3<Float>(t.columns.3.x, t.columns.3.y, t.columns.3.z)
        }

        // 3) Fallback to infinite plane
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

    /// Helper to check if there are any mesh anchors scanned
    private func hasMeshAnchors() -> Bool {
        return !scannedMeshAnchors.isEmpty
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

        // Encourage scanning but do not block placement if mesh is not present yet
        if scannedMeshAnchors.isEmpty {
            showHUDPrompt("Move closer and scan around for better accuracy.")
        }

        let location = gesture.location(in: sceneView)

        let resolver: (CGPoint) -> SIMD3<Float>? = { self.worldPointFromScreen($0, requireMeshHit: true) }

        if isManualOutlineMode { showManualPointOverlays = true }

        // Multi-sample averaging for stability
        let samplesCount = 8
        var collectedPoints: [SIMD3<Float>] = []
        let sampleInterval = 0.02
        var sampleIndex = 0

        Timer.scheduledTimer(withTimeInterval: sampleInterval, repeats: true) { timer in
            sampleIndex += 1
            if let point = resolver(location) {
                collectedPoints.append(point)
            }
            if sampleIndex >= samplesCount {
                timer.invalidate()
                guard !collectedPoints.isEmpty else {
                    self.showHUDPrompt("Aim at a real surface or move closer for better results.")
                    return
                }
                let finalPoint = self.averagePoint(collectedPoints)
                self.handleFinalTapPoint(finalPoint)
            }
        }
    }

    @objc private func handleGuidedPan(_ gesture: UIPanGestureRecognizer) {
        guard guidedActive && isManualOutlineMode else { return }
        let location = gesture.location(in: sceneView)

        if !hasMeshAnchors() {
            // Encourage scanning but allow outlining to proceed on best available estimates
            showHUDPrompt("Keep scanning for best accuracy. You can continue outlining.")
        }

        let requireMesh = true

        switch gesture.state {
        case .began, .changed:
            let currentPoints = currentUserPoints()
            if let last = projectToScreen(currentPoints).last {
                let dx = location.x - last.x
                let dy = location.y - last.y
                let dist = sqrt(dx*dx + dy*dy)
                if dist < 10 { return }
            }
            if let p = worldPointFromScreen(location, requireMeshHit: requireMesh) {
                appendPointToCurrentUserPoints(p)
                addPointToScene(at: p)
                updateFilledPolygon()
                updateMetrics()
                lastGuidedPointTime = CACurrentMediaTime()
                if currentUserPoints().count >= guidedMaxPoints {
                    finishManualOutline()
                }
            }
        case .ended, .cancelled, .failed:
            lastGuidedPointTime = CACurrentMediaTime()
        default:
            break
        }
    }


    /// Helper to compute the average of a list of 3D points
    private func averagePoint(_ points: [SIMD3<Float>]) -> SIMD3<Float> {
        guard !points.isEmpty else { return SIMD3<Float>(0,0,0) }
        var sum = SIMD3<Float>(0,0,0)
        for p in points { sum += p }
        return sum / Float(points.count)
    }

    private func handleFinalTapPoint(_ point: SIMD3<Float>) {
        appendPointToCurrentUserPoints(point)
        addPointToScene(at: point)
        if showManualPointOverlays {
            updateFilledPolygon()
        }
        let currentPoints = currentUserPoints()
        if currentPoints.count >= 3 {
            if let deep = deepestPointInROI(boundary: currentPoints) {
                selectedDepthPoint = deep
            } else {
                selectedDepthPoint = centroid(of: currentPoints)
            }
            updateDepthPointNode()
            updateDepthIndicatorLine()
        }
        updateMetrics()

        lastGuidedPointTime = CACurrentMediaTime()
        if guidedActive && currentPoints.count >= guidedMaxPoints {
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

        NotificationCenter.default.addObserver(self,
                                               selector: #selector(toggleWireframe),
                                               name: NSNotification.Name("ToggleWireframe"),
                                               object: nil)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(toggleClassificationColors),
                                               name: NSNotification.Name("ToggleClassificationColors"),
                                               object: nil)

        NotificationCenter.default.addObserver(self,
                                               selector: #selector(enableAutoSegmentation),
                                               name: NSNotification.Name("EnableAutoSegmentation"),
                                               object: nil)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(disableAutoSegmentation),
                                               name: NSNotification.Name("DisableAutoSegmentation"),
                                               object: nil)

        NotificationCenter.default.addObserver(self,
                                               selector: #selector(runWoundAnalysisNow),
                                               name: NSNotification.Name("RunWoundAnalysisNow"),
                                               object: nil)
/*
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(handleManualOverlaysToggle(_:)),
                                               name: NSNotification.Name("ManualOverlaysChanged"),
                                               object: nil)
*/
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(startMeasureFlow),
                                               name: NSNotification.Name("StartMeasureFlow"),
                                               object: nil)

        NotificationCenter.default.addObserver(self,
                                               selector: #selector(startManualOutline),
                                               name: NSNotification.Name("StartManualOutline"),
                                               object: nil)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(finishManualOutline),
                                               name: NSNotification.Name("FinishManualOutline"),
                                               object: nil)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(clearManualOutline),
                                               name: NSNotification.Name("ClearManualOutline"),
                                               object: nil)

        NotificationCenter.default.addObserver(self,
                                               selector: #selector(disableHUDPrompts),
                                               name: NSNotification.Name("DisableHUDPrompts"),
                                               object: nil)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(enableHUDPrompts),
                                               name: NSNotification.Name("EnableHUDPrompts"),
                                               object: nil)

        NotificationCenter.default.addObserver(self,
                                               selector: #selector(startGuidedWoundMeasurement),
                                               name: NSNotification.Name("StartGuidedWoundMeasurement"),
                                               object: nil)

        // Extra manual measurement / editing notifications
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(startManualOutline),
                                               name: NSNotification.Name("EnterManualMeasurement"),
                                               object: nil)/*
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(manualUndo),
                                               name: NSNotification.Name("ManualUndo"),
                                               object: nil)
        */
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(clearManualOutline),
                                               name: NSNotification.Name("ManualClear"),
                                               object: nil)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(finishManualOutline),
                                               name: NSNotification.Name("ManualFinish"),
                                               object: nil)/*
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(editOuterOutline),
                                               name: NSNotification.Name("ManualEditOuter"),
                                               object: nil)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(editInnerOutline),
                                               name: NSNotification.Name("ManualEditInner"),
                                               object: nil)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(setPlacementModeStrict),
                                               name: NSNotification.Name("SetPlacementModeStrict"),
                                               object: nil)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(setPlacementModeBalanced),
                                               name: NSNotification.Name("SetPlacementModeBalanced"),
                                               object: nil)*/

        NotificationCenter.default.addObserver(self,
                                               selector: #selector(enableContinuousScanning),
                                               name: NSNotification.Name("EnableContinuousScanning"),
                                               object: nil)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(disableContinuousScanning),
                                               name: NSNotification.Name("DisableContinuousScanning"),
                                               object: nil)
        
        // Added generic button notification observers
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(finishManualOutline),
                                               name: NSNotification.Name("Finish"),
                                               object: nil)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(clearManualOutline),
                                               name: NSNotification.Name("Clear"),
                                               object: nil)
    }

    @objc private func enableAutoSegmentation() { isAutoSegmentationEnabled = true }
    @objc private func disableAutoSegmentation() { isAutoSegmentationEnabled = false }

    @objc private func disableHUDPrompts() { hudPromptsEnabled = false; hideHUDPrompt() }
    @objc private func enableHUDPrompts() { hudPromptsEnabled = true }

    @objc private func enableContinuousScanning() {
        continuousScanningEnabled = true
        showHUDPrompt("Continuous scanning: ON")
        updateModeLabel()
    }

    @objc private func disableContinuousScanning() {
        continuousScanningEnabled = true
        showHUDPrompt("Continuous scanning forced ON")
        updateModeLabel()
    }

    // MARK: - HUD Prompt Methods

    private func showHUDPrompt(_ message: String, duration: TimeInterval = 1.5) {
        guard hudPromptsEnabled else { return }
        hudLabel.layer.removeAllAnimations()
        hudLabel.alpha = 0
        hudLabel.text = "  \(message)  "
        UIView.animate(withDuration: 0.2) { self.hudLabel.alpha = 1 }
        DispatchQueue.main.asyncAfter(deadline: .now() + duration) {
            UIView.animate(withDuration: 0.2) { self.hudLabel.alpha = 0 }
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

        showHUDPrompt("Running Wound Analysis…", duration: 1.0)
        let previousManualOverlayState = showManualPointOverlays
        showManualPointOverlays = false
        for n in pointNodes { n.removeFromParentNode() }
        pointNodes.removeAll()
        lineNode?.removeFromParentNode()
        lineNode = nil

        isAutoSegmentationEnabled = true
        isRefiningWound = true
        continuousScanningEnabled = true

        let trackingSettleDuration: TimeInterval = 0.1
        self.segmentationInterval = 1.5
        self.trackingSettleUntil = CACurrentMediaTime() + trackingSettleDuration

        showRefinePrompt()
        DispatchQueue.main.asyncAfter(deadline: .now() + refineDuration) { [weak self] in
            guard let self = self else { return }
            self.isAutoSegmentationEnabled = false
            self.isRefiningWound = false

            self.showManualPointOverlays = previousManualOverlayState

            self.segmentationInterval = 1.0
            self.hideHUDPrompt()
            self.updateMetrics()
        }
    }

    @objc private func startMeasureFlow() {
        continuousScanningEnabled = true
        runWoundAnalysisNow()
        refinePassesRemaining = maxRefinePasses
        stablePolygonCount = 0
    }

    @objc private func startManualOutline() {
        isManualOutlineMode = true
        continuousScanningEnabled = true
        showManualPointOverlays = true
        isAutoSegmentationEnabled = false
        isRefiningWound = false
        outlinePhase.wrappedValue = .top
        NotificationCenter.default.post(name: NSNotification.Name("ARShowOverlays"), object: nil)
        showHUDPrompt("Tap around the wound to outline the top surface")
    }

    @objc private func finishManualOutline() {
        // Fast-path: if both outlines are present, finalize regardless of current phase
        if topOutlinePoints.wrappedValue.count >= 3 && bottomOutlinePoints.wrappedValue.count >= 3 {
            outlinePhase.wrappedValue = .done
            isManualOutlineMode = false
            showManualPointOverlays = false
            guidedActive = false
            guidedAutoFinishTimer?.invalidate(); guidedAutoFinishTimer = nil
            showHUDPrompt("Bottom outline saved")
            updateMetrics()
            DispatchQueue.main.async { self.showFinalResultsMMAndWarnings() }
            return
        }
        
        switch outlinePhase.wrappedValue {
        case .top:
            outlinePhase.wrappedValue = .bottom
            userPointsClearVisuals()
            bottomOutlinePoints.wrappedValue.removeAll()
            showManualPointOverlays = true
            isManualOutlineMode = true
            guidedActive = false
            guidedAutoFinishTimer?.invalidate(); guidedAutoFinishTimer = nil
            showHUDPrompt("Now outline the bottom surface. Tap around the wound.")
        case .bottom:
            outlinePhase.wrappedValue = .done
            isManualOutlineMode = false
            showManualPointOverlays = false
            guidedActive = false
            guidedAutoFinishTimer?.invalidate(); guidedAutoFinishTimer = nil
            showHUDPrompt("Bottom outline saved")
            updateMetrics()
            shouldAnnounceFinalMetrics = true
        case .done:
            break
        case .none:
            break
        }
    }

    @objc private func clearManualOutline() {
        topOutlinePoints.wrappedValue.removeAll()
        bottomOutlinePoints.wrappedValue.removeAll()
        userPointsClearVisuals()
        showManualPointOverlays = false
        showHUDPrompt("Outlines cleared")
        guidedActive = false
        guidedAutoFinishTimer?.invalidate(); guidedAutoFinishTimer = nil
        outlinePhase.wrappedValue = .top
        isManualOutlineMode = true
        showManualPointOverlays = true
    }

    @objc private func startGuidedWoundMeasurement() {
        topOutlinePoints.wrappedValue.removeAll()
        bottomOutlinePoints.wrappedValue.removeAll()
        userPointsClearVisuals()
        depthPointNode?.removeFromParentNode(); depthPointNode = nil
        depthLineNode?.removeFromParentNode(); depthLineNode = nil
        selectedDepthPoint = nil

        isManualOutlineMode = true
        showManualPointOverlays = true
        continuousScanningEnabled = true
        isAutoSegmentationEnabled = false
        isRefiningWound = false
        guidedActive = true
        outlinePhase.wrappedValue = .top

        NotificationCenter.default.post(name: NSNotification.Name("ARShowOverlays"), object: nil)
        showHUDPrompt("Draw a rough circle around the wound. Tap Finish when done.")

        lastGuidedPointTime = CACurrentMediaTime()
        guidedAutoFinishTimer?.invalidate()
        guidedAutoFinishTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] t in
            guard let self = self else { t.invalidate(); return }
            let now = CACurrentMediaTime()
            let count = self.currentUserPoints().count
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

        let iface: UIInterfaceOrientation = {
            if let ws = self.view.window?.windowScene {
                if #available(iOS 18.0, *) { return ws.effectiveGeometry.interfaceOrientation }
                else { return ws.interfaceOrientation }
            }
            return .portrait
        }()
        let _ = UIImage(cgImage: cgImage, scale: 1.0, orientation: iface.toUIImageOrientationForBackCamera())

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
            guard let obs = req.results?.first as? VNPixelBufferObservation else { return }
            let pb = obs.pixelBuffer
            let ciMask = CIImage(cvPixelBuffer: pb)
            guard let cgMask = self.ciContext.createCGImage(ciMask, from: ciMask.extent) else { return }

            let roiScreen = self.projectToScreen(self.currentUserPoints())
            guard roiScreen.count >= 3 else { return }

            let viewSize = self.sceneView.bounds.size
            UIGraphicsBeginImageContextWithOptions(viewSize, false, 1.0)
            guard let ctx = UIGraphicsGetCurrentContext() else { UIGraphicsEndImageContext(); return }
            ctx.interpolationQuality = .low
            ctx.draw(cgMask, in: CGRect(origin: .zero, size: viewSize))

            let path = UIBezierPath()
            if let first = roiScreen.first { path.move(to: first) }
            for p in roiScreen.dropFirst() { path.addLine(to: p) }
            path.close()

            ctx.addPath(path.cgPath); ctx.clip()
            ctx.draw(cgMask, in: CGRect(origin: .zero, size: viewSize))
            let constrained = UIGraphicsGetImageFromCurrentImageContext()?.cgImage
            UIGraphicsEndImageContext()
            guard let constrainedMask = constrained else { return }

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
                if let wp = self.worldPointFromScreen(sp, requireMeshHit: true) {
                    new3D.append(wp)
                }
            }
            DispatchQueue.main.async {
                if new3D.count >= 3 {
                    switch self.outlinePhase.wrappedValue {
                    case .top:
                        self.topOutlinePoints.wrappedValue = new3D
                    case .bottom:
                        self.bottomOutlinePoints.wrappedValue = new3D
                    case .done:
                        break
                    case .none:
                        break
                    }
                    let currentPoints = self.currentUserPoints()
                    if let deep = self.deepestPointInROI(boundary: currentPoints) {
                        self.selectedDepthPoint = deep
                    } else {
                        self.selectedDepthPoint = self.centroid(of: currentPoints)
                    }
                    self.updateDepthPointNode()
                    self.updateDepthIndicatorLine()
                    self.updateFilledPolygon()
                    self.updateMetrics()
                    self.showHUDPrompt(self.hudMessageMeasurementsReady)
                }
            }
        }
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            DispatchQueue.main.async {
                self.showHUDPrompt("Using manual outline (Vision failed)")
                self.updateFilledPolygon(); self.updateMetrics()
            }
        }
    }

    // MARK: - Raycast


    // MARK: - Adding/Connecting Points

    private func addPointToScene(at point: SIMD3<Float>) {
        if showManualPointOverlays {
            let sphereNode = SCNNode(geometry: SCNSphere(radius: 0.005))
            sphereNode.geometry?.firstMaterial?.diffuse.contents = UIColor.red
            sphereNode.position = SCNVector3(point)
            sphereNode.categoryBitMask = NodeCategory.overlays
            sphereNode.renderingOrder = 0
            sceneView.scene.rootNode.addChildNode(sphereNode)
            pointNodes.append(sphereNode)
        }
        autoConnectPoints()
    }

    private func autoConnectPoints() {
        lineNode?.removeFromParentNode()
        let currentPoints = currentUserPoints()
        guard currentPoints.count > 1 else { return }
        guard showManualPointOverlays else { return }

        let vertices = currentPoints.map { SCNVector3($0.x, $0.y, $0.z) }
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
        lineNode?.categoryBitMask = NodeCategory.overlays
        lineNode?.renderingOrder = 0

        sceneView.scene.rootNode.addChildNode(lineNode!)
    }

    // MARK: - Wound Highlight (Filled Polygon)

    private func updateFilledPolygon() {
        filledPolygonNode?.removeFromParentNode()

        let now = CACurrentMediaTime()
        let minInterval = isRefiningWound ? 0.2 : polygonMinInterval
        if now - lastPolygonUpdateTime < minInterval { return }

        let points3D = currentUserPoints()
        let sampleScreen = projectToScreen(points3D)
        if !lastPolygonScreenSample.isEmpty {
            let dist = polygonScreenDistance(sampleScreen, lastPolygonScreenSample)
            if dist < contourChangeThreshold { return }
        }
        lastPolygonScreenSample = sampleScreen
        lastPolygonUpdateTime = now

        let points = points3D
        if points.count > 200 { return }
        guard points.count >= 3 else { return }

        let triangles = triangulatePolygon(points)
        guard !triangles.isEmpty else { return }

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
        node.categoryBitMask = NodeCategory.overlays
        node.renderingOrder = 0
        sceneView.scene.rootNode.addChildNode(node)
        filledPolygonNode = node

        updateMetricsTextNode(perimeter: nil, area: nil, volume: nil)
    }

    private func triangulatePolygon(_ points: [SIMD3<Float>]) -> [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] {
        guard points.count >= 3 else { return [] }

        let normal = polygonNormal(points)
        let (u, v) = planeBasis(from: normal)
        let origin = points[0]

        var pts2D: [CGPoint] = points.map { p in
            let d = p - origin
            let x = CGFloat(simd_dot(d, u))
            let y = CGFloat(simd_dot(d, v))
            return CGPoint(x: x, y: y)
        }

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

                if area(a, b, c) <= 0 { continue }

                var contains = false
                for j in 0..<indices.count where j != i0 && j != i1 && j != i2 {
                    if pointInTriangle(pts2D[j], a, b, c) {
                        contains = true
                        break
                    }
                }
                if contains { continue }

                let pa = points[indices[i0]]
                let pb = points[indices[i1]]
                let pc = points[indices[i2]]
                tris.append((pa, pb, pc))

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
        node.categoryBitMask = NodeCategory.overlays
        node.renderingOrder = 0
        sceneView.scene.rootNode.addChildNode(node)
        depthPointNode = node
    }

    private func updateDepthIndicatorLine() {
        depthLineNode?.removeFromParentNode()
        let boundary = currentUserPoints()
        guard boundary.count >= 3 else { return }
        guard let depth = selectedDepthPoint else { return }
        let n = polygonNormal(boundary)
        let p0 = boundary[0]
        let v = depth - p0
        let distance = simd_dot(v, n)
        let projected = depth - distance * n

        let start = SCNVector3(projected)
        let end = SCNVector3(depth)
        let radius: CGFloat = isRefiningWound ? 0.0025 : 0.0015
        let node = lineNodeBetween(start: start, end: end, radius: radius, color: .blue)
        node.categoryBitMask = NodeCategory.overlays
        node.renderingOrder = 0
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
        let dir = SCNVector3(vector.x / Float(height), vector.y / Float(height), vector.z / Float(height))
        node.look(at: SCNVector3(node.position.x + dir.x, node.position.y + dir.y, node.position.z + dir.z))
        return node
    }

    // MARK: - LiDAR Mesh Visualization

    private func makeSnapshot(from anchor: ARMeshAnchor) -> MeshSnapshot? {
        let geo = anchor.geometry
        let vDesc = geo.vertices
        let fDesc = geo.faces

        let vCount = vDesc.count
        let faceCount = fDesc.count
        guard vCount > 0, faceCount > 0 else { return nil }

        let vStride = vDesc.stride
        let vOffset = vDesc.offset

        // Copy just enough vertex data for count * stride (starting at offset)
        let vNeededBytes = vOffset + (vStride * vCount)
        guard vDesc.buffer.length >= vNeededBytes else { return nil }
        let vData = Data(bytes: vDesc.buffer.contents(), count: vNeededBytes)

        let indexCountPerPrim = fDesc.indexCountPerPrimitive
        let bytesPerIndex = fDesc.bytesPerIndex
        let totalIndexCount = faceCount * indexCountPerPrim
        let iNeededBytes = totalIndexCount * bytesPerIndex
        guard fDesc.buffer.length >= iNeededBytes else { return nil }
        let iData = Data(bytes: fDesc.buffer.contents(), count: iNeededBytes)

        return MeshSnapshot(
            anchorID: anchor.identifier,
            faceCount: faceCount,
            indexCountPerPrimitive: indexCountPerPrim,
            faceBytesPerIndex: bytesPerIndex,
            vertexCount: vCount,
            vertexStride: vStride,
            vertexOffset: vOffset,
            vertexData: vData,
            indexData: iData
        )
    }

    private func updateMeshNode(for anchor: ARMeshAnchor, parentNode: SCNNode? = nil) {
        guard shouldUpdateMesh() else { return }

        // Throttle updates
        let now = CACurrentMediaTime()
        if now - lastMeshUpdateTime < meshUpdateInterval { return }
        lastMeshUpdateTime = now

        guard let snapshot = makeSnapshot(from: anchor) else { return }

        // Prefer anchoring the mesh nodes under ARKit's anchor node to avoid drift as ARKit refines transforms.
        guard let hostNode = parentNode
            ?? meshOcclusionNodesByAnchorID[anchor.identifier]?.parent
            ?? meshNodesByAnchorID[anchor.identifier]?.parent
        else { return }

        meshUpdateQueue.async { [weak self] in
            guard let self = self else { return }
            guard let geom = self.scnGeometry(from: snapshot) else { return }

            // Occlusion geometry (depth-only)
            let occlusionGeom = geom.copy() as? SCNGeometry
            occlusionGeom?.materials = [self.makeOcclusionMaterial()]

            DispatchQueue.main.async {
                // If the host node has been removed, abort.
                if hostNode !== self.sceneView.scene.rootNode && hostNode.parent == nil { return }

                // Visible (debug) mesh node
                if let existing = self.meshNodesByAnchorID[snapshot.anchorID] {
                    existing.geometry = geom
                    if existing.parent !== hostNode {
                        existing.removeFromParentNode()
                        hostNode.addChildNode(existing)
                    }
                } else {
                    let n = SCNNode(geometry: geom)
                    n.renderingOrder = -50
                    n.categoryBitMask = NodeCategory.meshOcclusion
                    n.castsShadow = false
                    n.opacity = 0.0 // keep it invisible; hit-testing still works
                    self.meshNodesByAnchorID[snapshot.anchorID] = n
                    hostNode.addChildNode(n)
                }

                // Depth-only occlusion mesh node (used for hit-testing + visual occlusion)
                if let existingOcc = self.meshOcclusionNodesByAnchorID[snapshot.anchorID] {
                    existingOcc.geometry = occlusionGeom
                    if existingOcc.parent !== hostNode {
                        existingOcc.removeFromParentNode()
                        hostNode.addChildNode(existingOcc)
                    }
                } else if let occlusionGeom {
                    let occ = SCNNode(geometry: occlusionGeom)
                    occ.renderingOrder = -100
                    occ.categoryBitMask = NodeCategory.meshOcclusion
                    occ.castsShadow = false
                    self.meshOcclusionNodesByAnchorID[snapshot.anchorID] = occ
                    hostNode.addChildNode(occ)
                }
            }
        }
    }

    private func makeOcclusionMaterial() -> SCNMaterial {
        let mat = SCNMaterial()
        mat.diffuse.contents = UIColor.clear
        mat.isDoubleSided = true
        mat.readsFromDepthBuffer = true
        mat.writesToDepthBuffer = true
        mat.colorBufferWriteMask = []   // depth-only
        mat.lightingModel = .constant
        return mat
    }

    private func applyMeshVisualizationMaterial(to geometry: SCNGeometry) {
        let material = SCNMaterial()
        if colorMeshByClassification {
            // Prefer per-classification colors when available
            material.diffuse.contents = UIColor.systemTeal.withAlphaComponent(0.35)
        } else {
            material.diffuse.contents = UIColor.gray.withAlphaComponent(0.30)
        }
        material.isDoubleSided = true
        material.readsFromDepthBuffer = true
        material.writesToDepthBuffer = true

        if renderMeshAsWireframe {
            material.fillMode = .lines
        } else {
            material.fillMode = .fill
        }

        geometry.materials = [material]
    }

    private func removeMeshNode(for anchor: ARMeshAnchor) {
        if let node = meshNodesByAnchorID.removeValue(forKey: anchor.identifier) {
            node.removeFromParentNode()
        }
        if let node = meshOcclusionNodesByAnchorID.removeValue(forKey: anchor.identifier) {
            node.removeFromParentNode()
        }
    }

    private func scnGeometry(from snapshot: MeshSnapshot) -> SCNGeometry? {
        // Decode vertices in *anchor-local* space. We attach nodes under the ARKit-provided anchor node so
        // SceneKit keeps them aligned as ARKit refines the mesh anchor transform.
        let vCount = snapshot.vertexCount
        let vStride = snapshot.vertexStride
        let vOffset = snapshot.vertexOffset

        let vNeededBytes = vOffset + (vStride * vCount)
        guard snapshot.vertexData.count >= vNeededBytes else { return nil }

        var localVerts = [SIMD3<Float>]()
        localVerts.reserveCapacity(vCount)

        snapshot.vertexData.withUnsafeBytes { vRaw in
            for i in 0..<vCount {
                let base = vOffset + i * vStride
                let x = vRaw.loadUnaligned(fromByteOffset: base + 0, as: Float.self)
                let y = vRaw.loadUnaligned(fromByteOffset: base + 4, as: Float.self)
                let z = vRaw.loadUnaligned(fromByteOffset: base + 8, as: Float.self)
                localVerts.append(SIMD3<Float>(x, y, z))
            }
        }

        // Decode indices. ARMeshGeometry may use 16-bit indices (very common), so we must respect bytesPerIndex.
        let faceCount = snapshot.faceCount
        let indexCountPerPrim = snapshot.indexCountPerPrimitive
        guard indexCountPerPrim == 3 else { return nil } // triangles only
        let totalIndexCount = faceCount * indexCountPerPrim

        let bytesPerIndex = snapshot.faceBytesPerIndex
        let iNeededBytes = totalIndexCount * bytesPerIndex
        guard snapshot.indexData.count >= iNeededBytes else { return nil }

        var indices = [UInt32]()
        indices.reserveCapacity(totalIndexCount)

        snapshot.indexData.withUnsafeBytes { iRaw in
            if bytesPerIndex == 2 {
                for i in 0..<totalIndexCount {
                    let o = i * 2
                    let v = iRaw.loadUnaligned(fromByteOffset: o, as: UInt16.self)
                    indices.append(UInt32(v))
                }
            } else if bytesPerIndex == 4 {
                for i in 0..<totalIndexCount {
                    let o = i * 4
                    let v = iRaw.loadUnaligned(fromByteOffset: o, as: UInt32.self)
                    indices.append(v)
                }
            } else {
                // Unsupported index width
            }
        }

        guard indices.count == totalIndexCount else { return nil }

        // Filter out degenerate triangles to reduce hit-test weirdness on noisy/partial meshes.
        var filtered = [UInt32]()
        filtered.reserveCapacity(indices.count)

        for tri in stride(from: 0, to: indices.count, by: 3) {
            let ia = Int(indices[tri + 0])
            let ib = Int(indices[tri + 1])
            let ic = Int(indices[tri + 2])

            guard ia < localVerts.count, ib < localVerts.count, ic < localVerts.count else { continue }

            let a = localVerts[ia]
            let b = localVerts[ib]
            let c = localVerts[ic]

            let area2 = simd_length(simd_cross(b - a, c - a))
            if area2.isFinite && area2 > 1e-8 {
                filtered.append(indices[tri + 0])
                filtered.append(indices[tri + 1])
                filtered.append(indices[tri + 2])
            }
        }

        guard !filtered.isEmpty else { return nil }

        let vData = localVerts.withUnsafeBufferPointer { Data(buffer: $0) }
        let vSource = SCNGeometrySource(
            data: vData,
            semantic: .vertex,
            vectorCount: localVerts.count,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: MemoryLayout<SIMD3<Float>>.stride
        )

        let iData = filtered.withUnsafeBufferPointer { Data(buffer: $0) }
        let element = SCNGeometryElement(
            data: iData,
            primitiveType: .triangles,
            primitiveCount: filtered.count / 3,
            bytesPerIndex: MemoryLayout<UInt32>.size
        )

        return SCNGeometry(sources: [vSource], elements: [element])
    }

    // MARK: - Metrics Notification Helper (mm-based)

    // Add helper method for posting robust metrics notification
    private func postRobustMetricsNotification(topMM2: Float?, bottomMM2: Float?, volumeMM3: Float?, flags: VolumeQualityFlags) {
        var info: [String: Any] = [:]
        if let aTop = topMM2 { info["topAreaMM2"] = aTop }
        if let aBot = bottomMM2 { info["bottomAreaMM2"] = aBot }
        if let vol = volumeMM3 { info["volumeMM3"] = vol }
        info["contourMismatch"] = flags.contourMismatch
        info["incompleteOutline"] = flags.incompleteOutline
        info["bottomLargerThanTop"] = flags.bottomLargerThanTop
        info["calibrationUncertainty"] = flags.calibrationUncertainty
        NotificationCenter.default.post(name: NSNotification.Name("RobustMetricsUpdated"), object: nil, userInfo: info)
    }

    // MARK: - Final Results HUD (cm² / cm³) with warnings
    private func showFinalResultsMMAndWarnings() {
        // Gather current outlines
        let top = topOutlinePoints.wrappedValue
        let bottom = bottomOutlinePoints.wrappedValue

        // Compute robust metrics in mm units
        let reg = robustLoftMetricsMM(top: top, bottom: bottom)

        // Convert to cm² and cm³ for display
        func mm2ToCm2(_ mm2: Float?) -> Float? { mm2.map { $0 / 100.0 } }
        func mm3ToCm3(_ mm3: Float?) -> Float? { mm3.map { $0 / 1000.0 } }

        let topAreaCM2 = mm2ToCm2(reg.topAreaMM2)
        let bottomAreaCM2 = mm2ToCm2(reg.bottomAreaMM2)
        let volumeCM3 = mm3ToCm3(reg.volumeMM3)

        let safeVolumeCM3: Float? = {
            guard let v = volumeCM3, v.isFinite, v >= 0, v < 10_000 else { return nil }
            return v
        }()

        // Prefer average area if we have both, otherwise whichever exists
        let displayAreaCM2: Float? = {
            switch (topAreaCM2, bottomAreaCM2) {
            case let (a?, b?): return (a + b) / 2.0
            case let (a?, nil): return a
            case let (nil, b?): return b
            default: return nil
            }
        }()

        // Compose metrics string
        let areaText: String = {
            if let a = displayAreaCM2 { return String(format: "Area: %.2f cm²", a) }
            return "Area: --"
        }()
        let volumeText: String = {
            if let v = safeVolumeCM3 { return String(format: "Vol: %.2f cm³", v) }
            return "Vol: --"
        }()
        var message = areaText + "  |  " + volumeText

        // Append warnings if any flags are set
        var warnings: [String] = []
        if reg.flags.incompleteOutline { warnings.append("Incomplete outline") }
        if reg.flags.contourMismatch { warnings.append("Contours far apart") }
        if reg.flags.bottomLargerThanTop { warnings.append("Bottom > Top") }
        if reg.flags.calibrationUncertainty { warnings.append("Calibration uncertain") }
        if !warnings.isEmpty {
            message += "  •  " + warnings.joined(separator: ", ")
        }

        // Show the HUD prompt with a slightly longer duration for readability
        showHUDPrompt(message, duration: 2.5)

        // We showed the final results explicitly; avoid double-announcement if set elsewhere
        shouldAnnounceFinalMetrics = false
    }

    // MARK: - Metrics

    @objc private func updateMetrics() {
        // Debounce rapid point updates (dragging/outline) to keep UI smooth
        if let workItem = metricsDebounceWorkItem {
            workItem.cancel()
        }

        // Avoid re-processing identical outlines unless the user explicitly requests a "final" readout
        let currentBoundaryHash = userPointsHash()
        if currentBoundaryHash == lastProcessedBoundaryHash, !shouldAnnounceFinalMetrics {
            return
        }
        lastProcessedBoundaryHash = currentBoundaryHash

        let work = DispatchWorkItem { [weak self] in
            guard let self = self else { return }

            let top = self.topOutlinePoints.wrappedValue
            let bottom = self.bottomOutlinePoints.wrappedValue

            // --- Area (cm^2) ---
            let topArea = self.computePlanarArea(top)
            let bottomArea = self.computePlanarArea(bottom)
            let avgArea: Float? = {
                switch (topArea, bottomArea) {
                case let (a?, b?):
                    return max(0, (a + b) / 2)
                case let (a?, nil):
                    return max(0, a)
                case let (nil, b?):
                    return max(0, b)
                default:
                    return nil
                }
            }()

            // --- Perimeter (cm) ---
            let topPerim = self.computePerimeterCM(top)
            let bottomPerim = self.computePerimeterCM(bottom)
            let avgPerim: Float? = {
                switch (topPerim, bottomPerim) {
                case let (p?, q?):
                    return max(0, (p + q) / 2)
                case let (p?, nil):
                    return max(0, p)
                case let (nil, q?):
                    return max(0, q)
                default:
                    return nil
                }
            }()

            // --- Volume (cm^3) ---
            var volumeCM3: Float? = nil
            var flags = VolumeQualityFlags()

            if top.count >= 3 && bottom.count >= 3 {
                // Primary: robust loft between the two traced outlines (matches the user workflow)
                let robust = self.robustLoftMetricsMM(top: top, bottom: bottom)
                flags = robust.flags

                if let vMM3 = robust.volumeMM3 {
                    // mm^3 -> cm^3 (1 cm^3 = 1000 mm^3)
                    volumeCM3 = self.validateAndConvertVolume(vMM3 / 1000.0)
                } else if let v = self.stableLoftedVolumeCM3(top: top, bottom: bottom) {
                    volumeCM3 = self.validateAndConvertVolume(v)
                }

                // Any tracking or mesh sparsity issues should reduce confidence
                if let frame = self.sceneView.session.currentFrame {
                    if case .notAvailable = frame.camera.trackingState {
                        flags.calibrationUncertainty = true
                    } else {
                        // Any limited state increases uncertainty
                        switch frame.camera.trackingState {
                        case .normal:
                            break
                        default:
                            flags.calibrationUncertainty = true
                        }
                    }
                } else {
                    flags.calibrationUncertainty = true
                }
                if self.scannedMeshAnchors.isEmpty {
                    flags.calibrationUncertainty = true
                }

                // Broadcast detailed loft diagnostics (useful for UI debugging and QA)
                self.postRobustMetricsNotification(topMM2: robust.topAreaMM2, bottomMM2: robust.bottomAreaMM2, volumeMM3: robust.volumeMM3, flags: flags)

            } else if let depthPoint = self.selectedDepthPoint,
                      let topAreaCM2 = topArea,
                      top.count >= 3 {

                // Fallback: if user didn't trace the bottom yet but picked a depth point,
                // approximate volume as (top area) * (depth along top normal).
                let depthM = abs(simd_dot(depthPoint - self.centroid(of: top), self.polygonNormal(top)))
                let areaM2 = topAreaCM2 / 10_000.0
                let volM3 = areaM2 * depthM
                let volCM3 = volM3 * 1_000_000.0
                volumeCM3 = self.validateAndConvertVolume(volCM3)
            }

            // Confidence / preliminary state (driven mostly by stability + mesh presence)
            let boundaryForConfidence: [SIMD3<Float>] = {
                if top.count >= 3 { return top }
                if bottom.count >= 3 { return bottom }
                return self.currentUserPoints()
            }()
            let confidence = self.computeConfidence(boundary: boundaryForConfidence)
            let isPreliminary = (self.scannedMeshAnchors.isEmpty || confidence != .high)

            // Output hooks
            self.onMetricsCalculated?(avgPerim ?? 0, avgArea ?? 0, volumeCM3 ?? 0)
            self.onMeasurementState?(MeasurementState(area: avgArea, volume: volumeCM3, isPreliminary: isPreliminary, confidence: confidence))

            // Text overlay in AR
            self.updateMetricsTextNode(perimeter: avgPerim, area: avgArea, volume: volumeCM3)

            // Final readout if requested
            if self.shouldAnnounceFinalMetrics {
                if let v = volumeCM3, v > 0 {
                    self.showHUDPrompt("Final volume: \(String(format: "%.1f", v)) cm³")
                } else if let a = avgArea, a > 0 {
                    self.showHUDPrompt("Final area: \(String(format: "%.1f", a)) cm²")
                }
                self.shouldAnnounceFinalMetrics = false
            }

            // Save once we have non-zero measurements
            if (volumeCM3 ?? 0) > 0 || (avgArea ?? 0) > 0 {
                self.autoSaveScanIfNeeded()
            }

            DispatchQueue.main.async {
                NotificationCenter.default.post(
                    name: .init("MetricsUpdated"),
                    object: nil,
                    userInfo: [
                        "perimeter": avgPerim ?? 0,
                        "area": avgArea ?? 0,
                        "volume": volumeCM3 ?? 0
                    ]
                )
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

    // New helper method for mesh update gate
    private func shouldUpdateMesh() -> Bool {
        // If tracking is completely unavailable, mesh updates tend to be garbage.
        // Otherwise (normal/limited), keep updating so occlusion stays current.
        guard let frame = sceneView.session.currentFrame else { return false }
        switch frame.camera.trackingState {
        case .notAvailable:
            return false
        default:
            return true
        }
    }
    private func autoSaveScanIfNeeded() {
        // Only auto-save once per session/reset
        guard !didAutoSaveCurrentScan else { return }

        // Only save when we actually have something meaningful
        let topCount = topOutlinePoints.wrappedValue.count
        let bottomCount = bottomOutlinePoints.wrappedValue.count
        guard topCount >= 3 || bottomCount >= 3 || !scannedMeshAnchors.isEmpty else { return }

        didAutoSaveCurrentScan = true

        DispatchQueue.global(qos: .utility).async { [weak self] in
            self?.saveCurrentOutlineAndMesh()
        }
    }


    // MARK: - ARSCNViewDelegate

    func renderer(_ renderer: SCNSceneRenderer,
                  didAdd node: SCNNode,
                  for anchor: ARAnchor) {
        guard let meshAnchor = anchor as? ARMeshAnchor else { return }
        guard shouldUpdateMesh() else { return }
        scannedMeshAnchors.append(meshAnchor)
        DispatchQueue.main.async {
            self.updateMeshNode(for: meshAnchor, parentNode: node)
            self.updateMetrics()
        }
    }

    func renderer(_ renderer: SCNSceneRenderer,
                  didUpdate node: SCNNode,
                  for anchor: ARAnchor) {
        guard let meshAnchor = anchor as? ARMeshAnchor else { return }
        guard shouldUpdateMesh() else { return }
        if let index = scannedMeshAnchors.firstIndex(where: { $0.identifier == meshAnchor.identifier }) {
            scannedMeshAnchors[index] = meshAnchor
            let now = CACurrentMediaTime()
            if now - lastMeshUpdateTime > meshUpdateInterval {
                lastMeshUpdateTime = now
                if meshAnchor.geometry.vertices.count > 0 && meshAnchor.geometry.faces.count > 0 {
                    updateMeshNode(for: meshAnchor, parentNode: node)
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
        // Mesh anchor updates are handled primarily via the SceneKit renderer callbacks.
        // Avoid re-triggering metrics updates here to prevent redundant work.
    }

    // MARK: - Reset and Export

    @objc private func resetARScene() {
        print("Resetting AR Scene...")

        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification) {
            config.sceneReconstruction = .meshWithClassification
        } else if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }

        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth) {
            // Not required, but enabling depth semantics can improve depth stability on supported devices
        }
        if #available(iOS 12.0, *) {
            config.environmentTexturing = .automatic
        }

        scannedMeshAnchors.removeAll()
        topOutlinePoints.wrappedValue.removeAll()
        bottomOutlinePoints.wrappedValue.removeAll()
        selectedDepthPoint = nil

        userPointsClearVisuals()
        filledPolygonNode = nil
        depthPointNode = nil

        metricsTextNode?.removeFromParentNode()
        metricsTextNode = nil

        didAutoSaveCurrentScan = false
        sessionRunning = false
        outlinePhase.wrappedValue = .top

        sceneView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        sessionRunning = true

        continuousScanningEnabled = true
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
                    if a == b || b == c || a == c { continue }
                    let ia = a + vertexCount
                    let ib = b + vertexCount
                    let ic = c + vertexCount
                    if ia < 0 || ib < 0 || ic < 0 || ia >= vertices.count || ib >= vertices.count || ic >= vertices.count { continue }
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
        guard edgeOverlayView.window != nil else { return }

        displayLinkFrameCounter += 1
        if displayLinkFrameCounter % 3 != 0 { return }

        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        edgeFilter.inputImage = ciImage
        edgeFilter.intensity = 5.0
        guard let edged = edgeFilter.outputImage else { return }

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

        if let frame = sceneView.session.currentFrame {
            switch frame.camera.trackingState {
            case .normal:
                break
            default:
                if displayLinkFrameCounter % 15 == 0 {
                    let now = CACurrentMediaTime()
                    if now - lastTrackingHUDShownAt > 3.0 {
                        showHUDPrompt("Move slower / improve lighting / point at textured surfaces", duration: 1.0)
                        lastTrackingHUDShownAt = now
                    }
                }
            }
        }

        if scannedMeshAnchors.isEmpty && currentUserPoints().count >= 3 {
            showHUDPrompt(hudMessageMoveCloser)
        }

        if let frame = sceneView.session.currentFrame {
            switch frame.camera.trackingState {
            case .normal: break
            default: return
            }
        }
        if CACurrentMediaTime() < trackingSettleUntil && !continuousScanningEnabled { return }

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
        let ci = CIImage(cvPixelBuffer: mask)
        guard let cgMask = ciContext.createCGImage(ci, from: ci.extent) else { return }

        guard let dataProvider = cgMask.dataProvider,
              let _ = dataProvider.data as Data? else { return }

        let contoursRequest = VNDetectContoursRequest()
        contoursRequest.contrastAdjustment = 1.0
        contoursRequest.detectsDarkOnLight = true
        let handler = VNImageRequestHandler(cgImage: cgMask, options: [:])
        do {
            try handler.perform([contoursRequest])
            guard let observation = contoursRequest.results?.first as? VNContoursObservation else {
                return
            }
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

            let normalizedPoints = contour.normalizedPoints
            var rawScreen: [CGPoint] = []
            let size = self.sceneView.bounds.size
            for p in normalizedPoints {
                let x = CGFloat(p.x) * size.width
                let y = (1 - CGFloat(p.y)) * size.height
                rawScreen.append(CGPoint(x: x, y: y))
            }
            let ds = downsample(points: rawScreen, step: contourDownsampleStep)
            let smooth = rdp(ds, epsilon: rdpEpsilon)
            let cappedSmooth = Array(smooth.prefix(80))

            let newScreen = cappedSmooth
            let distForStability = self.polygonScreenDistance(newScreen, self.lastPolygonScreenSample)

            if self.isApplyingSegmentationResult { return }
            self.isApplyingSegmentationResult = true

            if distForStability < self.contourChangeThreshold {
                self.stablePolygonCount += 1
            } else {
                self.stablePolygonCount = 0
            }
            let isStableEnough = self.stablePolygonCount >= self.requiredStableChecks

            var shouldUpdate = true
            let existing = self.currentUserPoints()
            if existing.count >= 3 {
                shouldUpdate = true
            }

            var new3DPoints: [SIMD3<Float>] = []
            for sp in cappedSmooth {
                if let wp = self.worldPointFromScreen(sp, requireMeshHit: true) {
                    new3DPoints.append(wp)
                }
            }

            DispatchQueue.main.async {
                defer { self.isApplyingSegmentationResult = false }
                if shouldUpdate, new3DPoints.count >= 3, isStableEnough {
                    self.showRefinePrompt()

                    self.lastPolygonScreenSample = newScreen
                    switch self.outlinePhase.wrappedValue {
                    case .top:
                        self.topOutlinePoints.wrappedValue = new3DPoints
                    case .bottom:
                        self.bottomOutlinePoints.wrappedValue = new3DPoints
                    case .done:
                        break
                    case .none:
                        break
                    }
                    self.updateFilledPolygon()

                    if self.scannedMeshAnchors.isEmpty {
                        self.showHUDPrompt(self.hudMessageMoveCloser)
                    }

                    let currentPoints = self.currentUserPoints()
                    if let deep = self.deepestPointInROI(boundary: currentPoints) {
                        self.selectedDepthPoint = deep
                    } else {
                        self.selectedDepthPoint = self.centroid(of: currentPoints)
                    }
                    self.updateDepthPointNode()
                    self.updateDepthIndicatorLine()
                    self.updateMetrics()

                    if !self.scannedMeshAnchors.isEmpty {
                        self.showHUDPrompt(self.hudMessageMeasurementsReady)
                    }
                }
            }
        } catch {
            return
        }
    }

    // MARK: - Toggle Wireframe and Classification Colors

    @objc private func toggleWireframe() {
        renderMeshAsWireframe.toggle()
        for (id, node) in meshNodesByAnchorID {
            if let idx = scannedMeshAnchors.firstIndex(where: { $0.identifier == id }),
               let geom = node.geometry {
                let _ = scannedMeshAnchors[idx]
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
        for (id, _) in meshNodesByAnchorID {
            if let idx = scannedMeshAnchors.firstIndex(where: { $0.identifier == id }) {
                updateMeshNode(for: scannedMeshAnchors[idx])
            }
        }
    }

    // MARK: - Smoothing helpers

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
        let n = polygonNormal(boundary)
        let p0 = boundary[0]
        let (u, v) = planeBasis(from: n)

        func isInside(_ p: SIMD3<Float>) -> Bool {
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
                    let dist = simd_dot(world - p0, n)
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
        var area: CGFloat = 0
        for i in 0..<pts2D.count {
            let j = (i + 1) % pts2D.count
            area += pts2D[i].x * pts2D[j].y - pts2D[j].x * pts2D[i].y
        }
        return Float(abs(area) * 0.5) * 10000.0
    }

    private func computePerimeterCM(_ points: [SIMD3<Float>]) -> Float? {
        guard points.count >= 2 else { return nil }
        var sumM: Float = 0
        for i in 0..<points.count {
            let a = points[i]
            let b = points[(i + 1) % points.count]
            sumM += simd_length(b - a)
        }
        let cm = sumM * 100.0
        return cm.isFinite ? max(0, cm) : nil
    }


    private func computePlanarAreaM2(_ points: [SIMD3<Float>]) -> Float? {
        guard let cm2 = computePlanarArea(points) else { return nil }
        return cm2 / 10_000.0 // cm² -> m²
    }

    private func meanSeparationAlongNormal(top: [SIMD3<Float>], bottom: [SIMD3<Float>]) -> Float? {
        guard let loops = alignedResampledLoops(top: top, bottom: bottom, targetCount: max(top.count, bottom.count)) else { return nil }
        let t = loops.t
        let b = loops.b
        let N = min(t.count, b.count)
        guard N >= 3 else { return nil }
        let n = polygonNormal(t)
        var sum: Float = 0
        for i in 0..<N {
            let d = simd_dot((t[i] - b[i]), n)
            sum += abs(d)
        }
        return sum / Float(N)
    }

    // Stable two-outline volume estimate (returns cm³)
    private func stableLoftedVolumeCM3(top: [SIMD3<Float>], bottom: [SIMD3<Float>]) -> Float? {
        guard top.count >= 3, bottom.count >= 3 else { return nil }
        // Estimate using average area (m²) * mean separation (m)
        let areaTopM2 = computePlanarAreaM2(top)
        let areaBotM2 = computePlanarAreaM2(bottom)
        let avgAreaM2: Float? = {
            switch (areaTopM2, areaBotM2) {
            case let (a?, b?): return (a + b) / 2.0
            case let (a?, nil): return a
            case let (nil, b?): return b
            default: return nil
            }
        }()
        guard let meanSepM = meanSeparationAlongNormal(top: top, bottom: bottom), let areaM2 = avgAreaM2 else { return nil }
        let estM3 = max(0, areaM2 * meanSepM)
        let estCM3 = estM3 * 1_000_000.0

        // Compare with geometric loft (converted to cm³) and pick a conservative/consistent value
        var loftCM3: Float? = nil
        if let loftM3 = loftedVolume(top: top, bottom: bottom) { loftCM3 = loftM3 * 1_000_000.0 }
        guard let loft = loftCM3 else { return estCM3 }

        // If within a factor of 3, blend; otherwise, take the smaller to avoid spikes
        let maxVal = max(loft, estCM3)
        let minVal = min(loft, estCM3)
        if minVal > 0, maxVal / minVal <= 3.0 {
            return 0.5 * (loft + estCM3)
        } else {
            return minVal
        }
    }

    private func projectToScreen(_ points: [SIMD3<Float>]) -> [CGPoint] {
        guard let _ = sceneView.pointOfView else { return [] }
        let projected = points.compactMap { p -> CGPoint? in
            let scnPos = SCNVector3(p.x, p.y, p.z)
            let projectedVec = sceneView.projectPoint(scnPos)
            if projectedVec.z < 1 {
                return CGPoint(x: CGFloat(projectedVec.x), y: CGFloat(projectedVec.y))
            } else {
                return nil
            }
        }
        return projected
    }

    private func polygonScreenDistance(_ a: [CGPoint], _ b: [CGPoint]) -> CGFloat {
        guard !a.isEmpty, !b.isEmpty else { return .greatestFiniteMagnitude }
        let n = min(a.count, b.count)
        var total: CGFloat = 0
        for i in 0..<n { total += hypot(a[i].x - b[i].x, a[i].y - b[i].y) }
        return total / CGFloat(n)
    }

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
        let currentPoints = currentUserPoints()
        guard currentPoints.count >= 3 else {
            metricsTextNode?.removeFromParentNode()
            metricsTextNode = nil
            return
        }
        let c = centroid(of: currentPoints)
        let text = SCNText(string: formattedMetricsText(area: area, volume: volume), extrusionDepth: 0.001)
        text.font = UIFont.systemFont(ofSize: 0.02, weight: .semibold)
        text.firstMaterial?.diffuse.contents = UIColor.white
        text.firstMaterial?.isDoubleSided = true
        let node = SCNNode(geometry: text)
        let scale: Float = 0.01
        node.scale = SCNVector3(scale, scale, scale)
        let offset = SIMD3<Float>(0, 0.01, 0)
        node.position = SCNVector3(c + offset)
        node.categoryBitMask = NodeCategory.overlays
        node.renderingOrder = 0
        if sceneView.pointOfView != nil {
            node.constraints = [SCNBillboardConstraint()]
        }
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

        let outlineTopURL = exportDirectory.appendingPathComponent("WoundOutlineTop.json")
        let outlineBottomURL = exportDirectory.appendingPathComponent("WoundOutlineBottom.json")
        let outlineTop = topOutlinePoints.wrappedValue.map { ["x": $0.x, "y": $0.y, "z": $0.z] }
        let outlineBottom = bottomOutlinePoints.wrappedValue.map { ["x": $0.x, "y": $0.y, "z": $0.z] }
        if let jsonTop = try? JSONSerialization.data(withJSONObject: outlineTop, options: [.prettyPrinted]) {
            try? jsonTop.write(to: outlineTopURL)
        }
        if let jsonBottom = try? JSONSerialization.data(withJSONObject: outlineBottom, options: [.prettyPrinted]) {
            try? jsonBottom.write(to: outlineBottomURL)
        }
        let plyURL = exportDirectory.appendingPathComponent("WoundScan.ply")
        self.exportMeshToPLY(to: plyURL)
    }

    // MARK: - Helpers for multiple outlines

    private func currentUserPoints() -> [SIMD3<Float>] {
        switch outlinePhase.wrappedValue {
        case .top:
            return topOutlinePoints.wrappedValue
        case .bottom:
            return bottomOutlinePoints.wrappedValue
        case .done:
            return combinedUserPoints()
        case .none:
            return []
        }
    }

    private func appendPointToCurrentUserPoints(_ point: SIMD3<Float>) {
        switch outlinePhase.wrappedValue {
        case .top:
            topOutlinePoints.wrappedValue.append(point)
        case .bottom:
            bottomOutlinePoints.wrappedValue.append(point)
        case .done:
            break
        case .none:
            break
        }
    }

    private func userPointsClearVisuals() {
        for n in pointNodes { n.removeFromParentNode() }
        pointNodes.removeAll()
        lineNode?.removeFromParentNode()
        lineNode = nil
        filledPolygonNode?.removeFromParentNode()
        filledPolygonNode = nil
    }

    private func combinedUserPoints() -> [SIMD3<Float>] {
        var combined = [SIMD3<Float>]()
        combined.append(contentsOf: topOutlinePoints.wrappedValue)
        combined.append(contentsOf: bottomOutlinePoints.wrappedValue)
        return combined
    }


    /// Stable-ish hash for outlines + selected depth point.
    /// Quantizes coordinates to reduce noise-triggered recomputation.
    private func userPointsHash() -> Int {
        var hasher = Hasher()

        // Include phase so switching top/bottom triggers an update even if points are identical.
        hasher.combine(outlinePhase.wrappedValue)

        func combinePoint(_ p: SIMD3<Float>) {
            // Quantize to ~1 mm in world space (meters * 1000)
            let qx = Int((p.x * 1000.0).rounded())
            let qy = Int((p.y * 1000.0).rounded())
            let qz = Int((p.z * 1000.0).rounded())
            hasher.combine(qx)
            hasher.combine(qy)
            hasher.combine(qz)
        }

        for p in topOutlinePoints.wrappedValue { combinePoint(p) }
        hasher.combine(topOutlinePoints.wrappedValue.count)
        for p in bottomOutlinePoints.wrappedValue { combinePoint(p) }
        hasher.combine(bottomOutlinePoints.wrappedValue.count)

        if let d = selectedDepthPoint {
            combinePoint(d)
        } else {
            hasher.combine(0 as Int)
        }

        return hasher.finalize()
    }


    private func updateModeLabel() {
        var text = ""
        switch outlinePhase.wrappedValue {
        case .top: text = "Mode: Outlining Top"
        case .bottom: text = "Mode: Outlining Bottom"
        case .done: text = "Mode: Measurement Complete"
        case .none: text = "Mode: Idle"
        }
        let scanText = continuousScanningEnabled ? "Scanning: Continuous" : "Scanning: On demand"
        showHUDPrompt("\(text)  |  \(scanText)", duration: 1.0)
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

