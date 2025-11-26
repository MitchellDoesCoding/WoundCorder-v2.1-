import Foundation
import ARKit
import simd

class LiDARVolumeCalculator {
    var scannedMeshAnchors: [ARMeshAnchor] = []
    var woundBoundaryPoints: [SIMD3<Float>] = []
    var userSelectedPoint: SIMD3<Float>? = nil
    private var lastCalculatedMetrics: (Float, Float, Float)? = nil
    var calibrationScale: Float = 1.0

    init() {}

    // MARK: - Public methods

    /// Calculate perimeter (cm), area (cm^2) and volume (cm^3) based on current scannedMeshAnchors and woundBoundaryPoints.
    /// - Parameters:
    ///   - session: ARSession (unused here but part of interface)
    ///   - cameraPosition: Camera position in world space (unused but part of interface)
    ///   - completion: Callback with (perimeter, area, volume)
    func calculateMetrics(from session: ARSession, cameraPosition: SIMD3<Float>, completion: @escaping (Float, Float, Float) -> Void) {
        guard woundBoundaryPoints.count >= 3 else {
            completion(0, 0, 0)
            return
        }

        let perimeterCm = calculatePerimeterFromBoundary()
        let areaCm2 = calculateAreaFromBoundary()
        let volumeCm3 = calculateVolumeInteriorIntegration()

        let newMetrics = (perimeterCm, areaCm2, volumeCm3)
        if hasSignificantChange(newMetrics) || lastCalculatedMetrics == nil {
            lastCalculatedMetrics = newMetrics
            completion(perimeterCm, areaCm2, volumeCm3)
        }
    }

    /// Calibrate scale with known length in meters.
    /// Scale is multiplier for distances to get real-world scale.
    func calibrateWithKnownLength(_ knownLengthInM: Float) {
        guard woundBoundaryPoints.count >= 2 else { return }
        let dist = scaledDistance(woundBoundaryPoints[0], woundBoundaryPoints[1])
        guard dist > 0 else { return }
        calibrationScale = knownLengthInM / dist
    }

    /// Distance between two points scaled by calibrationScale.
    func scaledDistance(_ p1: SIMD3<Float>, _ p2: SIMD3<Float>) -> Float {
        let d = length(p1 - p2)
        return d * calibrationScale
    }

    /// Perimeter of wound boundary scaled by calibrationScale and returned in cm.
    func scaledPerimeter() -> Float {
        let perimeterM = calculatePerimeterFromBoundary() / 100.0 // Convert cm to m
        return perimeterM * 100.0 * calibrationScale // cm scaled (same as perimeterCm * calibrationScale)
    }

    // MARK: - Private helpers

    /// Smooth boundary points by Chaikin's algorithm over multiple iterations.
    private func smoothBoundaryPoints(_ points: [SIMD3<Float>], iterations: Int) -> [SIMD3<Float>] {
        guard points.count >= 3 else { return points }
        var smoothed = points
        for _ in 0..<iterations {
            smoothed = chaikinSmoothOnce(smoothed)
        }
        return smoothed
    }

    /// One iteration of Chaikin smoothing on closed loop.
    private func chaikinSmoothOnce(_ points: [SIMD3<Float>]) -> [SIMD3<Float>] {
        let n = points.count
        guard n >= 3 else { return points }
        var newPoints: [SIMD3<Float>] = []
        for i in 0..<n {
            let p0 = points[i]
            let p1 = points[(i + 1) % n]
            let q = p0 * 0.75 + p1 * 0.25
            let r = p0 * 0.25 + p1 * 0.75
            newPoints.append(q)
            newPoints.append(r)
        }
        return newPoints
    }

    /// Calculate perimeter from boundary points in cm.
    private func calculatePerimeterFromBoundary() -> Float {
        let count = woundBoundaryPoints.count
        guard count >= 2 else { return 0 }
        var perimeter: Float = 0
        for i in 0..<count {
            let p0 = woundBoundaryPoints[i]
            let p1 = woundBoundaryPoints[(i + 1) % count]
            let dist = length(p1 - p0)
            if dist.isFinite {
                perimeter += dist
            }
        }
        perimeter *= calibrationScale
        return perimeter * 100 // meters to cm
    }

    /// Calculate area from boundary points projected to best-fit plane in cm^2.
    private func calculateAreaFromBoundary() -> Float {
        guard woundBoundaryPoints.count >= 3 else { return 0 }
        let normal = bestFitNormal(points: woundBoundaryPoints)
        let origin = woundBoundaryPoints[0]
        let (u, v) = planeBasis(from: normal)
        var projectedPoints: [SIMD2<Float>] = []
        for p in woundBoundaryPoints {
            let pt = projectToPlaneSpace(p: p, origin: origin, u: u, v: v)
            projectedPoints.append(pt)
        }
        let areaMeters2 = abs(shoelaceArea2D(projectedPoints))
        let areaScaled = areaMeters2 * calibrationScale * calibrationScale
        return areaScaled * 10_000 // m^2 to cm^2
    }

    /// Calculate volume by integrating interior depth over the wound area clipped to mesh anchors.
    /// Output volume in cm³.
    private func calculateVolumeInteriorIntegration() -> Float {
        guard woundBoundaryPoints.count >= 3 else { return 0 }
        guard !scannedMeshAnchors.isEmpty else { return 0 }

        // Prepare rim plane and basis
        let boundary = woundBoundaryPoints
        let origin = boundary[0]
        let normal = bestFitNormal(points: boundary)
        let (u, v) = planeBasis(from: normal)

        // Project boundary to 2D plane space (meters, unscaled)
        var roiPolygon2D: [SIMD2<Float>] = []
        for p in boundary {
            let pt = projectToPlaneSpace(p: p, origin: origin, u: u, v: v)
            roiPolygon2D.append(pt)
        }

        var totalVolumeMeters3: Float = 0

        // For each ARMeshAnchor
        for anchor in scannedMeshAnchors {
            let geometry = anchor.geometry
            if geometry.vertices.count == 0 || geometry.faces.count == 0 { continue }

            // Access vertices buffer
            let vertexBuffer = geometry.vertices.buffer.contents()
            let vertexStride = geometry.vertices.stride
            let vertexOffset = geometry.vertices.offset

            // Prepare transformed vertices array in world space
            var verticesWorld: [SIMD3<Float>] = Array(repeating: SIMD3<Float>(0,0,0), count: geometry.vertices.count)
            let transform = anchor.transform

            for i in 0..<geometry.vertices.count {
                let vertexPtr = vertexBuffer.advanced(by: i * vertexStride + vertexOffset)
                let vertex = vertexPtr.bindMemory(to: (Float).self, capacity: 3)
                let v = SIMD3<Float>(vertex[0], vertex[1], vertex[2])
                let v4 = transform * SIMD4<Float>(v, 1)
                verticesWorld[i] = SIMD3<Float>(v4.x, v4.y, v4.z)
            }

            // Access faces buffer via Data and iterate indices
            let facesDesc = geometry.faces
            let faceCount = facesDesc.count
            let indexCountPerPrimitive = facesDesc.indexCountPerPrimitive
            let totalIndexCount = faceCount * indexCountPerPrimitive
            let indexBuffer = facesDesc.buffer
            let neededIndexBytes = totalIndexCount * MemoryLayout<UInt32>.size
            let iData = Data(bytesNoCopy: indexBuffer.contents(), count: neededIndexBytes, deallocator: .none)

            iData.withUnsafeBytes { raw in
                let u32 = raw.bindMemory(to: UInt32.self)
                guard u32.count >= totalIndexCount else { return }
                for faceIndex in 0..<faceCount {
                    let base = faceIndex * indexCountPerPrimitive
                    // Expecting triangles
                    if indexCountPerPrimitive != 3 { continue }
                    let idx0 = Int(u32[base + 0])
                    let idx1 = Int(u32[base + 1])
                    let idx2 = Int(u32[base + 2])
                    if idx0 >= verticesWorld.count || idx1 >= verticesWorld.count || idx2 >= verticesWorld.count { continue }

                    let p0 = verticesWorld[idx0]
                    let p1 = verticesWorld[idx1]
                    let p2 = verticesWorld[idx2]

                    // Project each vertex to 2D plane space and get depth
                    let projected0 = projectToPlaneSpace(p: p0, origin: origin, u: u, v: v)
                    let projected1 = projectToPlaneSpace(p: p1, origin: origin, u: u, v: v)
                    let projected2 = projectToPlaneSpace(p: p2, origin: origin, u: u, v: v)

                    let d0 = dot(p0 - origin, normal)
                    let d1 = dot(p1 - origin, normal)
                    let d2 = dot(p2 - origin, normal)

                    var polygon: [(pt: SIMD2<Float>, depth: Float)] = [
                        (projected0, d0),
                        (projected1, d1),
                        (projected2, d2),
                    ]

                    // Clip polygon to ROI polygon (2D)
                    let clippedPolygon = sutherlandHodgman(subject: polygon, clip: roiPolygon2D)
                    if clippedPolygon.count < 3 { continue }

                    // Triangulate clipped polygon fan-wise
                    let v0 = clippedPolygon[0]
                    for i in 1..<(clippedPolygon.count - 1) {
                        let v1 = clippedPolygon[i]
                        let v2 = clippedPolygon[i + 1]
                        let area = triArea2D(v0.pt, v1.pt, v2.pt)
                        if area <= 0 { continue }
                        // Average depth below rim (negative depth)
                        let depths = [v0.depth, v1.depth, v2.depth]
                        let depthsBelowRim = depths.map { max(0, -$0) }
                        let avgDepth = (depthsBelowRim[0] + depthsBelowRim[1] + depthsBelowRim[2]) / 3
                        if avgDepth <= 0 { continue }
                        totalVolumeMeters3 += area * avgDepth
                    }
                }
            }
        }

        // Convert to cm^3 and apply calibration scale for volume (scale^3)
        let scaledVolumeMeters3 = totalVolumeMeters3 * calibrationScale * calibrationScale * calibrationScale
        let volumeCm3 = scaledVolumeMeters3 * 1_000_000 // m^3 to cm^3

        return volumeCm3
    }

    /// Simple fallback volume calculation, currently returns 0.
    private func calculateVolumeSimple() -> Float {
        return 0
    }

    /// Returns true if newMetrics differ significantly from lastCalculatedMetrics.
    private func hasSignificantChange(_ newMetrics: (Float, Float, Float)) -> Bool {
        guard let last = lastCalculatedMetrics else { return true }
        let deltaPerimeter = abs(newMetrics.0 - last.0)
        let deltaArea = abs(newMetrics.1 - last.1)
        let deltaVolume = abs(newMetrics.2 - last.2)
        return deltaPerimeter > 0.1 || deltaArea > 0.5 || deltaVolume > 0.5
    }

    /// Compute best fit plane normal from points via PCA.
    private func bestFitNormal(points: [SIMD3<Float>]) -> SIMD3<Float> {
        guard points.count >= 3 else { return SIMD3<Float>(0, 1, 0) }
        let n = Float(points.count)
        let mean = points.reduce(SIMD3<Float>(0,0,0), +) / n
        var cov = simd_float3x3(0)
        for p in points {
            let d = p - mean
            cov.columns.0 += d * SIMD3<Float>(d.x, d.x, d.x)
            cov.columns.1 += d * SIMD3<Float>(d.y, d.y, d.y)
            cov.columns.2 += d * SIMD3<Float>(d.z, d.z, d.z)
        }
        // Better covariance matrix
        cov.columns.0 = SIMD3<Float>(
            points.map { pow($0.x - mean.x, 2) }.reduce(0, +)/n,
            points.map { ($0.x - mean.x)*($0.y - mean.y) }.reduce(0, +)/n,
            points.map { ($0.x - mean.x)*($0.z - mean.z) }.reduce(0, +)/n
        )
        cov.columns.1 = SIMD3<Float>(
            points.map { ($0.y - mean.y)*($0.x - mean.x) }.reduce(0, +)/n,
            points.map { pow($0.y - mean.y, 2) }.reduce(0, +)/n,
            points.map { ($0.y - mean.y)*($0.z - mean.z) }.reduce(0, +)/n
        )
        cov.columns.2 = SIMD3<Float>(
            points.map { ($0.z - mean.z)*($0.x - mean.x) }.reduce(0, +)/n,
            points.map { ($0.z - mean.z)*($0.y - mean.y) }.reduce(0, +)/n,
            points.map { pow($0.z - mean.z, 2) }.reduce(0, +)/n
        )

        // Compute eigenvectors of cov, normal = eigenvector with smallest eigenvalue
        let (_, vectors) = eigenDecomposition(cov)
        let normal = vectors[0] // smallest eigenvalue eigenvector
        return normalize(normal)
    }

    /// Compute signed area of 2D polygon with shoelace formula.
    private func shoelaceArea2D(_ pts: [SIMD2<Float>]) -> Float {
        let n = pts.count
        guard n >= 3 else { return 0 }
        var sum: Float = 0
        for i in 0..<n {
            let j = (i + 1) % n
            sum += pts[i].x * pts[j].y - pts[j].x * pts[i].y
        }
        return sum * 0.5
    }

    /// Distance from point to plane.
    private func pointToPlaneDistance(point: SIMD3<Float>, planeOrigin: SIMD3<Float>, planeNormal: SIMD3<Float>) -> Float {
        return dot(point - planeOrigin, planeNormal)
    }

    /// Compute perimeter in meters ignoring calibration scale.
    private func perimeterInMeters() -> Float {
        let count = woundBoundaryPoints.count
        guard count >= 2 else { return 0 }
        var perimeter: Float = 0
        for i in 0..<count {
            let p0 = woundBoundaryPoints[i]
            let p1 = woundBoundaryPoints[(i + 1) % count]
            let dist = length(p1 - p0)
            if dist.isFinite {
                perimeter += dist
            }
        }
        return perimeter
    }

    // MARK: - Helper math functions

    /// Create orthonormal basis (u, v) from normal vector.
    private func planeBasis(from normal: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) {
        let n = normalize(normal)
        var u: SIMD3<Float>
        if abs(n.x) > abs(n.z) {
            u = SIMD3<Float>(-n.y, n.x, 0)
        } else {
            u = SIMD3<Float>(0, -n.z, n.y)
        }
        u = normalize(u)
        let v = cross(n, u)
        return (u, v)
    }

    /// Project 3D point into 2D plane space.
    private func projectToPlaneSpace(p: SIMD3<Float>, origin: SIMD3<Float>, u: SIMD3<Float>, v: SIMD3<Float>) -> SIMD2<Float> {
        let d = p - origin
        return SIMD2<Float>(dot(d, u), dot(d, v))
    }

    /// Sutherland-Hodgman polygon clipping.
    /// Subject polygon: array of (point, depth)
    /// Clip polygon: array of points
    private func sutherlandHodgman(subject: [(pt: SIMD2<Float>, depth: Float)], clip: [SIMD2<Float>]) -> [(pt: SIMD2<Float>, depth: Float)] {
        guard subject.count > 0, clip.count >= 3 else { return [] }

        var output = subject

        for i in 0..<clip.count {
            let clipA = clip[i]
            let clipB = clip[(i + 1) % clip.count]
            var input = output
            output = []

            if input.isEmpty { break }

            for j in 0..<input.count {
                let current = input[j]
                let prev = input[(j + input.count - 1) % input.count]

                let currentInside = isInside(point: current.pt, edgeA: clipA, edgeB: clipB)
                let prevInside = isInside(point: prev.pt, edgeA: clipA, edgeB: clipB)

                if currentInside {
                    if !prevInside {
                        if let intersect = intersectEdge(p1: prev, p2: current, clipA: clipA, clipB: clipB) {
                            output.append(intersect)
                        }
                    }
                    output.append(current)
                } else if prevInside {
                    if let intersect = intersectEdge(p1: prev, p2: current, clipA: clipA, clipB: clipB) {
                        output.append(intersect)
                    }
                }
            }
        }

        return output
    }

    /// Check if point is inside edge defined by edgeA -> edgeB (left side)
    private func isInside(point: SIMD2<Float>, edgeA: SIMD2<Float>, edgeB: SIMD2<Float>) -> Bool {
        let edge = edgeB - edgeA
        let toPoint = point - edgeA
        return cross2(edge, toPoint) >= 0
    }

    /// 2D cross product scalar result
    private func cross2(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
        return a.x * b.y - a.y * b.x
    }

    /// Compute intersection point of segment p1-p2 with clip edge clipA-clipB.
    /// Line segment between p1.pt and p2.pt with associated depths.
    private func intersectEdge(p1: (pt: SIMD2<Float>, depth: Float), p2: (pt: SIMD2<Float>, depth: Float), clipA: SIMD2<Float>, clipB: SIMD2<Float>) -> (pt: SIMD2<Float>, depth: Float)? {
        let segDir = p2.pt - p1.pt
        let edgeDir = clipB - clipA

        let denom = cross2(segDir, edgeDir)
        if abs(denom) < 1e-6 { return nil } // Parallel lines

        let t = cross2(clipA - p1.pt, edgeDir) / denom
        if t < 0 || t > 1 { return nil }

        let intersectionPt = p1.pt + segDir * t
        let depthInterp = p1.depth + (p2.depth - p1.depth) * t
        return (intersectionPt, depthInterp)
    }

    /// Triangle area (signed) in 2D
    private func triArea2D(_ a: SIMD2<Float>, _ b: SIMD2<Float>, _ c: SIMD2<Float>) -> Float {
        return 0.5 * ((b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x))
    }

    /// Eigen decomposition of symmetric 3x3 matrix using Jacobi method.
    /// Returns eigenvalues sorted ascending and eigenvectors columns corresponding.
    private func eigenDecomposition(_ matrix: simd_float3x3) -> ([Float], [SIMD3<Float>]) {
        // Jacobi eigenvalue algorithm
        var a = matrix
        var v = simd_float3x3(1)
        let maxIter = 50
        let epsilon: Float = 1e-10

        func maxOffDiagonal(_ m: simd_float3x3) -> (Float, Int, Int) {
            var maxVal: Float = 0
            var p = 0
            var q = 1
            for i in 0..<3 {
                for j in i+1..<3 {
                    let val = abs(m[i, j])
                    if val > maxVal {
                        maxVal = val
                        p = i
                        q = j
                    }
                }
            }
            return (maxVal, p, q)
        }

        for _ in 0..<maxIter {
            let (maxVal, p, q) = maxOffDiagonal(a)
            if maxVal < epsilon { break }

            let app = a[p,p]
            let aqq = a[q,q]
            let apq = a[p,q]

            let phi: Float = 0.5 * atan2(2*apq, aqq - app)
            let c = cos(phi)
            let s = sin(phi)

            // Rotate matrix
            for i in 0..<3 {
                let aip = a[i,p]
                let aiq = a[i,q]
                a[i,p] = c*aip - s*aiq
                a[i,q] = s*aip + c*aiq
            }
            for i in 0..<3 {
                let api = a[p,i]
                let aqi = a[q,i]
                a[p,i] = c*api - s*aqi
                a[q,i] = s*api + c*aqi
            }
            a[p,p] = c*c*app - 2*s*c*apq + s*s*aqq
            a[q,q] = s*s*app + 2*s*c*apq + c*c*aqq
            a[p,q] = 0
            a[q,p] = 0

            // Rotate eigenvectors
            for i in 0..<3 {
                let vip = v[i,p]
                let viq = v[i,q]
                v[i,p] = c*vip - s*viq
                v[i,q] = s*vip + c*viq
            }
        }

        var eigenvals = [a[0,0], a[1,1], a[2,2]]
        var eigenvecs: [SIMD3<Float>] = [
            SIMD3<Float>(v[0,0], v[1,0], v[2,0]),
            SIMD3<Float>(v[0,1], v[1,1], v[2,1]),
            SIMD3<Float>(v[0,2], v[1,2], v[2,2])
        ]

        // Sort ascending eigenvalues and eigenvectors
        let sorted = zip(eigenvals, eigenvecs).sorted { $0.0 < $1.0 }
        eigenvals = sorted.map { $0.0 }
        eigenvecs = sorted.map { normalize($0.1) }

        return (eigenvals, eigenvecs)
    }
}
