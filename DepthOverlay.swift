import SwiftUI

struct DepthOverlay: View {
    @ObservedObject var manager: CameraManager
    @State private var opacity: Float
    @Binding var maxDepth: Float
    @Binding var minDepth: Float

    init(manager: CameraManager, opacity: Float = 0.6, maxDepth: Binding<Float>, minDepth: Binding<Float>) {
        self.manager = manager
        self._opacity = State(initialValue: opacity)
        self._maxDepth = maxDepth
        self._minDepth = minDepth
    }
    
    var body: some View {
        if manager.isDataAvailable {
            VStack {
                Slider(value: $opacity, in: 0...1, label: {
                    Text("Opacity: \(opacity, specifier: "%.2f")")
                })
                .padding()

                ZStack {
                    // Background visualization
                    MetalTextureViewColorWrapper(
                        rotationAngle: 0,
                        capturedData: manager.capturedData
                    )
                    .frame(height: 300)
                    .opacity(Double(opacity))

                    // Depth visualization with light shading
                    MetalTextureColorThresholdDepthWrapper(
                        rotationAngle: 0,
                        maxDepth: $maxDepth,
                        minDepth: $minDepth,
                        capturedData: manager.capturedData
                    )
                    .frame(height: 300)
                    .opacity(Double(opacity))
                    .overlay(Rectangle().fill(Color.red.opacity(0.3))) 
                }
            }
            .padding()
        } else {
            Text("No data available.")
                .padding()
        }
    }
}
