import CoreVideo

struct CameraCapturedData {
    var depth: CVPixelBuffer?
    var colorY: CVPixelBuffer?
    var colorCbCr: CVPixelBuffer?
}
