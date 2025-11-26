import Foundation
import AVFoundation
import Metal

extension CVPixelBuffer {
    func texture(withFormat pixelFormat: MTLPixelFormat, planeIndex: Int, addToCache cache: CVMetalTextureCache) -> MTLTexture? {
        let width = CVPixelBufferGetWidthOfPlane(self, planeIndex)
        let height = CVPixelBufferGetHeightOfPlane(self, planeIndex)
        
        var cvtexture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, cache, self, nil, pixelFormat, width, height, planeIndex, &cvtexture)
        guard let texture = cvtexture else { return nil }
        return CVMetalTextureGetTexture(texture)
    }
}

private extension CVPixelBuffer {
    func pixelBufferValue(at position: CGPoint) -> Float {
        CVPixelBufferLockBaseAddress(self, .readOnly)
        let rowData = CVPixelBufferGetBaseAddress(self)! + Int(position.y) * CVPixelBufferGetBytesPerRow(self)
        let value = rowData.assumingMemoryBound(to: Float.self)[Int(position.x)]
        CVPixelBufferUnlockBaseAddress(self, .readOnly)
        return value
    }
}

