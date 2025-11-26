import AVFoundation
import Combine
import Photos

class CameraManager: NSObject, ObservableObject, AVCapturePhotoCaptureDelegate {
    @Published var capturedData = CameraCapturedData()
    @Published var isDataAvailable: Bool = false

    public let captureSession: AVCaptureSession
    private var photoOutput: AVCapturePhotoOutput
    private var depthOutput: AVCaptureDepthDataOutput

    override init() {
        self.captureSession = AVCaptureSession()
        self.photoOutput = AVCapturePhotoOutput()
        self.depthOutput = AVCaptureDepthDataOutput()
        super.init()

        setupSession()
    }

    private func setupSession() {
        captureSession.beginConfiguration()

        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("Unable to access the camera.")
            return
        }

        do {
            let videoDeviceInput = try AVCaptureDeviceInput(device: videoDevice)
            if captureSession.canAddInput(videoDeviceInput) {
                captureSession.addInput(videoDeviceInput)
            }

            if captureSession.canAddOutput(photoOutput) {
                captureSession.addOutput(photoOutput)
                // Removed deprecated isHighResolutionCaptureEnabled
            }

            if captureSession.canAddOutput(depthOutput) {
                captureSession.addOutput(depthOutput)
                depthOutput.setDelegate(self, callbackQueue: DispatchQueue(label: "DepthDataQueue"))
            }
        } catch {
            print("Error setting up camera: \(error)")
        }

        captureSession.commitConfiguration()
    }

    func startStream() {
        if !captureSession.isRunning {
            captureSession.startRunning()
        }
    }

    func stopStream() {
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }

    func capturePhoto() {
        let settings = AVCapturePhotoSettings()
        if #available(iOS 16.0, *) {
            if let maxDims = photoOutput.availablePhotoPixelFormatTypes.first {
                // No direct mapping; prefer using maxPhotoDimensions when using Photo Capture API v2.
            }
            settings.maxPhotoDimensions = .init(width: 4032, height: 3024) // Adjust as desired
        }
        settings.flashMode = .off

        photoOutput.capturePhoto(with: settings, delegate: self)
        print("Capture photo initiated")
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            print("Error capturing photo: \(error)")
            return
        }

        guard let imageData = photo.fileDataRepresentation() else {
            print("Failed to get photo data")
            return
        }

        savePhotoToLibrary(imageData: imageData)
    }

    private func savePhotoToLibrary(imageData: Data) {
        PHPhotoLibrary.requestAuthorization { status in
            guard status == .authorized else {
                print("Photo Library access denied.")
                return
            }

            PHPhotoLibrary.shared().performChanges({
                let options = PHAssetResourceCreationOptions()
                let request = PHAssetCreationRequest.forAsset()
                request.addResource(with: .photo, data: imageData, options: options)
            }) { success, error in
                if success {
                    print("Photo successfully saved to library.")
                } else {
                    print("Error saving photo: \(error?.localizedDescription ?? "Unknown error")")
                }
            }
        }
    }

    public func makePreviewLayer() -> AVCaptureVideoPreviewLayer {
        let layer = AVCaptureVideoPreviewLayer(session: captureSession)
        layer.videoGravity = .resizeAspectFill
        return layer
    }
}

extension CameraManager: AVCaptureDepthDataOutputDelegate {
    func depthDataOutput(_ output: AVCaptureDepthDataOutput, didOutput depthData: AVDepthData, timestamp: CMTime, connection: AVCaptureConnection) {
        let pixelBuffer = depthData.depthDataMap
        DispatchQueue.main.async {
            self.capturedData.depth = pixelBuffer
            self.isDataAvailable = true
        }
    }
}
