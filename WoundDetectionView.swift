import SwiftUI
#if canImport(PhotosUI)
import PhotosUI
#endif

struct WoundDetectionView: View {
    @State private var showingImagePicker: Bool = false
    @State private var useCamera: Bool = false
    @State private var image: UIImage? = nil
    @State private var detectionResult: String? = nil
    @State private var isLoading: Bool = false
    private let detector = WoundDetector()

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                Text("Wound Detection")
                    .font(.title).bold()

                if let image = image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(.secondary.opacity(0.3), lineWidth: 1)
                        )
                } else {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(.secondary.opacity(0.1))
                        .frame(height: 220)
                        .overlay(Text("Pick an image to run detection").foregroundColor(.secondary))
                }

#if canImport(PhotosUI)
                if #available(iOS 16.0, *) {
                    PhotosPickerButton { uiImage in
                        Task { await analyze(uiImage) }
                    }
                } else {
                    HStack {
                        Button {
                            useCamera = true
                            showingImagePicker = true
                        } label: {
                            Label("Camera", systemImage: "camera")
                        }
                        .buttonStyle(.bordered)

                        Button {
                            useCamera = false
                            showingImagePicker = true
                        } label: {
                            Label("Library", systemImage: "photo")
                        }
                        .buttonStyle(.borderedProminent)
                    }
                }
#else
                HStack {
                    Button {
                        useCamera = true
                        showingImagePicker = true
                    } label: {
                        Label("Camera", systemImage: "camera")
                    }
                    .buttonStyle(.bordered)

                    Button {
                        useCamera = false
                        showingImagePicker = true
                    } label: {
                        Label("Library", systemImage: "photo")
                    }
                    .buttonStyle(.borderedProminent)
                }
#endif

                if isLoading {
                    ProgressView("Analyzing...")
                        .padding(.top, 8)
                }

                if let detectionResult = detectionResult {
                    Text(formattedParagraphs(detectionResult))
                        .font(.headline)
                        .padding(.top)
                }

                Spacer(minLength: 20)
            }
            .padding()
            .onChange(of: image) { _ in
                Task { if let img = image { await analyze(img) } }
            }
        }
        .sheet(isPresented: $showingImagePicker) {
            LegacyImagePicker(image: $image, source: useCamera ? .camera : .library)
        }
    }

    private func analyze(_ uiImage: UIImage) async {
        await MainActor.run {
            self.image = uiImage
            self.isLoading = true
            self.detectionResult = nil
        }
        do {
            let result = try await detector.analyze(image: uiImage)
            await MainActor.run {
                self.isLoading = false
                var summary = "\(result.summary)"
                self.detectionResult = summary
            }
        } catch {
            await MainActor.run {
                self.isLoading = false
                self.detectionResult = "Analysis failed: \(error.localizedDescription)"
            }
        }
    }

    private func formattedParagraphs(_ raw: String) -> AttributedString {
        // Normalize common escaped and Windows line breaks to standard newlines
        var normalized = raw.replacingOccurrences(of: "\\n", with: "\n")
        normalized = normalized.replacingOccurrences(of: "\r\n", with: "\n")
        return AttributedString(normalized)
    }
}

#if canImport(PhotosUI)
@available(iOS 16.0, *)
private struct PhotosPickerButton: View {
    @State private var selectedItem: PhotosPickerItem? = nil
    var onPicked: (UIImage) -> Void

    var body: some View {
        PhotosPicker(selection: $selectedItem, matching: .images) {
            Label("Pick Image", systemImage: "photo")
        }
        .buttonStyle(.borderedProminent)
        .onChange(of: selectedItem) { _ in
            Task { await load() }
        }
    }

    private func load() async {
        guard let data = try? await selectedItem?.loadTransferable(type: Data.self),
              let uiImage = UIImage(data: data) else { return }
        onPicked(uiImage)
    }
}
#endif

struct LegacyImagePicker: UIViewControllerRepresentable {
    enum Source { case camera, library }
    @Environment(\.presentationMode) private var presentationMode
    @Binding var image: UIImage?
    var source: Source = .library

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = (source == .camera && UIImagePickerController.isSourceTypeAvailable(.camera)) ? .camera : .photoLibrary
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: LegacyImagePicker
        init(_ parent: LegacyImagePicker) { self.parent = parent }
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let img = info[.originalImage] as? UIImage {
                parent.image = img
            }
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}


#Preview {
    NavigationView { WoundDetectionView() }
}

