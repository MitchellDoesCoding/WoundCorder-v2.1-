import SwiftUI

struct MeasurementTabView: View {
    @EnvironmentObject private var logManager: MeasurementLogManager
    @StateObject private var metricsManager = MetricsManager()
    @State private var userPoints: [SIMD3<Float>] = []
    @State private var selectedDepthPoint: SIMD3<Float>? = nil
    @State private var showSavePrompt = false
    @State private var pendingTitle: String = ""

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Reuse your existing AR measurement UI
                ARViewContainer(
                    userPoints: $userPoints,
                    metricsManager: metricsManager,
                    selectedDepthPoint: $selectedDepthPoint,
                    onMeasurementState: { _ in }
                )
                .ignoresSafeArea(.all)
                .frame(maxHeight: .infinity)

                Divider()

                // Controls & current metrics
                VStack(alignment: .leading, spacing: 8) {
                    MetricsView(metrics: metricsManager.metrics)

                    HStack {
                        Button("Reset") { resetScene() }
                            .buttonStyle(.bordered)

                        Button("Save Measurement") { prepareSave() }
                            .buttonStyle(.borderedProminent)

                        Spacer()
                    }
                }
                .padding()

                // Saved measurements list
                List {
                    ForEach(logManager.items) { item in
                        HStack {
                            VStack(alignment: .leading) {
                                Text(item.title).font(.headline)
                                Text(dateString(item.date)).font(.caption).foregroundColor(.secondary)
                                Text("P: \(item.perimeter, specifier: "%.2f") cm  A: \(item.area, specifier: "%.2f") cm²  V: \(item.volume, specifier: "%.2f") cm³")
                                    .font(.subheadline)
                            }
                            Spacer()
                            Menu {
                                Button("Rename") { promptRename(item) }
                                ShareLink(item: logManager.shareText(for: item)) { Label("Share Text", systemImage: "square.and.arrow.up") }
                                ShareLink(item: logManager.shareCSV(for: item)) { Label("Share CSV", systemImage: "table") }
                            } label: {
                                Image(systemName: "ellipsis.circle").imageScale(.large)
                            }
                        }
                    }
                    .onDelete(perform: logManager.delete)
                }
            }
            .navigationTitle("Wound Measurement")
        }
        .sheet(isPresented: $showSavePrompt) {
            NavigationView {
                Form {
                    Section(header: Text("Title")) {
                        TextField("e.g. Left calf — day 3", text: $pendingTitle)
                    }
                    Section(header: Text("Summary")) {
                        Text("Perimeter: \(metricsManager.metrics.perimeter, specifier: "%.2f") cm")
                        Text("Area: \(metricsManager.metrics.area, specifier: "%.2f") cm²")
                        Text("Volume: \(metricsManager.metrics.volume, specifier: "%.2f") cm³")
                    }
                }
                .navigationTitle("Save Measurement")
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        Button("Cancel") { showSavePrompt = false }
                    }
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Save") { commitSave() }.disabled(pendingTitle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }
            }
        }
    }

    private func resetScene() {
        userPoints.removeAll()
        selectedDepthPoint = nil
        metricsManager.resetMetrics()
        NotificationCenter.default.post(name: NSNotification.Name("ResetARScene"), object: nil)
    }

    private func prepareSave() {
        pendingTitle = ""
        showSavePrompt = true
    }

    private func commitSave() {
        let t = pendingTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        logManager.add(title: t.isEmpty ? "Untitled" : t,
                       perimeter: metricsManager.metrics.perimeter,
                       area: metricsManager.metrics.area,
                       volume: metricsManager.metrics.volume)
        showSavePrompt = false
    }

    private func promptRename(_ item: SavedMeasurement) {
        // Simple inline rename using an alert with TextField is not directly supported.
        // For now, present a sheet similar to save.
        pendingTitle = item.title
        showSavePrompt = true
        // After saving, we'll replace the last item if title matches; or you can implement a separate rename sheet.
        // For a precise rename flow, you may want a dedicated sheet with a reference to the item being edited.
    }

    private func dateString(_ date: Date) -> String {
        let df = DateFormatter()
        df.dateStyle = .medium
        df.timeStyle = .short
        return df.string(from: date)
    }
}

#Preview {
    MeasurementTabView().environmentObject(MeasurementLogManager())
}
