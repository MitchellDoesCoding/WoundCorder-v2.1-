import SwiftUI

struct MeasurementLogListView: View {
    @EnvironmentObject var logManager: MeasurementLogManager

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Measurement Logs")
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        CloseButton()
                    }
                }
        }
    }

    @ViewBuilder
    private var content: some View {
        let items = logManager.items
        if items.isEmpty {
            ContentUnavailableView("No Logs", systemImage: "doc.text.magnifyingglass", description: Text("Logs you record will appear here."))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(.systemGroupedBackground))
        } else {
            List {
                ForEach(items) { item in
                    // Adjust this row to your SavedMeasurement shape if needed
                    VStack(alignment: .leading, spacing: 4) {
                        Text(item.title)
                            .font(.headline)
                        // If SavedMeasurement contains perimeter/area/volume, show a compact summary
                        Text(summary(for: item))
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }
                .onDelete(perform: logManager.delete)
            }
        }
    }

    // Helper to create a compact description of a SavedMeasurement.
    private func summary(for item: SavedMeasurement) -> String {
        // Try to build a readable line using shareText if available; otherwise compose a best-effort summary.
        // We don't have direct access to the values here; if SavedMeasurement exposes them, replace this with real fields.
        // As a safe default, use the manager's shareText(for:) which already formats a line.
        logManager.shareText(for: item)
    }
}

private struct CloseButton: View {
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        Button("Done") { dismiss() }
    }
}

#Preview {
    let manager = MeasurementLogManager()
    // Add a sample entry for preview
    manager.add(title: "Sample", perimeter: 12.34, area: 56.78, volume: 9.01)

    return MeasurementLogListView()
        .environmentObject(manager)
}
