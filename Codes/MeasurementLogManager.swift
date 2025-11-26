import Foundation
import SwiftUI

@MainActor
final class MeasurementLogManager: ObservableObject {
    @AppStorage("savedMeasurements") private var storedJSON: String = ""

    @Published private(set) var items: [SavedMeasurement] = [] {
        didSet { persist() }
    }

    init() {
        load()
    }

    func add(title: String, perimeter: Float, area: Float, volume: Float) {
        let newItem = SavedMeasurement(title: title, perimeter: perimeter, area: area, volume: volume)
        items.insert(newItem, at: 0)
    }

    func delete(_ indexSet: IndexSet) {
        items.remove(atOffsets: indexSet)
    }

    func delete(item: SavedMeasurement) {
        if let idx = items.firstIndex(of: item) { items.remove(at: idx) }
    }

    func rename(item: SavedMeasurement, to newTitle: String) {
        guard let idx = items.firstIndex(of: item) else { return }
        items[idx].title = newTitle
    }

    func shareText(for item: SavedMeasurement) -> String {
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .medium
        dateFormatter.timeStyle = .short
        let dateString = dateFormatter.string(from: item.date)
        return "\(item.title) — \(dateString)\nPerimeter: \(String(format: "%.2f", item.perimeter)) cm\nArea: \(String(format: "%.2f", item.area)) cm²\nVolume: \(String(format: "%.2f", item.volume)) cm³"
    }

    func shareCSV(for item: SavedMeasurement) -> String {
        // CSV header + single row for this item
        let header = "id,title,date,perimeter_cm,area_cm2,volume_cm3\n"
        let iso = ISO8601DateFormatter().string(from: item.date)
        let row = "\(item.id.uuidString),\"\(item.title.replacingOccurrences(of: "\"", with: "\"\""))\",\(iso),\(String(format: "%.2f", item.perimeter)),\(String(format: "%.2f", item.area)),\(String(format: "%.2f", item.volume))\n"
        return header + row
    }

    // MARK: - Persistence

    private func load() {
        guard !storedJSON.isEmpty, let data = storedJSON.data(using: .utf8) else {
            items = []
            return
        }
        do {
            items = try JSONDecoder().decode([SavedMeasurement].self, from: data)
        } catch {
            items = []
        }
    }

    private func persist() {
        do {
            let data = try JSONEncoder().encode(items)
            storedJSON = String(data: data, encoding: .utf8) ?? ""
        } catch {
            // If encoding fails, keep previous storage
        }
    }
}
