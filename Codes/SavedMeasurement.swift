import Foundation

struct SavedMeasurement: Identifiable, Codable, Equatable {
    let id: UUID
    var title: String
    let date: Date
    let perimeter: Float
    let area: Float
    let volume: Float

    init(id: UUID = UUID(), title: String, date: Date = Date(), perimeter: Float, area: Float, volume: Float) {
        self.id = id
        self.title = title
        self.date = date
        self.perimeter = perimeter
        self.area = area
        self.volume = volume
    }
}
