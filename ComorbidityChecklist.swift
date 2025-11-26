import Foundation

/// A tri-state value used for checklist entries.
public enum Ternary: String, CaseIterable, Identifiable, Codable, Sendable {
    /// The state is unknown or not specified.
    case unknown
    /// The state is positive or affirmed.
    case yes
    /// The state is negative or denied.
    case no
    
    /// The unique identifier of the enum case.
    public var id: String { rawValue }
}

/// Checklist of comorbidities to optionally send along with a wound analysis.
public struct ComorbidityChecklist: Codable, Sendable {
    /// Presence of diabetes mellitus.
    public var diabetes_mellitus: Ternary = .unknown
    /// Presence of peripheral arterial disease (PAD).
    public var peripheral_arterial_disease_PAD: Ternary = .unknown
    /// Presence of chronic venous insufficiency or lymphedema.
    public var chronic_venous_or_lymphedema: Ternary = .unknown
    /// Presence of peripheral neuropathy or foot deformity.
    public var peripheral_neuropathy_or_foot_deformity: Ternary = .unknown
    /// Limited mobility or use of pressure risk devices.
    public var limited_mobility_or_pressure_risk_devices: Ternary = .unknown
    /// Presence of renal or hepatic failure, or malnutrition.
    public var renal_or_hepatic_failure_or_malnutrition: Ternary = .unknown
    /// Presence of heart failure or cardiovascular disease.
    public var heart_failure_or_cardiovascular_disease: Ternary = .unknown
    /// Current or recent tobacco use.
    public var tobacco_use_current_or_recent: Ternary = .unknown
    /// Immunosuppression, steroid use, or active cancer therapy.
    public var immunosuppression_steroids_or_active_cancer_therapy: Ternary = .unknown
    /// Prior ulcer, amputation, or osteomyelitis at the site.
    public var prior_ulcer_amputation_or_osteomyelitis_at_site: Ternary = .unknown

    /// Creates a new checklist with all values set to unknown.
    public init() {}

    /// Returns a plain dictionary representation suitable for JSON payloads.
    ///
    /// - Returns: A dictionary mapping property names to their string raw values.
    public func asDictionary() -> [String: String] {
        [
            "diabetes_mellitus": diabetes_mellitus.rawValue,
            "peripheral_arterial_disease_PAD": peripheral_arterial_disease_PAD.rawValue,
            "chronic_venous_or_lymphedema": chronic_venous_or_lymphedema.rawValue,
            "peripheral_neuropathy_or_foot_deformity": peripheral_neuropathy_or_foot_deformity.rawValue,
            "limited_mobility_or_pressure_risk_devices": limited_mobility_or_pressure_risk_devices.rawValue,
            "renal_or_hepatic_failure_or_malnutrition": renal_or_hepatic_failure_or_malnutrition.rawValue,
            "heart_failure_or_cardiovascular_disease": heart_failure_or_cardiovascular_disease.rawValue,
            "tobacco_use_current_or_recent": tobacco_use_current_or_recent.rawValue,
            "immunosuppression_steroids_or_active_cancer_therapy": immunosuppression_steroids_or_active_cancer_therapy.rawValue,
            "prior_ulcer_amputation_or_osteomyelitis_at_site": prior_ulcer_amputation_or_osteomyelitis_at_site.rawValue
        ]
    }

    /// Encodes the checklist as JSON data.
    ///
    /// - Parameter pretty: If true, produces pretty-printed JSON. Defaults to false.
    /// - Throws: An error if encoding fails.
    /// - Returns: The JSON data representing the checklist.
    public func jsonData(pretty: Bool = false) throws -> Data {
        let encoder = JSONEncoder()
        if pretty { encoder.outputFormatting.insert(.prettyPrinted) }
        return try encoder.encode(self)
    }
}
