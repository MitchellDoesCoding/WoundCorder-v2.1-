import SwiftUI

struct ComorbidityChecklistSheet: View {
    @Environment(\.dismiss) private var dismiss

    @State private var checklist = ComorbidityChecklist()

    var onUse: (ComorbidityChecklist) -> Void
    var onSkip: (() -> Void)?

    var body: some View {
        NavigationStack {
            List {
                Section("Comorbidities") {
                    PickerRow(title: "Diabetes mellitus", selection: $checklist.diabetes_mellitus)
                    PickerRow(title: "Peripheral arterial disease (PAD)", selection: $checklist.peripheral_arterial_disease_PAD)
                    PickerRow(title: "Chronic venous or lymphedema", selection: $checklist.chronic_venous_or_lymphedema)
                    PickerRow(title: "Peripheral neuropathy or foot deformity", selection: $checklist.peripheral_neuropathy_or_foot_deformity)
                    PickerRow(title: "Limited mobility / pressure-risk devices", selection: $checklist.limited_mobility_or_pressure_risk_devices)
                    PickerRow(title: "Renal/hepatic failure or malnutrition", selection: $checklist.renal_or_hepatic_failure_or_malnutrition)
                    PickerRow(title: "Heart failure or cardiovascular disease", selection: $checklist.heart_failure_or_cardiovascular_disease)
                    PickerRow(title: "Tobacco use (current or recent)", selection: $checklist.tobacco_use_current_or_recent)
                    PickerRow(title: "Immunosuppression / steroids / active cancer therapy", selection: $checklist.immunosuppression_steroids_or_active_cancer_therapy)
                    PickerRow(title: "Prior ulcer/amputation/osteomyelitis at site", selection: $checklist.prior_ulcer_amputation_or_osteomyelitis_at_site)
                }
            }
            .navigationTitle("Pre‑capture Checklist")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Skip") {
                        onSkip?()
                        dismiss()
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Use These") {
                        onUse(checklist)
                        dismiss()
                    }
                    .bold()
                }
            }
        }
    }
}

private struct PickerRow: View {
    let title: String
    @Binding var selection: Ternary

    var body: some View {
        HStack {
            Text(title)
            Spacer()
            Picker("", selection: $selection) {
                Text("Unknown").tag(Ternary.unknown)
                Text("Yes").tag(Ternary.yes)
                Text("No").tag(Ternary.no)
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 260)
        }
        .contentShape(Rectangle())
    }
}

#Preview {
    ComorbidityChecklistSheet(onUse: { _ in }, onSkip: {})
}
