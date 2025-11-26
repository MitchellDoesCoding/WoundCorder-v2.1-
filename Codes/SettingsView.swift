import SwiftUI

private enum Audience: String, CaseIterable, Identifiable {
    case patient
    case doctor
    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .patient: return "Patient"
        case .doctor: return "Doctor"
        }
    }
}

struct SettingsView: View {
    @AppStorage("woundServerBaseURL") private var baseURL: String = ""
    @AppStorage("woundSummaryAudience") private var audienceRaw: String = Audience.patient.rawValue

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("https://6727e030ebe5.ngrok-free.app", text: $baseURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled(true)
                        .keyboardType(.URL)

                    Picker("Summary Audience", selection: $audienceRaw) {
                        ForEach(Audience.allCases) { a in
                            Text(a.displayName).tag(a.rawValue)
                        }
                    }
                    .pickerStyle(.segmented)
                    .padding(.vertical, 4)
                } footer: {
                    Text("Audience controls how AI summaries are written: Patient uses simpler language, Doctor uses technical terms.")
                }
            }
            .navigationTitle("Settings")
        }
    }
}

#Preview {
    SettingsView()
}
