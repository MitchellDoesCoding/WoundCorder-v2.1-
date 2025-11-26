import SwiftUI

struct HomeView: View {
    var body: some View {
        NavigationView {
            ZStack {
                Color.black.edgesIgnoringSafeArea(.all)

                VStack(spacing: 30) {
                    Text("Wound Measurement App")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .foregroundColor(.white)

                    Text("Measure wounds using LiDAR and AR technology.")
                        .multilineTextAlignment(.center)
                        .padding()
                        .foregroundColor(.gray)

                    NavigationLink(destination: ContentView()) {
                        Text("Start Measurement")
                            .font(.title2)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    .padding(.horizontal, 20)
                }
                .padding()
            }
        }
    }
}
