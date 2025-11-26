// SummaryFormatter.swift
// Utilities to format analysis text returned from backend (GPT)
// Provides: paragraph-styled AttributedString and Markdown rendering helpers

import Foundation
import SwiftUI
import UIKit
@preconcurrency import Foundation
@preconcurrency import UIKit
@MainActor
public enum SummaryFormatter {
    /// Normalizes literal "\\n" into real newlines and applies paragraph/line spacing.
    /// Use in SwiftUI with `Text(SummaryFormatter.formatParagraphs(summary))`.
    @MainActor
    public static func formatParagraphs(_ raw: String,
                                        lineSpacing: CGFloat = 2,
                                        paragraphSpacing: CGFloat = 6) -> Foundation.AttributedString {
        var normalized = raw.replacingOccurrences(of: "\\n", with: "\n")
        // Also normalize Windows newlines just in case
        normalized = normalized.replacingOccurrences(of: "\r\n", with: "\n")

        var attributed = Foundation.AttributedString(normalized)

        // Build NSParagraphStyle and apply via AttributeContainer
        let nsParagraph = NSMutableParagraphStyle()
        nsParagraph.lineBreakMode = .byWordWrapping
        nsParagraph.lineSpacing = lineSpacing
        nsParagraph.paragraphSpacing = paragraphSpacing

        var container = AttributeContainer()
        container.paragraphStyle = nsParagraph
        attributed.mergeAttributes(container)

        return attributed
    }

    /// Parses Markdown (after normalizing literal "\\n") to an AttributedString.
    /// Use in SwiftUI with `Text(SummaryFormatter.formatMarkdown(summary))`.
    @MainActor
    public static func formatMarkdown(_ raw: String) -> Foundation.AttributedString {
        let normalized = raw
            .replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\r\n", with: "\n")
        do {
            var attributed = try Foundation.AttributedString(markdown: normalized)

            // Build NSParagraphStyle and apply via AttributeContainer
            let nsParagraph = NSMutableParagraphStyle()
            nsParagraph.lineBreakMode = .byWordWrapping
            nsParagraph.lineSpacing = 2
            nsParagraph.paragraphSpacing = 6

            var container = AttributeContainer()
            container.paragraphStyle = nsParagraph
            attributed.mergeAttributes(container)

            return attributed
        } catch {
            return formatParagraphs(normalized)
        }
    }
}

// MARK: - UIKit conveniences
public enum SummaryFormatterUIKit {
    /// Builds an NSAttributedString with paragraph/line spacing for UILabel.
    public static func makeParagraphAttributedText(_ raw: String,
                                                   font: UIFont = .preferredFont(forTextStyle: .body),
                                                   lineSpacing: CGFloat = 2,
                                                   paragraphSpacing: CGFloat = 6) -> NSAttributedString {
        let normalized = raw
            .replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\r\n", with: "\n")
        let paragraph = NSMutableParagraphStyle()
        paragraph.lineBreakMode = .byWordWrapping
        paragraph.lineSpacing = lineSpacing
        paragraph.paragraphSpacing = paragraphSpacing
        return NSAttributedString(
            string: normalized,
            attributes: [
                .paragraphStyle: paragraph,
                .font: font
            ]
        )
    }

    /// Parses Markdown into NSAttributedString for UILabel. Falls back to paragraph styling on failure.
    public static func makeMarkdownAttributedText(_ raw: String,
                                                  font: UIFont = .preferredFont(forTextStyle: .body),
                                                  lineSpacing: CGFloat = 2,
                                                  paragraphSpacing: CGFloat = 6) -> NSAttributedString {
        let normalized = raw
            .replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\r\n", with: "\n")
        do {
            let swiftAttributed = try Foundation.AttributedString(markdown: normalized)
            let nsAttributed = NSAttributedString(swiftAttributed)
            // Apply font and paragraph style overrides
            let mutable = NSMutableAttributedString(attributedString: nsAttributed)

            let fullRange = NSRange(location: 0, length: mutable.length)
            // Add font attribute
            mutable.addAttribute(.font, value: font, range: fullRange)

            // Paragraph style
            let paragraph = NSMutableParagraphStyle()
            paragraph.lineBreakMode = .byWordWrapping
            paragraph.lineSpacing = lineSpacing
            paragraph.paragraphSpacing = paragraphSpacing
            mutable.addAttribute(.paragraphStyle, value: paragraph, range: fullRange)

            return mutable
        } catch {
            // Fallback to paragraph styling
            return makeParagraphAttributedText(normalized, font: font, lineSpacing: lineSpacing, paragraphSpacing: paragraphSpacing)
        }
    }
}

