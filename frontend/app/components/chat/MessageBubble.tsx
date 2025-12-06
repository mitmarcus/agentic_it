"use client";
import { useState, ReactNode } from "react";
import type { Message } from "../../types/chat";

// Parse markdown bold (**text**) and convert to <strong> elements
function parseMarkdownBold(text: string): ReactNode[] {
  const boldRegex = /\*\*([^*]+)\*\*/g;
  const parts: ReactNode[] = [];
  let lastIndex = 0;
  let match;

  while ((match = boldRegex.exec(text)) !== null) {
    // Add text before the match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    // Add the bold text
    parts.push(
      <strong key={`bold-${match.index}`} className="font-semibold">
        {match[1]}
      </strong>
    );
    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length > 0 ? parts : [text];
}

// Convert URLs in text to clickable links, preserving bold formatting
function formatText(text: string): ReactNode[] {
  const urlRegex = /(https?:\/\/[^\s<>"{}|\\^`[\]]+)/g;
  const parts = text.split(urlRegex);

  return parts.flatMap((part, index) => {
    if (urlRegex.test(part)) {
      // Reset regex lastIndex since we're reusing it
      urlRegex.lastIndex = 0;

      // Strip trailing punctuation that shouldn't be part of URL
      let url = part;
      let trailing = "";
      const trailingPunctuation = /[.,;:!?)]+$/;
      const match = url.match(trailingPunctuation);
      if (match) {
        trailing = match[0];
        url = url.slice(0, -trailing.length);
      }

      return [
        <a
          key={`link-${index}`}
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 hover:text-blue-800 underline break-all"
        >
          {url}
        </a>,
        trailing,
      ].filter(Boolean);
    }
    // Parse bold markdown in non-URL parts
    return parseMarkdownBold(part);
  });
}

function getResponseTypeLabel(type?: string) {
  switch (type) {
    case "answer":
      return "💡 Answer";
    case "clarify":
      return "❓ Clarification";
    case "search_kb":
      return "📚 Documentation";
    case "troubleshoot":
      return "🔧 Troubleshooting";
    case "exit_troubleshoot":
      return "✅ Resolved";
    case "escalate":
      return "🚨 Escalated";
    case "create_ticket":
      return "🎫 Create Ticket";
    case "search_tickets":
      return "🔍 Search Tickets";
    case "not_implemented":
      return "⚠️ Not Available";
    default:
      return type ? `📋 ${type}` : "";
  }
}

function getConfidenceColor(confidence?: number) {
  if (confidence === undefined || confidence === null) return "#94a3b8";
  if (confidence >= 0.8) return "#16a34a"; // green
  if (confidence >= 0.6) return "#f97316"; // orange
  return "#dc2626"; // red
}

// Thumbs Up Icon
function ThumbsUpIcon({ filled }: { filled?: boolean }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
    </svg>
  );
}

// Thumbs Down Icon
function ThumbsDownIcon({ filled }: { filled?: boolean }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" />
    </svg>
  );
}

interface MessageBubbleProps {
  message: Message;
  sessionId?: string;
  onFeedback?: (
    messageId: string,
    feedbackType: "positive" | "negative"
  ) => void;
}

export function MessageBubble({
  message,
  sessionId,
  onFeedback,
}: MessageBubbleProps) {
  const isUser = message.role === "user";
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [localFeedback, setLocalFeedback] = useState<
    "positive" | "negative" | null
  >(message.feedback || null);

  const handleFeedback = async (feedbackType: "positive" | "negative") => {
    if (isSubmitting || localFeedback) return;

    setIsSubmitting(true);
    setLocalFeedback(feedbackType);

    try {
      if (onFeedback) {
        onFeedback(message.id, feedbackType);
      }
    } catch (error) {
      if (process.env.NODE_ENV === "development") {
        console.error("Failed to submit feedback:", error);
      }
      setLocalFeedback(null);
    } finally {
      setIsSubmitting(false);
    }
  };

  const showFeedbackButtons =
    message.role === "assistant" &&
    message.responseType &&
    sessionId &&
    message.userQuery;

  return (
    <div
      className={`flex animate-[fadeIn_0.3s_ease-in] ${
        isUser ? "justify-end" : "justify-start"
      }`}
    >
      <div
        className={`max-w-[680px] px-5 py-4 rounded-2xl relative border shadow-[0_6px_18px_rgba(15,23,42,0.04)] ${
          isUser
            ? "ml-auto bg-blue-700 text-white rounded-tr-sm"
            : "mr-auto bg-white text-slate-900 border-slate-300/30 rounded-tl-sm shadow-none"
        }`}
      >
        <div className="leading-relaxed whitespace-pre-wrap wrap-break-word">
          {isUser ? message.content : formatText(message.content)}
        </div>
        {message.role === "assistant" && message.responseType && (
          <div className="flex gap-3 mt-2.5 text-xs text-slate-600 flex-wrap items-center">
            <span
              className={`font-semibold px-2.5 py-1 rounded-full ${
                isUser
                  ? "bg-blue-500/10 text-blue-700"
                  : "bg-slate-300/20 text-slate-900"
              }`}
            >
              {getResponseTypeLabel(message.responseType)}
            </span>
            {/* Show intent type and decision confidence */}
            <div className="flex items-center gap-2">
              {message.intentType && (
                <span
                  className="font-medium text-slate-600"
                  title="Intent classification"
                >
                  {message.intentType}
                </span>
              )}
              {message.decisionConfidence !== undefined && (
                <span
                  className="font-medium flex items-center gap-1"
                  style={{
                    color: getConfidenceColor(message.decisionConfidence),
                  }}
                  title="Answer/decision confidence based on retrieved data"
                >
                  <span
                    className="inline-block w-2 h-2 rounded-full"
                    style={{
                      backgroundColor: getConfidenceColor(
                        message.decisionConfidence
                      ),
                    }}
                  />
                  Answer: {(message.decisionConfidence * 100).toFixed(0)}%
                </span>
              )}
            </div>

            {/* Feedback Buttons */}
            {showFeedbackButtons && (
              <div className="flex items-center gap-1 ml-auto">
                {localFeedback ? (
                  <span className="text-xs text-slate-500 italic">
                    {localFeedback === "positive"
                      ? "Thanks! 👍"
                      : "Thanks for the feedback"}
                  </span>
                ) : (
                  <>
                    <button
                      onClick={() => handleFeedback("positive")}
                      disabled={isSubmitting}
                      className="p-1.5 rounded-md hover:bg-green-100 text-slate-400 hover:text-green-600 transition-colors disabled:opacity-50"
                      title="This was helpful"
                    >
                      <ThumbsUpIcon />
                    </button>
                    <button
                      onClick={() => handleFeedback("negative")}
                      disabled={isSubmitting}
                      className="p-1.5 rounded-md hover:bg-red-100 text-slate-400 hover:text-red-600 transition-colors disabled:opacity-50"
                      title="This wasn't helpful"
                    >
                      <ThumbsDownIcon />
                    </button>
                  </>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
