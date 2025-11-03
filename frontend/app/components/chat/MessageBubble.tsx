"use client";
import type { Message } from "../../types/chat";

function getResponseTypeLabel(type?: string) {
  switch (type) {
    case "answer":
      return "💡 Answer";
    case "workflow":
    case "clarify":
      return "📋 Workflow";
    case "link_docs":
    case "search_kb":
      return "📚 Documentation";
    case "escalate":
      return "🚨 Escalated";
    default:
      return "";
  }
}

function getConfidenceColor(confidence?: number) {
  if (!confidence) return "#94a3b8";
  if (confidence >= 0.8) return "#16a34a";
  if (confidence >= 0.6) return "#f97316";
  return "#dc2626";
}

export function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div
      className={`flex animate-[fadeIn_0.3s_ease-in] ${
        isUser ? "justify-end" : "justify-start"
      }`}
    >
      <div
        className={`max-w-[680px] px-5 py-4 rounded-2xl relative border shadow-[0_6px_18px_rgba(15,23,42,0.04)] ${
          isUser
            ? "ml-auto bg-blue-700 text-white rounded-tr-[4px]"
            : "mr-auto bg-white text-slate-900 border-slate-300/30 rounded-tl-[4px] shadow-none"
        }`}
      >
        <div className="leading-relaxed whitespace-pre-wrap break-words">
          {message.content}
        </div>
        {message.role === "assistant" && message.responseType && (
          <div className="flex gap-3 mt-2.5 text-xs text-slate-600 flex-wrap">
            <span
              className={`font-semibold px-2.5 py-1 rounded-full ${
                isUser
                  ? "bg-blue-500/10 text-blue-700"
                  : "bg-slate-300/20 text-slate-900"
              }`}
            >
              {getResponseTypeLabel(message.responseType)}
            </span>
            {message.confidence !== undefined && (
              <span
                className="font-medium"
                style={{ color: getConfidenceColor(message.confidence) }}
              >
                {(message.confidence * 100).toFixed(0)}% confidence
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
