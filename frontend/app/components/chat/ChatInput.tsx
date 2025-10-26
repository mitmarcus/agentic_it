"use client";
import { useState } from "react";

export function ChatInput({
  isLoading,
  onSend,
}: {
  isLoading: boolean;
  onSend: (text: string) => Promise<void> | void;
}) {
  const [input, setInput] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;
    setInput("");
    await onSend(text);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex gap-3 px-6 py-5 bg-slate-50 border-t border-slate-300/20"
    >
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Ask me about IT support..."
        className="flex-1 px-4 py-3.5 border border-slate-400/50 rounded-full text-[0.95rem] outline-none transition-all duration-200 bg-white text-slate-900 placeholder:text-slate-400 focus:border-blue-600 focus:shadow-[0_0_0_3px_rgba(37,99,235,0.15)] disabled:bg-slate-100 disabled:cursor-not-allowed"
        disabled={isLoading}
      />
      <button
        type="submit"
        className="px-7 py-3.5 bg-blue-700 text-white border-none rounded-full text-lg cursor-pointer transition-all duration-200 min-w-[58px] shadow-[0_12px_24px_rgba(29,78,216,0.18)] hover:enabled:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none"
        disabled={isLoading || !input.trim()}
      >
        {isLoading ? "⏳" : "📤"}
      </button>
    </form>
  );
}
