"use client";
import { useEffect, useRef } from "react";
import type { Message } from "../../types/chat";
import { MessageBubble } from "./MessageBubble";

export function MessageList({
  messages,
  isLoading,
}: {
  messages: Message[];
  isLoading: boolean;
}) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto px-8 py-6 flex flex-col gap-5 bg-white [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-gray-100 [&::-webkit-scrollbar-thumb]:bg-gray-400 [&::-webkit-scrollbar-thumb]:rounded [&::-webkit-scrollbar-thumb:hover]:bg-gray-500">
      {messages.map((m) => (
        <MessageBubble key={m.id} message={m} />
      ))}
      {isLoading && (
        <div className="flex justify-start animate-[fadeIn_0.3s_ease-in]">
          <div className="max-w-[680px] px-5 py-4 rounded-2xl relative border border-slate-300/30 shadow-none mr-auto bg-white text-slate-900 rounded-tl-sm">
            <div className="flex gap-1 items-center py-2">
              <span className="w-2 h-2 bg-[#cbd5f5] rounded-full animate-[typing_1.4s_infinite]"></span>
              <span className="w-2 h-2 bg-[#cbd5f5] rounded-full animate-[typing_1.4s_infinite] [animation-delay:0.2s]"></span>
              <span className="w-2 h-2 bg-[#cbd5f5] rounded-full animate-[typing_1.4s_infinite] [animation-delay:0.4s]"></span>
            </div>
          </div>
        </div>
      )}
      <div ref={endRef} />
    </div>
  );
}
