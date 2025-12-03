"use client";

import { useState, useCallback } from "react";
import { ChatHeader } from "./components/chat/ChatHeader";
import { MessageList } from "./components/chat/MessageList";
import { ChatInput } from "./components/chat/ChatInput";
import { StatusBar } from "./components/chat/StatusBar";
import type { Message, ChatResponse } from "./types/chat";
import { postJSON, submitFeedback } from "./lib/api";
import { getUserAgentInfo } from "./lib/userAgent";

export default function Home() {
  // The layout is intentionally minimal; styling lives in page.module.css
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content:
        "Hello! I'm your IT Support Assistant. How can I help you today?",
      timestamp: Date.now(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>("");
  const onSend = async (text: string) => {
    if (!text.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: text,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Detect user's OS
      const userAgent = getUserAgentInfo();

      const data = await postJSON<
        {
          query: string;
          session_id?: string;
          user_os?: string;
        },
        ChatResponse
      >("/query", {
        query: text,
        session_id: sessionId || undefined,
        user_os: userAgent.os,
      });

      if (!sessionId && data.session_id) {
        setSessionId(data.session_id);
      }

      // Extract intent (now just a string) and decision confidence
      const intentType = data.metadata?.intent;
      const decisionConfidence = data.metadata?.decision?.confidence;
      const retrievedDocIds = data.metadata?.retrieved_doc_ids;

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response,
        intentType: typeof intentType === "string" ? intentType : undefined,
        decisionConfidence:
          typeof decisionConfidence === "number"
            ? decisionConfidence
            : undefined,
        responseType: data.action_taken,
        timestamp: Date.now(),
        userQuery: text, // Store the original query for feedback
        retrievedDocIds: retrievedDocIds, // Store doc IDs for feedback
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content:
          "Sorry, I encountered an error. Please make sure the backend server is running on port 8000.",
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle feedback submission
  const handleFeedback = useCallback(
    async (messageId: string, feedbackType: "positive" | "negative") => {
      const message = messages.find((m) => m.id === messageId);
      if (!message || !sessionId || !message.userQuery) return;

      try {
        await submitFeedback(
          sessionId,
          message.userQuery,
          message.content,
          feedbackType,
          message.retrievedDocIds // doc IDs for feedback-aware retrieval
        );

        // Update message state to show feedback was received
        setMessages((prev) =>
          prev.map((m) =>
            m.id === messageId ? { ...m, feedback: feedbackType } : m
          )
        );
      } catch (error) {
        console.error("Failed to submit feedback:", error);
      }
    },
    [messages, sessionId]
  );

  return (
    <main className="flex flex-col items-center p-8 sm:p-4 min-h-screen bg-gray-100">
      <div className="w-full max-w-[860px] h-[85vh] flex flex-col bg-white rounded-[18px] shadow-[0_20px_45px_rgba(15,23,42,0.08)] border border-slate-300/20 overflow-hidden">
        <ChatHeader />

        <div className="flex-1 flex flex-col overflow-hidden">
          <MessageList
            messages={messages}
            isLoading={isLoading}
            sessionId={sessionId}
            onFeedback={handleFeedback}
          />
          <ChatInput isLoading={isLoading} onSend={onSend} />
        </div>

        <StatusBar sessionId={sessionId} isLoading={isLoading} />
      </div>
    </main>
  );
}
