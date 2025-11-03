"use client";
import Link from "next/link";

export function ChatHeader() {
  return (
    <header className="flex justify-between items-center px-7 py-5 border-b border-slate-300/20 bg-slate-50/90 backdrop-blur-xl">
      <div className="flex flex-col gap-1.5">
        <h1 className="m-0 text-2xl font-semibold text-slate-900">
          IT Support Chatbot
        </h1>
      </div>
      <Link
        href="/ingest"
        className="text-blue-700 no-underline text-[0.95rem] px-4 py-2.5 rounded-full border border-blue-400/25 bg-blue-500/8 transition-all duration-200 whitespace-nowrap font-medium hover:bg-blue-500/15 hover:border-blue-400/45"
      >
        Manage Knowledge Base
      </Link>
    </header>
  );
}
