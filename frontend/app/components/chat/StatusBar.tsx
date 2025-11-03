"use client";

export function StatusBar({
  sessionId,
  isLoading,
}: {
  sessionId?: string;
  isLoading: boolean;
}) {
  return (
    <footer className="flex justify-between items-center px-6 py-3.5 bg-slate-50/90 border-t border-slate-300/20 text-xs text-slate-600">
      <p className="m-0">Session ID: {sessionId || "Not started"}</p>
      <p className="m-0">Status: {isLoading ? "🔄 Thinking..." : "✅ Ready"}</p>
    </footer>
  );
}
