"use client";

import { useState, useEffect } from "react";
import { getJSON, postJSON } from "../lib/api";
import { StatsCard, type HealthStatus } from "./components/StatsCard";
import { MessageBanner } from "./components/MessageBanner";
import { IngestForm } from "./components/IngestForm";
import { SummaryCard, type IndexSummary } from "./components/SummaryCard";

interface IndexResponse {
  status: string;
  documents_loaded: number;
  chunks_created: number;
  chunks_indexed: number;
  message?: string;
}

export default function IngestPage() {
  const [sourceDir, setSourceDir] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [lastIndexResult, setLastIndexResult] = useState<IndexResponse | null>(
    null
  );

  // Fetch service health on mount
  useEffect(() => {
    fetchHealth();
  }, []);

  const fetchHealth = async () => {
    try {
      const data = await getJSON<HealthStatus>(`/health`);
      setHealth(data);
    } catch (error) {
      console.error("Error fetching health status:", error);
    }
  };

  const handleIndexSubmit = async (srcDir?: string) => {
    setLoading(true);
    setMessage(null);

    try {
      const data = await postJSON<{ source_dir?: string }, IndexResponse>(
        `/index`,
        {
          source_dir: srcDir,
        }
      );
      setLastIndexResult(data);

      if (data.status === "success") {
        setMessage({
          type: "success",
          text: `Indexing complete. ${data.documents_loaded} documents, ${data.chunks_indexed} chunks indexed.`,
        });
        fetchHealth();
      } else {
        setMessage({
          type: "error",
          text: data.message || "Failed to index documents",
        });
      }
    } catch (error) {
      setMessage({
        type: "error",
        text: `Error: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 bg-gradient-to-br from-indigo-500 to-purple-600">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-4xl font-bold text-white m-0 drop-shadow-[2px_2px_4px_rgba(0,0,0,0.2)]">
          Document Ingestion
        </h1>
        <a
          href="/"
          className="text-white no-underline text-lg px-4 py-2 rounded-lg bg-white/20 transition-all duration-300 hover:bg-white/30 hover:-translate-x-1"
        >
          ← Back to Chat
        </a>
      </div>

      {/* Stats Card */}
      <StatsCard health={health} onRefresh={fetchHealth} disabled={loading} />

      {/* Message Display */}
      {message && <MessageBanner type={message.type} text={message.text} />}

      {/* Ingest Form */}
      <IngestForm
        loading={loading}
        onSubmit={async (dir?: string) => {
          setSourceDir(dir || "");
          await handleIndexSubmit(dir || undefined);
        }}
      />

      {/* Summary */}
      <SummaryCard
        summary={
          lastIndexResult
            ? ({
                files_indexed: lastIndexResult.documents_loaded,
                chunks_created: lastIndexResult.chunks_created,
                chunks_indexed: lastIndexResult.chunks_indexed,
                message: lastIndexResult.message,
              } as IndexSummary)
            : null
        }
      />
    </div>
  );
}
