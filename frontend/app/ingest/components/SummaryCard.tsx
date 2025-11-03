"use client";

export type IndexSummary = {
  files_indexed: number;
  chunks_created?: number;
  chunks_indexed?: number;
  duration_seconds?: number;
  message?: string;
};

export function SummaryCard({ summary }: { summary: IndexSummary | null }) {
  if (!summary) return null;
  return (
    <div className="bg-white rounded-2xl p-8 mb-8 shadow-[0_10px_30px_rgba(0,0,0,0.2)]">
      <h2 className="mt-0 mb-6 text-indigo-500 text-2xl">
        Last Indexing Summary
      </h2>
      <div className="grid grid-cols-[repeat(auto-fit,minmax(200px,1fr))] gap-6 mb-4">
        <div className="flex flex-col gap-2">
          <div className="text-sm text-gray-600 font-medium">Files Indexed</div>
          <div className="text-3xl font-bold text-gray-800">
            {summary.files_indexed ?? "-"}
          </div>
        </div>
        {summary.chunks_created !== undefined && (
          <div className="flex flex-col gap-2">
            <div className="text-sm text-gray-600 font-medium">
              Chunks Created
            </div>
            <div className="text-3xl font-bold text-gray-800">
              {summary.chunks_created}
            </div>
          </div>
        )}
        {summary.chunks_indexed !== undefined && (
          <div className="flex flex-col gap-2">
            <div className="text-sm text-gray-600 font-medium">
              Chunks Indexed
            </div>
            <div className="text-3xl font-bold text-gray-800">
              {summary.chunks_indexed}
            </div>
          </div>
        )}
        {summary.duration_seconds !== undefined && (
          <div className="flex flex-col gap-2">
            <div className="text-sm text-gray-600 font-medium">
              Duration (s)
            </div>
            <div className="text-3xl font-bold text-gray-800">
              {summary.duration_seconds}
            </div>
          </div>
        )}
      </div>
      {summary.message && (
        <div className="text-sm text-gray-600 m-0 italic">
          {summary.message}
        </div>
      )}
    </div>
  );
}
