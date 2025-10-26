"use client";
import { useState } from "react";

export function IngestForm({
  loading,
  onSubmit,
}: {
  loading: boolean;
  onSubmit: (sourceDir?: string) => Promise<void> | void;
}) {
  const [sourceDir, setSourceDir] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onSubmit(sourceDir || undefined);
  };

  return (
    <div className="bg-white rounded-2xl p-8 mb-8 shadow-[0_10px_30px_rgba(0,0,0,0.2)]">
      <h2 className="mt-0 mb-6 text-indigo-500 text-2xl">
        Run Document Indexing
      </h2>
      <form onSubmit={handleSubmit} className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <label
            htmlFor="sourceDir"
            className="font-semibold text-gray-800 text-[0.95rem]"
          >
            Source Directory
          </label>
          <input
            id="sourceDir"
            type="text"
            value={sourceDir}
            onChange={(e) => setSourceDir(e.target.value)}
            placeholder="Defaults to ./data/docs"
            className="px-3 py-3 border-2 border-gray-300 rounded-lg text-base font-[inherit] transition-all duration-300 focus:outline-none focus:border-indigo-500 focus:shadow-[0_0_0_3px_rgba(102,126,234,0.1)]"
          />
          <p className="text-sm text-gray-600 m-0">
            Leave blank to use the server default directory defined in{" "}
            <code>INGESTION_SOURCE_DIR</code>.
          </p>
        </div>

        <button
          type="submit"
          className="px-8 py-4 bg-gradient-to-br from-indigo-500 to-purple-600 text-white border-none rounded-xl text-lg font-semibold cursor-pointer transition-all duration-300 shadow-[0_4px_15px_rgba(102,126,234,0.4)] hover:enabled:-translate-y-0.5 hover:enabled:shadow-[0_6px_20px_rgba(102,126,234,0.6)] disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none"
          disabled={loading}
        >
          {loading ? "Processing..." : "Start Indexing"}
        </button>
      </form>
    </div>
  );
}
