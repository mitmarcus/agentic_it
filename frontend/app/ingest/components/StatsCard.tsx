"use client";

export interface HealthStatus {
  status: string;
  timestamp: string;
  version: string;
}

export function StatsCard({
  health,
  onRefresh,
  disabled,
}: {
  health: HealthStatus | null;
  onRefresh: () => void;
  disabled?: boolean;
}) {
  return (
    <div className="bg-white rounded-2xl p-8 mb-8 shadow-[0_10px_30px_rgba(0,0,0,0.2)]">
      <h2 className="mt-0 mb-6 text-indigo-500 text-2xl">Service Status</h2>
      {health ? (
        <div className="grid grid-cols-[repeat(auto-fit,minmax(200px,1fr))] gap-6 mb-4">
          <div className="flex flex-col gap-2">
            <span className="text-sm text-gray-600 font-medium">Status:</span>
            <span className="text-3xl font-bold text-gray-800">
              {health.status}
            </span>
          </div>
          <div className="flex flex-col gap-2">
            <span className="text-sm text-gray-600 font-medium">Version:</span>
            <span className="text-3xl font-bold text-gray-800">
              {health.version}
            </span>
          </div>
          <div className="flex flex-col gap-2">
            <span className="text-sm text-gray-600 font-medium">
              Last Check:
            </span>
            <span className="text-3xl font-bold text-gray-800">
              {new Date(health.timestamp).toLocaleString()}
            </span>
          </div>
        </div>
      ) : (
        <p className="text-gray-600 italic">Loading service status...</p>
      )}
      <button
        onClick={onRefresh}
        className="px-4 py-2 bg-indigo-500 text-white border-none rounded-lg cursor-pointer text-base transition-all duration-300 hover:enabled:bg-indigo-600 hover:enabled:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed"
        disabled={disabled}
      >
        ↻ Refresh
      </button>
    </div>
  );
}
