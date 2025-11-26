"use client";

import { useState, useEffect } from "react";
import { getJSON } from "../lib/api";
import { StatsCard, type HealthStatus } from "./components/StatsCard";
import { FileUpload } from "../components/upload/FileUpload";

interface CollectionDocument {
  id: string;
  content: string;
  metadata: Record<string, any>;
  chunks?: Array<{
    id: string;
    content: string;
    chunk_index: number;
  }>;
}

interface CollectionInfo {
  collection_name: string;
  total_documents: number;
  documents: CollectionDocument[];
}

export default function IngestPage() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [collectionInfo, setCollectionInfo] = useState<CollectionInfo | null>(
    null
  );
  const [loadingCollection, setLoadingCollection] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState<CollectionDocument | null>(
    null
  );
  const [deletingDoc, setDeletingDoc] = useState<string | null>(null);

  // Fetch service health and collection info on mount
  useEffect(() => {
    fetchHealth();
    fetchCollectionInfo();
  }, []);

  const fetchHealth = async () => {
    try {
      const data = await getJSON<HealthStatus>(`/health`);
      setHealth(data);
    } catch (error) {
      console.error("Error fetching health status:", error);
    }
  };

  const fetchCollectionInfo = async () => {
    setLoadingCollection(true);
    try {
      const data = await getJSON<CollectionInfo>(`/collection/info?limit=100`);
      setCollectionInfo(data);
    } catch (error) {
      console.error("Error fetching collection info:", error);
    } finally {
      setLoadingCollection(false);
    }
  };

  const handleDeleteDocument = async (sourceFile: string, filename: string) => {
    if (
      !confirm(
        `Are you sure you want to delete "${filename}"? This will remove all chunks from the knowledge base.`
      )
    ) {
      return;
    }

    setDeletingDoc(sourceFile);
    try {
      const encodedPath = encodeURIComponent(sourceFile);
      const response = await fetch(
        `http://localhost:8000/documents/${encodedPath}`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to delete document: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("Delete result:", result);

      // Refresh the collection info
      await fetchCollectionInfo();
    } catch (error) {
      console.error("Error deleting document:", error);
      alert(`Failed to delete document: ${error}`);
    } finally {
      setDeletingDoc(null);
    }
  };

  return (
    <div className="min-h-screen p-8 bg-linear-to-br from-indigo-500 to-purple-600">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-4xl font-bold text-white m-0 drop-shadow-[2px_2px_4px_rgba(0,0,0,0.2)]">
          Document Upload
        </h1>
        <a
          href="/"
          className="text-white no-underline text-lg px-4 py-2 rounded-lg bg-white/20 transition-all duration-300 hover:bg-white/30 hover:-translate-x-1"
        >
          Back to Chat
        </a>
      </div>

      {/* Stats Card */}
      <StatsCard health={health} onRefresh={fetchHealth} />

      {/* File Upload Section */}
      <div className="bg-white rounded-2xl shadow-2xl p-6 mb-6">
        <FileUpload />
      </div>

      {/* Collection Documents List */}
      <div className="bg-white rounded-2xl shadow-2xl p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-gray-800">
            Indexed Documents
          </h2>
          <button
            onClick={fetchCollectionInfo}
            disabled={loadingCollection}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 transition-colors"
          >
            {loadingCollection ? "Loading..." : "Refresh"}
          </button>
        </div>

        {collectionInfo ? (
          <>
            <div className="text-sm text-gray-600 mb-4">
              Collection:{" "}
              <span className="font-semibold">
                {collectionInfo.collection_name}
              </span>
              {" • "}
              Total documents:{" "}
              <span className="font-semibold">
                {collectionInfo.total_documents}
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Filename
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Chunks
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Content Preview
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {collectionInfo.documents.map((doc) => (
                    <tr key={doc.id} className="hover:bg-gray-50">
                      <td className="px-4 py-3 text-sm font-medium text-gray-900">
                        {doc.metadata.filename || "N/A"}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-600">
                        <button
                          onClick={() => setSelectedDoc(doc)}
                          className="text-indigo-600 hover:text-indigo-800 hover:underline"
                        >
                          {doc.metadata.chunk_count || 0} chunks
                        </button>
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-600 max-w-lg truncate">
                        {doc.content}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-600">
                        <button
                          onClick={() =>
                            handleDeleteDocument(
                              doc.metadata.source_file,
                              doc.metadata.filename
                            )
                          }
                          disabled={deletingDoc === doc.metadata.source_file}
                          className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors text-xs"
                        >
                          {deletingDoc === doc.metadata.source_file
                            ? "Deleting..."
                            : "Delete"}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <div className="text-center text-gray-500 py-8">
            {loadingCollection
              ? "Loading documents..."
              : "No collection data available"}
          </div>
        )}
      </div>

      {/* Modal for viewing all chunks */}
      {selectedDoc && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
          onClick={() => setSelectedDoc(null)}
        >
          <div
            className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center p-6 border-b">
              <div>
                <h3 className="text-2xl font-bold text-gray-900">
                  {selectedDoc.metadata.filename}
                </h3>
                <p className="text-sm text-gray-500 mt-1">
                  {selectedDoc.metadata.chunk_count || 0} chunks
                </p>
              </div>
              <button
                onClick={() => setSelectedDoc(null)}
                className="text-gray-400 hover:text-gray-600 text-3xl leading-none"
              >
                ×
              </button>
            </div>

            <div className="overflow-y-auto p-6 flex-1">
              {selectedDoc.chunks && selectedDoc.chunks.length > 0 ? (
                <div className="space-y-4">
                  {selectedDoc.chunks.map((chunk, idx) => (
                    <div
                      key={chunk.id}
                      className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50"
                    >
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-xs font-semibold text-indigo-600">
                          Chunk {chunk.chunk_index + 1}
                        </span>
                        <span className="text-xs text-gray-500">
                          {chunk.content.length} characters
                        </span>
                      </div>
                      <div className="text-sm text-gray-700 whitespace-pre-wrap">
                        {chunk.content}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center text-gray-500 py-8">
                  No chunks available
                </div>
              )}
            </div>

            <div className="p-4 border-t bg-gray-50 flex justify-end">
              <button
                onClick={() => setSelectedDoc(null)}
                className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
