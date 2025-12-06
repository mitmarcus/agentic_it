"use client";

import { useState, useRef, DragEvent } from "react";
import { postFormData } from "../../lib/api";

interface UploadedFile {
  file: File;
  status: "pending" | "uploading" | "success" | "error";
  error?: string;
}

interface FileUploadResponse {
  status: string;
  files_uploaded: number;
  files_failed: number;
  chunks_indexed: number;
  file_details: Array<{
    filename: string;
    size_bytes: number;
    chunks_created: number;
    status: string;
    error?: string;
  }>;
}

export function FileUpload() {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadResult, setUploadResult] = useState<FileUploadResponse | null>(
    null
  );
  const fileInputRef = useRef<HTMLInputElement>(null);

  const allowedExtensions = [".txt", ".md", ".html", ".pdf"];
  const maxFileSize = 100 * 1024 * 1024; // 100MB

  const handleFileSelect = (selectedFiles: FileList | null) => {
    if (!selectedFiles) return;

    const newFiles: UploadedFile[] = [];
    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];
      const ext = "." + file.name.split(".").pop()?.toLowerCase();

      if (!allowedExtensions.includes(ext)) {
        newFiles.push({
          file,
          status: "error",
          error: `Unsupported file type. Allowed: ${allowedExtensions.join(
            ", "
          )}`,
        });
      } else if (file.size > maxFileSize) {
        newFiles.push({
          file,
          status: "error",
          error: `File too large (${(file.size / 1024 / 1024).toFixed(
            1
          )}MB). Max: 100MB`,
        });
      } else if (file.size === 0) {
        newFiles.push({
          file,
          status: "error",
          error: "File is empty",
        });
      } else {
        newFiles.push({
          file,
          status: "pending",
        });
      }
    }

    setFiles((prev) => [...prev, ...newFiles]);
    setUploadResult(null);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const clearAll = () => {
    setFiles([]);
    setUploadResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const uploadFiles = async () => {
    const validFiles = files.filter((f) => f.status === "pending");
    if (validFiles.length === 0) return;

    setIsUploading(true);
    setUploadResult(null);

    try {
      const formData = new FormData();
      validFiles.forEach((f) => {
        formData.append("files", f.file);
      });

      // Update status to uploading
      setFiles((prev) =>
        prev.map((f) =>
          f.status === "pending" ? { ...f, status: "uploading" } : f
        )
      );

      const result = await postFormData<FileUploadResponse>(
        "/upload",
        formData
      );

      // Update file statuses based on result
      setFiles((prev) =>
        prev.map((f) => {
          if (f.status === "uploading") {
            const detail = result.file_details.find(
              (d: { filename: string }) => d.filename === f.file.name
            );
            if (detail) {
              if (detail.status === "indexed") {
                return { ...f, status: "success" };
              } else {
                return {
                  ...f,
                  status: "error",
                  error: detail.error || "Upload failed",
                };
              }
            }
            return { ...f, status: "success" };
          }
          return f;
        })
      );

      setUploadResult(result);
    } catch (error) {
      if (process.env.NODE_ENV === "development") {
        console.error("Upload error:", error);
      }
      setFiles((prev) =>
        prev.map((f) =>
          f.status === "uploading"
            ? { ...f, status: "error", error: "Upload failed" }
            : f
        )
      );
    } finally {
      setIsUploading(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1024 / 1024).toFixed(1) + " MB";
  };

  const pendingCount = files.filter((f) => f.status === "pending").length;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-800">Upload Documents</h2>
        {files.length > 0 && (
          <button
            onClick={clearAll}
            className="text-sm text-gray-600 hover:text-red-600 transition-colors"
            disabled={isUploading}
          >
            Clear All
          </button>
        )}
      </div>

      <div className="text-sm text-gray-600">
        Upload documents to add them to the knowledge base. Supports .txt, .md,
        .html, and .pdf files (max 100MB each).
      </div>

      {/* Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-all cursor-pointer ${
          isDragging
            ? "border-blue-500 bg-blue-50"
            : "border-gray-300 hover:border-gray-400"
        }`}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="text-gray-700 font-medium mb-1">
          Drop files here or click to browse
        </div>
        <div className="text-sm text-gray-500">
          Supported formats: .txt, .md, .html, .pdf (max 100MB)
        </div>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".txt,.md,.html,.pdf"
          onChange={(e) => handleFileSelect(e.target.files)}
          className="hidden"
        />
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="flex flex-col gap-2 max-h-80 overflow-y-auto">
          {files.map((fileItem, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200"
            >
              <div className="flex items-center gap-3 flex-1 min-w-0">
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-gray-800 truncate">
                    {fileItem.file.name}
                  </div>
                  <div className="text-xs text-gray-500">
                    {formatFileSize(fileItem.file.size)}
                    {fileItem.error && (
                      <span className="text-red-600 ml-2">
                        • {fileItem.error}
                      </span>
                    )}
                  </div>
                </div>
              </div>
              {fileItem.status === "pending" && !isUploading && (
                <button
                  onClick={() => removeFile(index)}
                  className="text-gray-400 hover:text-red-600 transition-colors ml-2"
                >
                  ×
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Upload Button */}
      {pendingCount > 0 && (
        <button
          onClick={uploadFiles}
          disabled={isUploading}
          className={`py-3 px-6 rounded-lg font-semibold transition-all ${
            isUploading
              ? "bg-gray-300 text-gray-500 cursor-not-allowed"
              : "bg-blue-600 text-white hover:bg-blue-700 shadow-md hover:shadow-lg"
          }`}
        >
          {isUploading
            ? "Uploading..."
            : `Upload ${pendingCount} file${pendingCount > 1 ? "s" : ""}`}
        </button>
      )}

      {/* Upload Result */}
      {uploadResult && (
        <div
          className={`p-4 rounded-lg ${
            uploadResult.status === "success"
              ? "bg-green-50 border border-green-200"
              : uploadResult.status === "partial_success"
              ? "bg-yellow-50 border border-yellow-200"
              : "bg-red-50 border border-red-200"
          }`}
        >
          <div className="font-semibold mb-2">
            {uploadResult.status === "success"
              ? "✅ Upload Successful!"
              : uploadResult.status === "partial_success"
              ? "⚠️ Partial Success"
              : "❌ Upload Failed"}
          </div>
          <div className="text-sm space-y-1">
            <div>
              📊 Files uploaded: {uploadResult.files_uploaded}
              {uploadResult.files_failed > 0 &&
                ` (${uploadResult.files_failed} failed)`}
            </div>
            <div>📝 Chunks indexed: {uploadResult.chunks_indexed}</div>
          </div>
        </div>
      )}
    </div>
  );
}
