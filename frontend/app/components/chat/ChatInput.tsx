"use client";
import { useState, useRef } from "react";
import { AttachedFile } from "../../types/chat";

export function ChatInput({
  isLoading,
  onSend,
}: {
  isLoading: boolean;
  onSend: (text: string, files?: AttachedFile[]) => Promise<void> | void;
}) {
  const [input, setInput] = useState("");
  const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if ((!text && attachedFiles.length === 0) || isLoading) return;

    const filesToSend = [...attachedFiles];
    setInput("");
    setAttachedFiles([]);

    await onSend(text || "Please analyze these files", filesToSend);
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);

    try {
      const uploadPromises = Array.from(files).map(async (file) => {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("http://localhost:8080/upload", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Failed to upload ${file.name}`);
        }

        const data = await response.json();
        return {
          fileId: data.file_id,
          filename: data.filename,
          fileType: data.file_type,
          sizeBytes: data.size_bytes,
          contentPreview: data.content_preview,
        } as AttachedFile;
      });

      const uploadedFiles = await Promise.all(uploadPromises);
      setAttachedFiles((prev) => [...prev, ...uploadedFiles]);
    } catch (error) {
      console.error("Error uploading files:", error);
      alert("Failed to upload one or more files. Please try again.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const removeFile = (fileId: string) => {
    setAttachedFiles((prev) => prev.filter((f) => f.fileId !== fileId));
  };

  return (
    <div className="bg-slate-50 border-t border-slate-300/20">
      {/* Attached Files Display */}
      {attachedFiles.length > 0 && (
        <div className="px-6 pt-3 flex flex-wrap gap-2">
          {attachedFiles.map((file) => (
            <div
              key={file.fileId}
              className="flex items-center gap-2 px-3 py-1.5 bg-blue-100 text-blue-800 rounded-full text-sm"
            >
              <span className="font-medium">📎 {file.filename}</span>
              <span className="text-xs text-blue-600">
                ({(file.sizeBytes / 1024).toFixed(1)}KB)
              </span>
              <button
                onClick={() => removeFile(file.fileId)}
                className="ml-1 text-blue-600 hover:text-blue-800 font-bold"
                type="button"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="flex gap-3 px-6 py-5">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          className="hidden"
          multiple
          accept=".pdf,.docx,.doc,.txt,.md,.jpg,.jpeg,.png,.gif,.bmp,.tiff"
        />

        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            attachedFiles.length > 0
              ? "Ask about the attached files..."
              : "Ask me about IT support..."
          }
          className="flex-1 px-4 py-3.5 border border-slate-400/50 rounded-full text-[0.95rem] outline-none transition-all duration-200 bg-white text-slate-900 placeholder:text-slate-400 focus:border-blue-600 focus:shadow-[0_0_0_3px_rgba(37,99,235,0.15)] disabled:bg-slate-100 disabled:cursor-not-allowed"
          disabled={isLoading || uploading}
        />

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="px-4 py-3.5 bg-slate-200 text-slate-700 border-none rounded-full text-lg cursor-pointer transition-all duration-200 hover:bg-slate-300 disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={isLoading || uploading}
          title="Attach file (PDF, DOCX, TXT, Image)"
        >
          {uploading ? "⏳" : "📎"}
        </button>

        <button
          type="submit"
          className="px-7 py-3.5 bg-blue-700 text-white border-none rounded-full text-lg cursor-pointer transition-all duration-200 min-w-[58px] shadow-[0_12px_24px_rgba(29,78,216,0.18)] hover:enabled:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none"
          disabled={
            isLoading ||
            uploading ||
            (!input.trim() && attachedFiles.length === 0)
          }
        >
          {isLoading ? "⏳" : "📤"}
        </button>
      </form>
    </div>
  );
}
