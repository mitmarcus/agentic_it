"use client";

export function MessageBanner({
  type,
  text,
}: {
  type: "success" | "error";
  text: string;
}) {
  const bgColor = type === "success" ? "bg-green-100" : "bg-red-100";
  const textColor = type === "success" ? "text-green-900" : "text-red-900";
  const borderColor =
    type === "success" ? "border-l-green-600" : "border-l-red-600";

  return (
    <div
      className={`px-6 py-4 rounded-xl mb-8 font-medium shadow-[0_4px_12px_rgba(0,0,0,0.1)] border-l-4 ${bgColor} ${textColor} ${borderColor}`}
    >
      {text}
    </div>
  );
}
