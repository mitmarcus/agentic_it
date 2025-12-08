import { ImageResponse } from "next/og";

// Image metadata
export const size = {
  width: 32,
  height: 32,
};
export const contentType = "image/png";

// Image generation
export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          fontSize: 24,
          background: "#3B82F6",
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "white",
          borderRadius: "50%",
        }}
      >
        <svg
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M5 8C5 6.34315 6.34315 5 8 5H16C17.6569 5 19 6.34315 19 8V14C19 15.6569 17.6569 17 16 17H13L10 20V17H8C6.34315 17 5 15.6569 5 14V8Z"
            fill="white"
          />
          <rect x="8" y="10" width="8" height="1.5" rx="0.75" fill="#3B82F6" />
          <rect x="8" y="13" width="5" height="1.5" rx="0.75" fill="#3B82F6" />
        </svg>
      </div>
    ),
    {
      ...size,
    }
  );
}
