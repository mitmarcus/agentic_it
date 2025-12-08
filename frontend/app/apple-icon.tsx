import { ImageResponse } from "next/og";

// Image metadata
export const size = {
  width: 180,
  height: 180,
};
export const contentType = "image/png";

// Image generation
export default function AppleIcon() {
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
          borderRadius: "22%",
        }}
      >
        <svg
          width="120"
          height="120"
          viewBox="0 0 120 120"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M25 40C25 31.7157 31.7157 25 40 25H80C88.2843 25 95 31.7157 95 40V70C95 78.2843 88.2843 85 80 85H65L50 100V85H40C31.7157 85 25 78.2843 25 70V40Z"
            fill="white"
          />
          <rect x="40" y="50" width="40" height="6" rx="3" fill="#3B82F6" />
          <rect x="40" y="64" width="25" height="6" rx="3" fill="#3B82F6" />
        </svg>
      </div>
    ),
    {
      ...size,
    }
  );
}
