/**
 * User Agent Detection Utility
 * 
 * Detects the user's operating system from the User-Agent string.
 * This information is sent to the backend to provide OS-specific support instructions.
 */

export interface UserAgentInfo {
  os: string;
}

/**
 * Detect the user's operating system
 * Uses browser's navigator.userAgent property
 */
function detectOS(): string {
  if (typeof window === "undefined") {
    return "Unknown";
  }

  const userAgent = window.navigator.userAgent;

  // Windows
  if (/Windows/.test(userAgent)) {
    return "Windows";
  }

  // macOS
  if (/Mac/.test(userAgent)) {
    return "macOS";
  }

  // Linux distributions
  if (/Linux/.test(userAgent)) {
    return "Linux";
  }

  // Chrome OS
  if (/CrOS/.test(userAgent)) {
    return "Chrome OS";
  }

  return "Unknown";
}

/**
 * Get user agent information with OS
 */
export function getUserAgentInfo(): UserAgentInfo {
  return {
    os: detectOS(),
  };
}
