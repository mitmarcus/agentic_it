export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  confidence?: number;
  responseType?: string;
  timestamp: number;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  action_taken: string;
  requires_followup: boolean;
  metadata?: {
    intent?: {
      confidence?: number;
      [key: string]: unknown; // Additional intent metadata
    };
    [key: string]: unknown; // file type and the likes
  };
}
