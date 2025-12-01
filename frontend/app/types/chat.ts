export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  intentType?: string; // The classified intent (e.g., "search_kb", "greeting")
  intentConfidence?: number; // Confidence in understanding what user wants
  decisionConfidence?: number; // Confidence in the action/answer quality
  responseType?: string;
  timestamp: number;
  userQuery?: string; // The user query this response answers (for feedback)
  retrievedDocIds?: string[]; // Document IDs used for this response (for feedback)
  feedback?: "positive" | "negative" | null; // Track feedback state
}

export interface FeedbackRequest {
  session_id: string;
  query: string;
  response: string;
  feedback_type: "positive" | "negative" | "neutral";
  feedback_score: number;
  feedback_comment?: string;
  retrieved_doc_ids?: string[]; // For feedback-aware retrieval
}

export interface FeedbackResponse {
  feedback_id: string;
  message: string;
  timestamp: string;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  action_taken: string;
  requires_followup: boolean;
  metadata?: {
    intent?: {
      intent?: string; // The classified intent category
      confidence?: number;
      [key: string]: unknown;
    };
    decision?: {
      action?: string;
      confidence?: number;
      reasoning?: string;
      [key: string]: unknown;
    };
    retrieved_docs_count?: number;
    retrieved_doc_ids?: string[]; // For feedback tracking
    turn_count?: number;
    [key: string]: unknown;
  };
}
