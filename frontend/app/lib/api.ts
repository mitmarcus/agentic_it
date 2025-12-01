export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function postJSON<TReq extends object, TRes = any>(path: string, body: TReq): Promise<TRes> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Request failed ${res.status}: ${text}`);
  }
  return res.json();
}

export async function getJSON<TRes = any>(path: string): Promise<TRes> {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Request failed ${res.status}: ${text}`);
  }
  return res.json();
}

export async function postFormData<TRes = any>(path: string, formData: FormData): Promise<TRes> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    body: formData,
    // Note: Don't set Content-Type header, browser will set it with boundary for multipart/form-data
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Request failed ${res.status}: ${text}`);
  }
  return res.json();
}

// Feedback API
export async function submitFeedback(
  sessionId: string,
  query: string,
  response: string,
  feedbackType: "positive" | "negative",
  retrievedDocIds?: string[]
): Promise<{ feedback_id: string; message: string; timestamp: string }> {
  return postJSON("/feedback", {
    session_id: sessionId,
    query: query,
    response: response,
    feedback_type: feedbackType,
    feedback_score: feedbackType === "positive" ? 1 : -1,
    retrieved_doc_ids: retrievedDocIds,
  });
}
