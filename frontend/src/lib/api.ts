import type { IdentifyResponse } from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export async function identifySkyImage(file: File): Promise<IdentifyResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/identify`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = "Identification failed.";
    try {
      const data = (await response.json()) as { detail?: string };
      if (data.detail) {
        message = data.detail;
      }
    } catch {
      // Ignore JSON parsing failures and keep the default message.
    }
    throw new Error(message);
  }

  return (await response.json()) as IdentifyResponse;
}
