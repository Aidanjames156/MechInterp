const BASE = import.meta.env.VITE_API_URL ?? "/api";

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers ?? {}) },
    });
  } catch (err) {
    throw new ApiError(0, `cannot reach backend: ${err.message}`);
  }
  if (res.status === 204) return null;
  const text = await res.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!res.ok) {
    const detail = body && typeof body === "object" && "detail" in body ? body.detail : body;
    throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

const post = (path, body) => request(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () => request("/health"),
  listProbes: () => request("/probes"),
  getProbe: (id) => request(`/probes/${id}`),
  deleteProbe: (id) => request(`/probes/${id}`, { method: "DELETE" }),
  getPromptSet: (id) => request(`/prompt_sets/${id}`),
  extractProbe: (body) => post("/extract_probe", body),
  measure: (body) => post("/measure", body),
  steer: (body) => post("/generate_with_steering", body),
  listRuns: (limit = 20) => request(`/runs?limit=${limit}`),
};
