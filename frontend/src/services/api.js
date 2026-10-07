// Thin fetch wrapper: JWT in localStorage, JSON errors surfaced as readable messages.
const TOKEN_KEY = "qv_token";
export const getToken = () => { try { return localStorage.getItem(TOKEN_KEY); } catch { return null; } };
export const setToken = (t) => { try { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); } catch { /* storage unavailable */ } };

let onUnauthorized = () => {};
export const setUnauthorizedHandler = (fn) => { onUnauthorized = fn; };

export class ApiError extends Error {
  constructor(message, status, code) { super(message); this.status = status; this.code = code; }
}

async function request(path, { method = "GET", body, form, params } = {}) {
  const url = new URL(path, window.location.origin);
  if (params) Object.entries(params).forEach(([k, v]) => v != null && v !== "" && url.searchParams.set(k, v));
  const headers = {};
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  let payload;
  if (form) payload = form;
  else if (body !== undefined) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(body); }
  let res;
  try {
    res = await fetch(url, { method, headers, body: payload });
  } catch {
    throw new ApiError("Cannot reach the QuantumVision server. Is the backend running?", 0, "network");
  }
  if (res.status === 401 && !path.includes("/auth/login")) onUnauthorized();
  if (!res.ok) {
    let detail = `Request failed (${res.status})`, code;
    try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); code = j.code; } catch { /* non-JSON */ }
    throw new ApiError(detail, res.status, code);
  }
  return res.status === 204 ? null : res.json();
}

const imageForm = (file, extra = {}) => {
  const f = new FormData();
  f.append("file", file);
  Object.entries(extra).forEach(([k, v]) => v != null && f.append(k, v));
  return f;
};

export const api = {
  login: async (email, password) => {
    const form = new URLSearchParams({ username: email, password });
    const res = await fetch("/api/auth/login", { method: "POST", body: form });
    const j = await res.json().catch(() => ({}));
    if (!res.ok) throw new ApiError(j.detail || "Login failed", res.status);
    return j;
  },
  me: () => request("/api/auth/me"),
  users: () => request("/api/auth/users"),
  createUser: (body) => request("/api/auth/users", { method: "POST", body }),
  health: () => request("/api/health"),
  emotions: () => request("/api/emotions"),
  predict: (file, model) => request("/api/predict", { method: "POST", form: imageForm(file), params: { model } }),
  predictMultiple: (file, model) => request("/api/predict/multiple", { method: "POST", form: imageForm(file), params: { model } }),
  analytics: (params) => request("/api/analytics", { params }),
  alerts: (params) => request("/api/alerts", { params }),
  acknowledge: (id, note) => request(`/api/alerts/${id}/acknowledge`, { method: "POST", body: { note } }),
  resolve: (id, note) => request(`/api/alerts/${id}/resolve`, { method: "POST", body: { note } }),
  cameras: () => request("/api/cameras"),
  createCamera: (body) => request("/api/cameras", { method: "POST", body }),
  modelInfo: () => request("/api/model/info"),
  modelMetrics: () => request("/api/model/metrics"),
  quantumCircuit: (params) => request("/api/quantum/circuit", { params }),
};

export function surveillanceSocketUrl(cameraId) {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws/surveillance/${encodeURIComponent(cameraId)}?token=${encodeURIComponent(getToken() || "")}`;
}
