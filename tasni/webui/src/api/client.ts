// Thin fetch wrappers over the platform API. Same-origin in prod; Vite proxies
// /api -> :8000 in dev.

// A failed request carries the HTTP status and, when FastAPI raised with a dict
// `detail` (e.g. the scan module's `large_surface_required` 409), that structured
// payload — so a caller can branch on the state instead of parsing a message.
// `new Error(detail)` on an object used to stringify it to "[object Object]".
export interface ApiError extends Error {
  status: number;
  detail?: Record<string, any>;
}

async function unwrap(r: Response) {
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    const detail = body?.detail;
    const structured = detail && typeof detail === "object" ? detail : null;
    const err = new Error(
      (typeof detail === "string" && detail)
      || (typeof structured?.message === "string" && structured.message)
      || r.statusText) as ApiError;
    err.status = r.status;
    if (structured) err.detail = structured;
    throw err;
  }
  return r.json();
}

export const apiGet = <T = any>(path: string): Promise<T> =>
  fetch(path).then(unwrap);

export const apiPost = <T = any>(path: string, body?: unknown): Promise<T> =>
  fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  }).then(unwrap);

// Helper bound to one module's REST prefix.
export const moduleApi = (id: string) => ({
  get: <T = any>(p: string) => apiGet<T>(`/api/modules/${id}${p}`),
  post: <T = any>(p: string, b?: unknown) => apiPost<T>(`/api/modules/${id}${p}`, b),
});

export interface ModuleMeta {
  id: string;
  title: string;
  description: string;
  icon: string;
  order: number;
}
