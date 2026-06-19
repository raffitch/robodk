// Thin fetch wrappers over the platform API. Same-origin in prod; Vite proxies
// /api -> :8000 in dev.

async function unwrap(r: Response) {
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail || r.statusText);
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
