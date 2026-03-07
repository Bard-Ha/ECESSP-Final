const RAW_API_BASE =
  (import.meta.env.VITE_ECESSP_API_BASE as string | undefined)?.trim() || "/api";

function normalizePath(path: string): string {
  if (!path) return "/";
  return path.startsWith("/") ? path : `/${path}`;
}

export function resolveApiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) {
    return path;
  }

  const normalizedPath = normalizePath(path);

  if (RAW_API_BASE === "/api") {
    if (normalizedPath === "/api" || normalizedPath.startsWith("/api/")) {
      return normalizedPath;
    }
    return `/api${normalizedPath}`;
  }

  const base = RAW_API_BASE.replace(/\/+$/, "");
  if (base.endsWith("/api") && (normalizedPath === "/api" || normalizedPath.startsWith("/api/"))) {
    return `${base}${normalizedPath.slice(4)}`;
  }
  return `${base}${normalizedPath}`;
}

export const API_BASE = RAW_API_BASE;
