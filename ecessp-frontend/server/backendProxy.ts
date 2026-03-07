import { Router, Request, Response } from "express";
import fetch from "node-fetch";
import { randomBytes, timingSafeEqual } from "crypto";
import { config } from "./config";

const router = Router();
const PY_BACKEND_BASE = config.ecesspMlUrl;
const RETRYABLE_STATUS = new Set([502, 503, 504]);
const appTokens = new Map<string, { username: string; exp: number }>();

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function secureEquals(a: string, b: string): boolean {
  const ab = Buffer.from(a);
  const bb = Buffer.from(b);
  if (ab.length !== bb.length) return false;
  return timingSafeEqual(ab, bb);
}

function issueAppToken(username: string): { token: string; exp: number } {
  const token = randomBytes(32).toString("hex");
  const exp = Date.now() + config.authTokenTtlSec * 1000;
  appTokens.set(token, { username, exp });
  return { token, exp };
}

function parseBearer(authHeader?: string): string | null {
  if (!authHeader) return null;
  if (!authHeader.toLowerCase().startsWith("bearer ")) return null;
  return authHeader.slice(7).trim() || null;
}

function isAppTokenValid(token: string | null): boolean {
  if (!token) return false;
  const entry = appTokens.get(token);
  if (!entry) return false;
  if (Date.now() > entry.exp) {
    appTokens.delete(token);
    return false;
  }
  return true;
}

function requireAppAuth(req: Request, res: Response): boolean {
  if (!config.authEnabled) return true;
  const token = parseBearer(req.header("authorization"));
  if (!isAppTokenValid(token)) {
    res.status(401).json({ message: "Unauthorized" });
    return false;
  }
  return true;
}

type ProxyFetchOptions = {
  method: string;
  body?: string;
  headers?: Record<string, string>;
};

async function fetchWithRetry(path: string, init: ProxyFetchOptions, retries = config.proxyRetries) {
  let attempt = 0;
  let lastError: any = null;

  while (attempt <= retries) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), config.proxyTimeoutMs);

    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        ...(init.headers as Record<string, string> | undefined),
      };

      if (config.ecesspMlApiKey && !headers["x-api-key"]) {
        headers["x-api-key"] = config.ecesspMlApiKey;
      }

      const response = await fetch(`${PY_BACKEND_BASE}${path}`, {
        ...init,
        headers,
        signal: controller.signal,
      } as any);
      clearTimeout(timeout);

      if (RETRYABLE_STATUS.has(response.status) && attempt < retries) {
        attempt += 1;
        await sleep(200 * attempt);
        continue;
      }

      return response;
    } catch (err: any) {
      clearTimeout(timeout);
      lastError = err;
      if (attempt >= retries) {
        throw lastError;
      }
      attempt += 1;
      await sleep(200 * attempt);
    }
  }

  throw lastError || new Error("Unknown proxy error");
}

async function forwardJson(req: Request, res: Response, path: string, method: string) {
  try {
    const forwardHeaders: Record<string, string> = {};

    const requestApiKey = req.header("x-api-key");
    const requestAuth = req.header("authorization");

    if (config.forwardClientApiKey && requestApiKey) {
      forwardHeaders["x-api-key"] = requestApiKey;
    }
    if (config.forwardClientAuthHeader && requestAuth) {
      forwardHeaders["authorization"] = requestAuth;
    }

    if (!forwardHeaders["x-api-key"] && config.ecesspMlApiKey) {
      forwardHeaders["x-api-key"] = config.ecesspMlApiKey;
    }
    if (!forwardHeaders["authorization"] && config.ecesspMlBearerToken) {
      forwardHeaders["authorization"] = `Bearer ${config.ecesspMlBearerToken}`;
    }

    const response = await fetchWithRetry(path, {
      method,
      body: method === "GET" ? undefined : JSON.stringify(req.body),
      headers: forwardHeaders,
    });

    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : { message: await response.text() };

    res.status(response.status).json(payload);
  } catch (err: any) {
    const isTimeout = err?.name === "AbortError";
    res.status(isTimeout ? 504 : 503).json({ message: isTimeout ? "Proxy timeout" : err.message });
  }
}

router.get("/health", async (_req: Request, res: Response) => {
  await forwardJson(_req, res, "/api/health", "GET");
});

router.get("/runtime-diagnostics", async (req: Request, res: Response) => {
  if (!requireAppAuth(req, res)) return;
  await forwardJson(req, res, "/api/runtime-diagnostics", "GET");
});

router.get("/materials", async (req: Request, res: Response) => {
  if (!requireAppAuth(req, res)) return;
  const qIndex = req.url.indexOf("?");
  const queryString = qIndex >= 0 ? req.url.slice(qIndex) : "";
  await forwardJson(req, res, `/api/materials${queryString}`, "GET");
});

router.get("/auth/config", async (_req: Request, res: Response) => {
  return res.status(200).json({ auth_enabled: config.authEnabled });
});

router.post("/auth/login", async (req: Request, res: Response) => {
  if (!config.authEnabled) {
    return res.status(200).json({
      auth_enabled: false,
      token: null,
      token_type: "none",
      expires_at: null,
    });
  }

  const { username, password } = req.body || {};
  if (!username || !password) {
    return res.status(400).json({ message: "username and password are required" });
  }

  if (!secureEquals(String(username), config.authUsername) || !secureEquals(String(password), config.authPassword)) {
    return res.status(401).json({ message: "Invalid credentials" });
  }

  const issued = issueAppToken(String(username));
  return res.status(200).json({
    token: issued.token,
    token_type: "bearer",
    expires_at: new Date(issued.exp).toISOString(),
  });
});

router.post("/auth/logout", async (req: Request, res: Response) => {
  if (!config.authEnabled) {
    return res.status(200).json({ status: "ok", auth_enabled: false });
  }

  const token = parseBearer(req.header("authorization"));
  if (token) {
    appTokens.delete(token);
  }
  return res.status(200).json({ status: "ok" });
});

router.get("/auth/me", async (req: Request, res: Response) => {
  if (!config.authEnabled) {
    return res.status(200).json({
      username: "guest",
      expires_at: null,
      auth_enabled: false,
    });
  }

  const token = parseBearer(req.header("authorization"));
  if (!token || !isAppTokenValid(token)) {
    return res.status(401).json({ message: "Unauthorized" });
  }
  const entry = appTokens.get(token)!;
  return res.status(200).json({
    username: entry.username,
    expires_at: new Date(entry.exp).toISOString(),
  });
});

// Discover route - translate frontend format to Python backend format
router.post("/discover", async (req: Request, res: Response) => {
  try {
    if (!requireAppAuth(req, res)) return;

    const payload = req.body;

    // Canonical contract only:
    // { system: {...}, objective: { objectives: {...} }, ... }
    if (!payload?.system || !payload?.objective?.objectives) {
      return res.status(400).json({
        message: "Invalid /discover payload. Expected canonical schema with system and objective.objectives.",
      });
    }

    await forwardJson(req, res, "/api/discover", "POST");
  } catch (err: any) {
    res.status(500).json({ message: err.message });
  }
});

// Predict route - proxy directly to Python backend
router.post("/predict", async (req: Request, res: Response) => {
  if (!requireAppAuth(req, res)) return;
  await forwardJson(req, res, "/api/predict", "POST");
});

router.post("/discover-from-cif", async (req: Request, res: Response) => {
  if (!requireAppAuth(req, res)) return;
  await forwardJson(req, res, "/api/discover-from-cif", "POST");
});

export default router;
