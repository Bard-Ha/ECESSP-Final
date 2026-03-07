import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { API_BASE, resolveApiUrl } from "@/lib/api-base";

const TOKEN_KEY = "ecessp_app_token";

type AuthConfigResponse = {
  auth_enabled: boolean;
};

type LoginResponse = {
  token: string | null;
  token_type: string;
  expires_at: string | null;
};

type AuthMeResponse = {
  username: string;
  expires_at: string | null;
  auth_enabled?: boolean;
};

function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

function setStoredToken(token: string | null): void {
  if (typeof window === "undefined") return;
  if (!token) {
    window.localStorage.removeItem(TOKEN_KEY);
    return;
  }
  window.localStorage.setItem(TOKEN_KEY, token);
}

export function useAppAuth() {
  const [token, setToken] = useState<string | null>(getStoredToken);

  useEffect(() => {
    setStoredToken(token);
  }, [token]);

  const authConfig = useQuery({
    queryKey: ["auth-config"],
    queryFn: async (): Promise<AuthConfigResponse> => {
      const res = await fetch(resolveApiUrl(`${API_BASE}/auth/config`));
      if (!res.ok) throw new Error(`Auth config failed (${res.status})`);
      return res.json();
    },
    staleTime: 60_000,
  });

  const authEnabled = Boolean(authConfig.data?.auth_enabled);

  const authMe = useQuery({
    queryKey: ["auth-me", token, authEnabled],
    enabled: authEnabled && Boolean(token),
    queryFn: async (): Promise<AuthMeResponse> => {
      const res = await fetch(resolveApiUrl(`${API_BASE}/auth/me`), {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok) throw new Error(`Auth me failed (${res.status})`);
      return res.json();
    },
    retry: false,
  });

  useEffect(() => {
    if (authMe.isError && authEnabled) {
      setToken(null);
    }
  }, [authMe.isError, authEnabled]);

  const loginMutation = useMutation({
    mutationFn: async (credentials: { username: string; password: string }) => {
      const res = await fetch(resolveApiUrl(`${API_BASE}/auth/login`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(credentials),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Login failed (${res.status})`);
      }
      return (await res.json()) as LoginResponse;
    },
    onSuccess: (data) => {
      if (data.token) setToken(data.token);
    },
  });

  const logoutMutation = useMutation({
    mutationFn: async () => {
      await fetch(resolveApiUrl(`${API_BASE}/auth/logout`), {
        method: "POST",
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
    },
    onSettled: () => setToken(null),
  });

  const authenticated = useMemo(() => {
    if (!authEnabled) return true;
    if (!token) return false;
    return authMe.isSuccess;
  }, [authEnabled, token, authMe.isSuccess]);

  return {
    authEnabled,
    authConfigLoading: authConfig.isLoading,
    authenticated,
    user: authMe.data,
    token,
    login: loginMutation.mutateAsync,
    loginPending: loginMutation.isPending,
    loginError: loginMutation.error,
    logout: logoutMutation.mutateAsync,
    logoutPending: logoutMutation.isPending,
  };
}
