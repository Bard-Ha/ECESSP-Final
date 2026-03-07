import { useState } from "react";
import { useAppAuth } from "@/hooks/use-app-auth";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

type Props = {
  children: React.ReactNode;
};

export function AuthGate({ children }: Props) {
  const {
    authEnabled,
    authConfigLoading,
    authenticated,
    login,
    loginPending,
    loginError,
  } = useAppAuth();

  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [localError, setLocalError] = useState<string | null>(null);

  if (authConfigLoading) {
    return <div className="p-8">Loading authentication settings...</div>;
  }

  if (!authEnabled || authenticated) {
    return <>{children}</>;
  }

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);
    try {
      await login({ username, password });
    } catch (err: any) {
      setLocalError(err?.message || "Login failed");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Sign In</CardTitle>
          <CardDescription>
            Enter credentials to access discovery and prediction routes.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form className="space-y-4" onSubmit={submit}>
            <Input
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
            />
            <Input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
            />
            <Button type="submit" className="w-full" disabled={loginPending}>
              {loginPending ? "Signing in..." : "Sign in"}
            </Button>
            {localError ? <p className="text-sm text-red-600">{localError}</p> : null}
            {loginError ? <p className="text-sm text-red-600">{String(loginError)}</p> : null}
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

