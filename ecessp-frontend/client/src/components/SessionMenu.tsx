import { useAppAuth } from "@/hooks/use-app-auth";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { LogOut, UserCircle2 } from "lucide-react";

export function SessionMenu() {
  const { authEnabled, authenticated, user, logout, logoutPending } = useAppAuth();

  if (!authEnabled) {
    return null;
  }

  if (!authenticated) {
    return null;
  }

  const username = user?.username || "user";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" className="gap-2">
          <UserCircle2 className="w-4 h-4" />
          <span className="text-sm">{username}</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>{username}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onClick={() => void logout()}
          disabled={logoutPending}
          className="gap-2"
        >
          <LogOut className="w-4 h-4" />
          {logoutPending ? "Signing out..." : "Sign out"}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
