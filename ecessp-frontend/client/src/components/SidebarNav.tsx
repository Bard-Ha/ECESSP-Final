import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import { 
  Atom, 
  Home,
  Search, 
  FlaskConical
} from "lucide-react";

const NAV_ITEMS = [
  {
    href: "/",
    label: "Home",
    icon: Home,
    description: "Platform overview",
  },
  { 
    href: "/discovery", 
    label: "Material Discovery", 
    icon: Search,
    description: "Find optimal systems" 
  },
  { 
    href: "/prediction", 
    label: "Performance Predictor", 
    icon: FlaskConical,
    description: "Simulate properties"
  },
];

export function SidebarNav() {
  const [location] = useLocation();

  return (
    <aside className="w-64 bg-sidebar border-r border-border h-screen flex flex-col fixed left-0 top-0 z-50">
      <div className="p-6 border-b border-border/50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <Atom className="w-5 h-5 text-primary" />
          </div>
          <h1 className="font-bold text-lg tracking-tight">ECESSP-ML</h1>
        </div>
        <p className="text-xs text-muted-foreground mt-1">Battery Informatics Platform</p>
      </div>

      <nav className="flex-1 p-4 space-y-2">
        {NAV_ITEMS.map((item) => {
          const isActive = location === item.href;
          const Icon = item.icon;
          
          return (
            <Link key={item.href} href={item.href}>
              <div
                className={cn(
                  "w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-all duration-200 cursor-pointer group",
                  isActive 
                    ? "bg-primary/10 text-primary shadow-sm" 
                    : "hover:bg-muted text-muted-foreground hover:text-foreground"
                )}
              >
                <Icon className={cn("w-5 h-5", isActive && "text-primary")} />
                <div>
                  <div className="font-medium text-sm">{item.label}</div>
                  <div className={cn(
                    "text-xs transition-colors",
                    isActive ? "text-primary/70" : "text-muted-foreground/60"
                  )}>
                    {item.description}
                  </div>
                </div>
              </div>
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-border/50 bg-muted/20">
        <div className="flex items-center gap-3 px-2">
          <div className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]" />
          <span className="text-xs font-mono text-muted-foreground">System Online</span>
        </div>
      </div>
    </aside>
  );
}
