import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SidebarNav } from "@/components/SidebarNav";
import { AuthGate } from "@/components/AuthGate";
import { SessionMenu } from "@/components/SessionMenu";
import NotFound from "@/pages/not-found";
import Home from "@/pages/Home";
import Discovery from "@/pages/Discovery";
import Prediction from "@/pages/Prediction";

function Router() {
  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <SidebarNav />
      <main className="flex-1 ml-64 transition-all duration-300">
        <header className="h-14 border-b border-border/50 px-6 flex items-center justify-end sticky top-0 bg-background/90 backdrop-blur z-40">
          <SessionMenu />
        </header>
        <Switch>
          <Route path="/" component={Home} />
          <Route path="/discovery" component={Discovery} />
          <Route path="/prediction" component={Prediction} />
          <Route component={NotFound} />
        </Switch>
      </main>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <AuthGate>
          <Router />
        </AuthGate>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
