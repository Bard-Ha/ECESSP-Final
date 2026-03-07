import { cn } from "@/lib/utils";
import { type LucideIcon } from "lucide-react";

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  icon: LucideIcon;
  trend?: "up" | "down" | "neutral";
  description?: string;
  hint?: string;
  className?: string;
}

export function MetricCard({ 
  label, 
  value, 
  unit, 
  icon: Icon, 
  description,
  hint,
  className 
}: MetricCardProps) {
  return (
    <div className={cn(
      "p-5 rounded-2xl bg-card border border-border/50 shadow-sm hover:shadow-md transition-all duration-300",
      className
    )}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-muted-foreground">
            {label}
            {hint ? <span className="ml-2 text-[10px] text-muted-foreground/70">{hint}</span> : null}
          </p>
          <div className="mt-2 flex items-baseline gap-1">
            <h4 className="text-2xl font-bold tracking-tight text-foreground">{value}</h4>
            {unit && <span className="text-sm text-muted-foreground font-medium">{unit}</span>}
          </div>
        </div>
        <div className="p-2.5 rounded-xl bg-primary/10 text-primary">
          <Icon className="w-5 h-5" />
        </div>
      </div>
      {description && (
        <p className="mt-3 text-xs text-muted-foreground leading-relaxed">
          {description}
        </p>
      )}
    </div>
  );
}
