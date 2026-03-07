import { 
  ResponsiveContainer, 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  Radar, 
  Tooltip,
  Legend
} from "recharts";
import type { CandidateSystem } from "@shared/schema";

interface PropertyChartProps {
  data: CandidateSystem;
  className?: string;
}

const PROPERTY_LABELS: Record<string, string> = {
  average_voltage: "Avg Voltage",
  capacity_grav: "Grav Cap",
  capacity_vol: "Vol Cap", 
  energy_grav: "Grav Energy",
  energy_vol: "Vol Energy",
  max_delta_volume: "Max dV/V",
  stability_charge: "Charge Stab",
  stability_discharge: "Disch Stab"
};

// Normalize data for radar chart (0-100 scale for visual comparison)
function normalizeValue(key: string, value: number) {
  // Rough max baselines for normalization based on battery properties
  const maxValues: Record<string, number> = {
    average_voltage: 4.4,
    capacity_grav: 120,
    capacity_vol: 260,
    energy_grav: 450,
    energy_vol: 1200,
    max_delta_volume: 0.25,
    stability_charge: 4.4,
    stability_discharge: 4.4
  };

  const max = maxValues[key] || 100;
  return Math.min(100, (value / max) * 100);
}

export function PropertyChart({ data, className }: PropertyChartProps) {
  const chartData = Object.entries(data.properties).map(([key, value]) => ({
    subject: PROPERTY_LABELS[key] || key,
    A: normalizeValue(key, value),
    rawValue: value,
    fullMark: 100,
  }));

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={chartData}>
          <PolarGrid stroke="hsl(var(--muted-foreground))" strokeOpacity={0.2} />
          <PolarAngleAxis 
            dataKey="subject" 
            tick={{ fill: "hsl(var(--foreground))", fontSize: 12, fontWeight: 500 }}
          />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
          <Radar
            name={data.name}
            dataKey="A"
            stroke="hsl(var(--primary))"
            strokeWidth={2}
            fill="hsl(var(--primary))"
            fillOpacity={0.3}
          />
          <Tooltip 
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const item = payload[0].payload;
                return (
                  <div className="bg-popover border border-border p-2 rounded-lg shadow-xl text-xs">
                    <p className="font-semibold text-foreground">{item.subject}</p>
                    <p className="text-muted-foreground">
                      Normalized: {Math.round(item.A)}%
                    </p>
                    <p className="text-primary font-mono mt-1">
                      Raw: {item.rawValue}
                    </p>
                  </div>
                );
              }
              return null;
            }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

