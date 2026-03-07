import { useState } from "react";
import { useDiscover, createDiscoveryRequest, extractCandidateSystems, formatSystemForDisplay } from "@/hooks/use-ecessp-ml";
import { useToast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { PropertyChart } from "@/components/PropertyChart";
import {
  Zap, 
  Activity, 
  ShieldCheck, 
  Battery, 
  RefreshCw, 
  Sparkles,
  Search,
  AlertCircle
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { DiscoveryParams, TargetConfiguration } from "@shared/ecessp-ml-schemas";

// Types for form state
type Targets = TargetConfiguration;
type WorkingIon = "Li" | "Na" | "K" | "Mg" | "Ca" | "Zn" | "Al" | "Y";
type SourceMode = "existing" | "generated" | "hybrid";
type TuningParams = Required<
  Pick<
    DiscoveryParams,
    | "num_candidates"
    | "diversity_weight"
    | "novelty_weight"
    | "optimize_steps"
    | "interpolation_enabled"
    | "extrapolation_enabled"
    | "material_source_mode"
    | "component_source_mode"
    | "separator_options_count"
    | "additive_options_count"
  >
>;

const REALISTIC_LIMITS = {
  average_voltage: { min: 1.0, max: 4.4, step: 0.05 },
  capacity_grav: { min: 60, max: 120, step: 1 },
  capacity_vol: { min: 150, max: 260, step: 1 },
  energy_grav: { min: 120, max: 450, step: 1 },
  energy_vol: { min: 400, max: 1200, step: 1 },
  max_delta_volume: { min: 0.02, max: 0.25, step: 0.005 },
  stability_charge: { min: 1.0, max: 4.4, step: 0.05 },
  stability_discharge: { min: 1.0, max: 4.4, step: 0.05 },
} as const;

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

function withDerivedEnergy(targets: Targets): Targets {
  const energyGravRaw = Number((targets.average_voltage * targets.capacity_grav).toFixed(2));
  const energyVolRaw = Number((targets.average_voltage * targets.capacity_vol).toFixed(2));
  const energyGrav = clamp(energyGravRaw, REALISTIC_LIMITS.energy_grav.min, REALISTIC_LIMITS.energy_grav.max);
  const energyVol = clamp(energyVolRaw, REALISTIC_LIMITS.energy_vol.min, REALISTIC_LIMITS.energy_vol.max);
  return {
    ...targets,
    energy_grav: energyGrav,
    energy_vol: energyVol,
  };
}

const INITIAL_TARGETS: Targets = withDerivedEnergy({
  average_voltage: 3.6,
  capacity_grav: 95,
  capacity_vol: 220,
  energy_grav: 0,
  energy_vol: 0,
  max_delta_volume: 0.12,
  stability_charge: 4.1,
  stability_discharge: 3.0
});

const INITIAL_TUNING: TuningParams = {
  num_candidates: 48,
  diversity_weight: 0.4,
  novelty_weight: 0.3,
  optimize_steps: 24,
  interpolation_enabled: true,
  extrapolation_enabled: true,
  material_source_mode: "hybrid",
  component_source_mode: "hybrid",
  separator_options_count: 3,
  additive_options_count: 3,
};

const WORKING_ION_OPTIONS: Array<{ value: WorkingIon; label: string; maturity: string }> = [
  { value: "Li", label: "Li (Lithium)", maturity: "mature" },
  { value: "Na", label: "Na (Sodium)", maturity: "scaling" },
  { value: "K", label: "K (Potassium)", maturity: "exploratory" },
  { value: "Mg", label: "Mg (Magnesium)", maturity: "exploratory" },
  { value: "Ca", label: "Ca (Calcium)", maturity: "exploratory" },
  { value: "Zn", label: "Zn (Zinc)", maturity: "aqueous-focused" },
  { value: "Al", label: "Al (Aluminum)", maturity: "research" },
  { value: "Y", label: "Y (Yttrium)", maturity: "research" },
];

const WORKING_ION_STACK_HINTS: Record<WorkingIon, {
  hostFamilies: string;
  electrolyte: string;
  separator: string;
  additive: string;
}> = {
  Li: {
    hostFamilies: "Layered oxides, olivines, spinels, graphite/silicon-carbon",
    electrolyte: "1M LiPF6 in EC/EMC",
    separator: "PP/PE trilayer (16-20 um)",
    additive: "FEC + VC",
  },
  Na: {
    hostFamilies: "Prussian blue analogs, layered NaMO2, hard-carbon anodes",
    electrolyte: "NaPF6 in EC/DEC or diglyme",
    separator: "Ceramic-coated PP/PE",
    additive: "FEC + NaDFOB",
  },
  K: {
    hostFamilies: "Layered oxides, polyanion phosphates, hard-carbon hosts",
    electrolyte: "KPF6 in carbonate or ether solvents",
    separator: "Microporous PP/PE",
    additive: "FEC + KFSI stabilizer",
  },
  Mg: {
    hostFamilies: "Chevrel phases, spinel oxides, titanates",
    electrolyte: "Mg(TFSI)2 in glyme-based solvents",
    separator: "Glass fiber or ceramic-coated polymer",
    additive: "Interphase stabilizer package",
  },
  Ca: {
    hostFamilies: "Polyanion cathodes, titanates, Prussian-blue analogs",
    electrolyte: "Ca(TFSI)2 in carbonate/ether blends",
    separator: "Ceramic-coated polymer",
    additive: "SEI-promoting additive blend",
  },
  Zn: {
    hostFamilies: "MnO2 / V-oxide / Prussian-blue hosts (often aqueous)",
    electrolyte: "ZnSO4 or Zn(TFSI)2 aqueous blend",
    separator: "Glass fiber or cellulose separator",
    additive: "Mn2+ / interface stabilizer additives",
  },
  Al: {
    hostFamilies: "Graphitic hosts, sulfides, polyanion frameworks",
    electrolyte: "Chloroaluminate ionic liquid systems",
    separator: "High-stability polymer membrane",
    additive: "Corrosion inhibitor package",
  },
  Y: {
    hostFamilies: "Exploratory oxide/phosphate host frameworks",
    electrolyte: "Research electrolyte screening required",
    separator: "Ceramic-coated separator (exploratory)",
    additive: "Research additive screening required",
  },
};

const PROPERTY_CONFIG = [
  { key: "average_voltage", label: "Average Voltage", unit: "V", min: REALISTIC_LIMITS.average_voltage.min, max: REALISTIC_LIMITS.average_voltage.max, step: REALISTIC_LIMITS.average_voltage.step, icon: Zap },
  { key: "capacity_grav", label: "Gravimetric Capacity", unit: "mAh/g", min: REALISTIC_LIMITS.capacity_grav.min, max: REALISTIC_LIMITS.capacity_grav.max, step: REALISTIC_LIMITS.capacity_grav.step, icon: Battery },
  { key: "capacity_vol", label: "Volumetric Capacity", unit: "mAh/cm^3", min: REALISTIC_LIMITS.capacity_vol.min, max: REALISTIC_LIMITS.capacity_vol.max, step: REALISTIC_LIMITS.capacity_vol.step, icon: Battery },
  { key: "energy_grav", label: "Gravimetric Energy", unit: "Wh/kg", min: REALISTIC_LIMITS.energy_grav.min, max: REALISTIC_LIMITS.energy_grav.max, step: REALISTIC_LIMITS.energy_grav.step, icon: Sparkles },
  { key: "energy_vol", label: "Volumetric Energy", unit: "Wh/L", min: REALISTIC_LIMITS.energy_vol.min, max: REALISTIC_LIMITS.energy_vol.max, step: REALISTIC_LIMITS.energy_vol.step, icon: Sparkles },
  { key: "max_delta_volume", label: "Max Delta Volume", unit: "dV/V", min: REALISTIC_LIMITS.max_delta_volume.min, max: REALISTIC_LIMITS.max_delta_volume.max, step: REALISTIC_LIMITS.max_delta_volume.step, icon: Activity },
  { key: "stability_charge", label: "Charge Stability", unit: "V", min: REALISTIC_LIMITS.stability_charge.min, max: REALISTIC_LIMITS.stability_charge.max, step: REALISTIC_LIMITS.stability_charge.step, icon: ShieldCheck },
  { key: "stability_discharge", label: "Discharge Stability", unit: "V", min: REALISTIC_LIMITS.stability_discharge.min, max: REALISTIC_LIMITS.stability_discharge.max, step: REALISTIC_LIMITS.stability_discharge.step, icon: ShieldCheck },
] as const;

export default function DiscoveryEcesspMl() {
  const [targets, setTargets] = useState<Targets>(INITIAL_TARGETS);
  const [tuning, setTuning] = useState<TuningParams>(INITIAL_TUNING);
  const [workingIon, setWorkingIon] = useState<WorkingIon>("Li");
  const [ionScope, setIonScope] = useState<WorkingIon[]>(["Li"]);
  const [selectedSystemId, setSelectedSystemId] = useState<string | null>(null);
  const { toast } = useToast();
  
  const discover = useDiscover();
  const CONTROL_PROPERTY_KEYS = new Set(["average_voltage", "capacity_grav", "capacity_vol", "max_delta_volume", "stability_charge", "stability_discharge"]);
  const discoveryMetadata = (discover.data?.metadata ?? {}) as Record<string, any>;
  const discoveryReport = (discoveryMetadata.discovery_report_card ?? {}) as Record<string, any>;
  const resolvedDiscoveryParams = (discoveryMetadata.discovery_params ?? {}) as Record<string, any>;
  const queueInfo = (discoveryMetadata.active_learning_queue ?? {}) as Record<string, any>;
  const candidateSystems = discover.data ? extractCandidateSystems(discover.data) : [];
  const rankingModeRaw = String(discoveryReport.stage_6_7?.ranking_mode ?? "n/a");
  const rankingModeDisplay = rankingModeRaw === "n/a" ? rankingModeRaw : rankingModeRaw.replace(/_/g, " ");
  const selectedIonOption = WORKING_ION_OPTIONS.find((opt) => opt.value === workingIon);
  const selectedIonStackHint = WORKING_ION_STACK_HINTS[workingIon];
  const candidateSummaryText = discover.data
    ? candidateSystems.length > 0
      ? `Showing ${candidateSystems.length} ranked candidate system${candidateSystems.length === 1 ? "" : "s"} (max shown: 15).`
      : "No ranked candidates returned for this run."
    : "Run discovery to generate and rank battery candidates.";
  const fmtPct = (value: unknown) => {
    if (typeof value !== "number") return "n/a";
    return `${(value * 100).toFixed(1)}%`;
  };
  
  const handleSliderChange = (key: keyof Targets, value: number[]) => {
    setTargets(prev => withDerivedEnergy({ ...prev, [key]: value[0] }));
  };

  const handleTuningChange = (key: keyof TuningParams, value: number[]) => {
    const intFields = new Set<keyof TuningParams>([
      "optimize_steps",
      "num_candidates",
      "separator_options_count",
      "additive_options_count",
    ]);
    const nextValue =
      intFields.has(key)
        ? Math.round(value[0])
        : Number(value[0].toFixed(2));
    setTuning(prev => ({ ...prev, [key]: nextValue }));
  };

  const handleSourceModeChange = (key: "material_source_mode" | "component_source_mode", value: string) => {
    const mode: SourceMode = value === "existing" || value === "generated" ? value : "hybrid";
    setTuning((prev) => ({ ...prev, [key]: mode }));
  };

  const handleGenerationModeToggle = (key: "interpolation_enabled" | "extrapolation_enabled") => {
    setTuning((prev) => {
      const next = !prev[key];
      if (key === "interpolation_enabled" && !next && !prev.extrapolation_enabled) {
        return { ...prev, interpolation_enabled: false, extrapolation_enabled: true };
      }
      if (key === "extrapolation_enabled" && !next && !prev.interpolation_enabled) {
        return { ...prev, interpolation_enabled: true, extrapolation_enabled: false };
      }
      return { ...prev, [key]: next };
    });
  };

  const handlePrimaryIonChange = (value: WorkingIon) => {
    setWorkingIon(value);
    setIonScope((prev) => {
      const without = prev.filter((ion) => ion !== value);
      return [value, ...without].slice(0, 4);
    });
  };

  const toggleIonScope = (ion: WorkingIon) => {
    setIonScope((prev) => {
      if (prev.includes(ion)) {
        if (prev.length <= 1) return prev;
        return prev.filter((v) => v !== ion);
      }
      if (prev.length >= 4) return prev;
      return [...prev, ion];
    });
  };

  const handleRunDiscovery = async () => {
    try {
      const request = createDiscoveryRequest(
        targets,
        "general",
        {
          ...tuning,
          working_ion_candidates: ionScope,
        },
        workingIon,
      );
      const response = await discover.mutateAsync(request);
      
      const candidateSystems = extractCandidateSystems(response);
      if (candidateSystems.length > 0) {
        const displaySystem = formatSystemForDisplay(candidateSystems[0], response);
        setSelectedSystemId(displaySystem.id);
        
        toast({
          title: "Discovery Complete",
          description: `Found ${candidateSystems.length} candidate system(s) across ion scope: ${ionScope.join(", ")}`,
          variant: "default"
        });
      }
    } catch (error) {
      console.error("Discovery failed:", error);
      toast({
        title: "Discovery Failed",
        description: error instanceof Error ? error.message : "An error occurred during discovery",
        variant: "destructive"
      });
    }
  };

  return (
    <div className="p-8 max-w-[1600px] mx-auto min-h-screen">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">
          ECESSP-ML Material Discovery
        </h1>
        <p className="text-muted-foreground mt-2 text-lg">
          Define your target parameters to identify optimal battery chemistry candidates using the ECESSP-ML backend.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 h-[calc(100vh-12rem)]">
        {/* LEFT PANEL: CONTROLS */}
        <Card className="lg:col-span-4 flex flex-col h-full border-border/60 shadow-lg">
          <CardHeader className="pb-4 border-b border-border/50 bg-muted/20">
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-primary" />
              Target Parameters
            </CardTitle>
            <CardDescription>
              Adjust sliders within physically bounded, near-term achievable ranges. These values are sent directly to the ECESSP-ML backend.
            </CardDescription>
          </CardHeader>
          <ScrollArea className="flex-1 px-6 py-6">
            <div className="space-y-8 pr-4">
              {PROPERTY_CONFIG.filter((prop) => CONTROL_PROPERTY_KEYS.has(prop.key)).map((prop) => {
                const Icon = prop.icon;
                const rawValue = targets[prop.key as keyof Targets];
                const sliderValue = typeof rawValue === "number" ? rawValue : prop.min;
                return (
                  <div key={prop.key} className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-sm font-medium">
                        <div className="p-1.5 rounded-md bg-primary/10 text-primary">
                          <Icon className="w-4 h-4" />
                        </div>
                        {prop.label}
                      </div>
                      <span className="font-mono text-sm font-bold bg-muted px-2 py-0.5 rounded text-foreground">
                        {sliderValue} <span className="text-muted-foreground font-normal text-xs">{prop.unit}</span>
                      </span>
                    </div>
                    <Slider
                      value={[sliderValue]}
                      min={prop.min}
                      max={prop.max}
                      step={prop.step}
                      onValueChange={(val) => handleSliderChange(prop.key as keyof Targets, val)}
                      className="py-2 cursor-pointer"
                    />
                  </div>
                );
              })}
              <div className="rounded-lg border border-border/60 bg-muted/20 p-4 space-y-2">
                <div className="text-xs uppercase tracking-wide text-muted-foreground font-semibold">Derived Targets</div>
                <div className="flex items-center justify-between text-sm">
                  <span>Gravimetric Energy</span>
                  <span className="font-mono font-semibold">{targets.energy_grav} Wh/kg</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>Volumetric Energy</span>
                  <span className="font-mono font-semibold">{targets.energy_vol} Wh/L</span>
                </div>
              </div>
              <div className="rounded-lg border border-border/60 bg-muted/20 p-4 space-y-4">
                <div className="text-xs uppercase tracking-wide text-muted-foreground font-semibold">Working Ion</div>
                <Select value={workingIon} onValueChange={(val) => handlePrimaryIonChange(val as WorkingIon)}>
                  <SelectTrigger className="h-10">
                    <SelectValue placeholder="Select working ion" />
                  </SelectTrigger>
                  <SelectContent>
                    {WORKING_ION_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label} ({option.maturity})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                    <span>Ion Search Scope (max 4)</span>
                    <span className="font-mono">{ionScope.length}/4</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {WORKING_ION_OPTIONS.map((option) => {
                      const selected = ionScope.includes(option.value);
                      return (
                        <button
                          key={option.value}
                          type="button"
                          onClick={() => toggleIonScope(option.value)}
                          className={cn(
                            "px-2 py-1 rounded border text-[11px] font-medium transition-colors",
                            selected
                              ? "border-primary bg-primary/10 text-primary"
                              : "border-border/70 bg-background/60 text-foreground hover:bg-muted/40"
                          )}
                        >
                          {option.value}
                        </button>
                      );
                    })}
                  </div>
                  <div className="text-[11px] text-muted-foreground">
                    Discovery will generate and rank candidates across: <span className="font-medium text-foreground">{ionScope.join(", ")}</span>
                  </div>
                </div>
                <div className="rounded border border-border/50 bg-background/60 px-3 py-2 space-y-1">
                  <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                    Suggested Stack Defaults
                  </div>
                  <div className="text-xs text-foreground">
                    <span className="font-semibold">Host families:</span> {selectedIonStackHint.hostFamilies}
                  </div>
                  <div className="text-xs text-foreground">
                    <span className="font-semibold">Electrolyte:</span> {selectedIonStackHint.electrolyte}
                  </div>
                  <div className="text-xs text-foreground">
                    <span className="font-semibold">Separator:</span> {selectedIonStackHint.separator}
                  </div>
                  <div className="text-xs text-foreground">
                    <span className="font-semibold">Additive:</span> {selectedIonStackHint.additive}
                  </div>
                  {selectedIonOption && selectedIonOption.maturity !== "mature" && (
                    <div className="text-[11px] text-muted-foreground pt-1">
                      Selected ion is {selectedIonOption.maturity}; expect lower hit rate and higher uncertainty.
                    </div>
                  )}
                </div>
              </div>
              <div className="rounded-lg border border-border/60 bg-muted/20 p-4 space-y-4">
                <div className="text-xs uppercase tracking-wide text-muted-foreground font-semibold">Generation Tuning</div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span>Candidate Pool</span>
                    <span className="font-mono">{tuning.num_candidates}</span>
                  </div>
                  <Slider
                    value={[tuning.num_candidates]}
                    min={20}
                    max={150}
                    step={2}
                    onValueChange={(val) => handleTuningChange("num_candidates", val)}
                    className="py-1"
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span>Diversity Weight</span>
                    <span className="font-mono">{tuning.diversity_weight.toFixed(2)}</span>
                  </div>
                  <Slider
                    value={[tuning.diversity_weight]}
                    min={0}
                    max={1}
                    step={0.05}
                    onValueChange={(val) => handleTuningChange("diversity_weight", val)}
                    className="py-1"
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span>Novelty Weight</span>
                    <span className="font-mono">{tuning.novelty_weight.toFixed(2)}</span>
                  </div>
                  <Slider
                    value={[tuning.novelty_weight]}
                    min={0}
                    max={1}
                    step={0.05}
                    onValueChange={(val) => handleTuningChange("novelty_weight", val)}
                    className="py-1"
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span>Latent Optimize Steps</span>
                    <span className="font-mono">{tuning.optimize_steps}</span>
                  </div>
                  <Slider
                    value={[tuning.optimize_steps]}
                    min={0}
                    max={64}
                    step={1}
                    onValueChange={(val) => handleTuningChange("optimize_steps", val)}
                    className="py-1"
                  />
                </div>
                <div className="space-y-2">
                  <div className="text-[11px] text-muted-foreground">Latent Generation Modes</div>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => handleGenerationModeToggle("interpolation_enabled")}
                      className={cn(
                        "px-2 py-1 rounded border text-[11px] font-medium transition-colors",
                        tuning.interpolation_enabled
                          ? "border-primary bg-primary/10 text-primary"
                          : "border-border/70 bg-background/60 text-foreground hover:bg-muted/40"
                      )}
                    >
                      Interpolation {tuning.interpolation_enabled ? "ON" : "OFF"}
                    </button>
                    <button
                      type="button"
                      onClick={() => handleGenerationModeToggle("extrapolation_enabled")}
                      className={cn(
                        "px-2 py-1 rounded border text-[11px] font-medium transition-colors",
                        tuning.extrapolation_enabled
                          ? "border-primary bg-primary/10 text-primary"
                          : "border-border/70 bg-background/60 text-foreground hover:bg-muted/40"
                      )}
                    >
                      Extrapolation {tuning.extrapolation_enabled ? "ON" : "OFF"}
                    </button>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="text-[11px] text-muted-foreground">Host Material Source</div>
                  <Select
                    value={tuning.material_source_mode}
                    onValueChange={(val) => handleSourceModeChange("material_source_mode", val)}
                  >
                    <SelectTrigger className="h-9">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="hybrid">Hybrid (existing + generated)</SelectItem>
                      <SelectItem value="existing">Existing catalog only</SelectItem>
                      <SelectItem value="generated">Generated/new only</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <div className="text-[11px] text-muted-foreground">Separator/Additive Source</div>
                  <Select
                    value={tuning.component_source_mode}
                    onValueChange={(val) => handleSourceModeChange("component_source_mode", val)}
                  >
                    <SelectTrigger className="h-9">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="hybrid">Hybrid (existing + generated)</SelectItem>
                      <SelectItem value="existing">Existing templates only</SelectItem>
                      <SelectItem value="generated">Generated templates only</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span>Separator Options</span>
                    <span className="font-mono">{tuning.separator_options_count}</span>
                  </div>
                  <Slider
                    value={[tuning.separator_options_count]}
                    min={1}
                    max={6}
                    step={1}
                    onValueChange={(val) => handleTuningChange("separator_options_count", val)}
                    className="py-1"
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span>Additive Options</span>
                    <span className="font-mono">{tuning.additive_options_count}</span>
                  </div>
                  <Slider
                    value={[tuning.additive_options_count]}
                    min={1}
                    max={6}
                    step={1}
                    onValueChange={(val) => handleTuningChange("additive_options_count", val)}
                    className="py-1"
                  />
                </div>
              </div>
            </div>
          </ScrollArea>
          <div className="p-6 border-t border-border/50 bg-card">
            <Button 
              size="lg" 
              className="w-full text-base font-semibold shadow-lg shadow-primary/20"
              onClick={handleRunDiscovery}
              disabled={discover.isPending}
            >
              {discover.isPending ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Running ECESSP-ML Discovery...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 mr-2" />
                  Run Discovery
                </>
              )}
            </Button>
            
            {discover.isPending && (
              <div className="mt-4 p-4 bg-muted rounded-lg">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <AlertCircle className="w-4 h-4" />
                  <span>Discovery in progress. This may take a moment as the ML model processes your request.</span>
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* RIGHT PANEL: MERGED CANDIDATES + ANALYSIS */}
        <div className="lg:col-span-8 flex flex-col h-full overflow-hidden">
          <Card className="flex-1 flex flex-col border-border/60 shadow-lg overflow-hidden">
            <CardHeader className="pb-3 border-b border-border/50 bg-muted/20 flex-shrink-0">
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-primary" />
                Candidate Systems
              </CardTitle>
              <CardDescription className="text-xs">
                {candidateSummaryText}
              </CardDescription>
            </CardHeader>
            {discover.data && (
              <div className="px-4 py-3 border-b border-border/40 bg-muted/10">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-5 gap-2 text-[11px]">
                  <div className="rounded border border-border/50 bg-background/60 px-2 py-1.5">
                    <div className="text-muted-foreground uppercase tracking-wide">Ranking</div>
                    <div className="font-mono leading-snug break-words whitespace-normal" title={rankingModeRaw}>
                      {rankingModeDisplay}
                    </div>
                  </div>
                  <div className="rounded border border-border/50 bg-background/60 px-2 py-1.5">
                    <div className="text-muted-foreground uppercase tracking-wide">Hit@K</div>
                    <div className="font-mono leading-snug break-words whitespace-normal">{fmtPct(discoveryReport.hit_rate_at_k)}</div>
                  </div>
                  <div className="rounded border border-border/50 bg-background/60 px-2 py-1.5">
                    <div className="text-muted-foreground uppercase tracking-wide">Validity</div>
                    <div className="font-mono leading-snug break-words whitespace-normal">{fmtPct(discoveryReport.constraint_validity_rate)}</div>
                  </div>
                  <div className="rounded border border-border/50 bg-background/60 px-2 py-1.5">
                    <div className="text-muted-foreground uppercase tracking-wide">Physics Viol.</div>
                    <div className="font-mono leading-snug break-words whitespace-normal">{fmtPct(discoveryReport.physics_violation_rate)}</div>
                  </div>
                  <div className="rounded border border-border/50 bg-background/60 px-2 py-1.5">
                    <div className="text-muted-foreground uppercase tracking-wide">AL Queue</div>
                    <div className="font-mono leading-snug break-words whitespace-normal">
                      {typeof queueInfo.queued_count === "number" ? queueInfo.queued_count : "n/a"}
                    </div>
                  </div>
                  <div className="rounded border border-border/50 bg-background/60 px-2 py-1.5">
                    <div className="text-muted-foreground uppercase tracking-wide">Ion Scope</div>
                    <div className="font-mono leading-snug break-words whitespace-normal">
                      {Array.isArray(resolvedDiscoveryParams.working_ion_candidates)
                        ? resolvedDiscoveryParams.working_ion_candidates.join(", ")
                        : "n/a"}
                    </div>
                  </div>
                  <div className="rounded border border-border/50 bg-background/60 px-2 py-1.5">
                    <div className="text-muted-foreground uppercase tracking-wide">Component Source</div>
                    <div className="font-mono leading-snug break-words whitespace-normal">
                      {String(resolvedDiscoveryParams.component_source_mode ?? "n/a")}
                    </div>
                  </div>
                </div>
              </div>
            )}
            <ScrollArea className="flex-1">
              {!discover.data ? (
                <div className="h-full flex flex-col items-center justify-center p-8 text-center text-muted-foreground min-h-[300px]">
                  <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
                    <Search className="w-8 h-8 opacity-50" />
                  </div>
                  <p className="text-sm">Run discovery to generate candidates from the ECESSP-ML backend.</p>
                </div>
              ) : candidateSystems.length === 0 ? (
                <div className="p-6 text-center text-muted-foreground text-sm">
                  No candidates found. Try adjusting your target parameters.
                </div>
              ) : (
                <div className="divide-y divide-border/40">
                  {candidateSystems.map((system, idx) => {
                      const displaySystem = formatSystemForDisplay(system, discover.data);
                      const isSelected = selectedSystemId === system.battery_id;
                      const isGenerated =
                        system.source === "latent_generated" ||
                        system.source === "generated" ||
                        system.source === "staged_pipeline" ||
                        system.speculative === true;
                      const isVariation = system.source === "variation_of_existing";
                      const sourceLabel =
                        system.source === "staged_pipeline"
                          ? "Staged-generated"
                          : isGenerated
                            ? "Latent-generated"
                            : isVariation
                              ? "Variation of existing"
                              : "Dataset match";
                      
                      return (
                        <div 
                          key={system.battery_id}
                          onClick={() => setSelectedSystemId(system.battery_id)}
                          className={cn(
                            "p-4 cursor-pointer transition-all hover:bg-muted/30",
                            isSelected ? "bg-primary/5 border-l-4 border-l-primary pl-3" : "pl-4"
                          )}
                        >
                          {/* Header: Rank + Name + Score */}
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-3">
                              <span className="flex items-center justify-center w-7 h-7 rounded-full bg-primary/15 text-primary text-sm font-bold font-mono">
                                {idx + 1}
                              </span>
                              <div>
                                <h4 className="font-bold text-foreground text-sm">{displaySystem.name}</h4>
                                <span className={cn(
                                  "text-[10px] font-medium px-1.5 py-0.5 rounded",
                                  isGenerated 
                                    ? "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300"
                                    : isVariation
                                      ? "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300"
                                      : "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300"
                                )}>
                                  {sourceLabel}
                                </span>
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="text-lg font-bold text-primary font-mono">
                                {displaySystem.score.toFixed(2)}
                              </div>
                              <div className="text-[9px] text-muted-foreground uppercase tracking-wide">Score</div>
                            </div>
                          </div>
                          <div className="flex flex-wrap items-center gap-1.5 mb-2">
                            {typeof displaySystem.diagnostics?.paretoRank === "number" && (
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300">
                                Pareto #{displaySystem.diagnostics.paretoRank}
                              </span>
                            )}
                            {typeof displaySystem.diagnostics?.objectiveAlignment === "number" && (
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-sky-100 text-sky-700 dark:bg-sky-900/30 dark:text-sky-300">
                                Obj {(displaySystem.diagnostics.objectiveAlignment * 100).toFixed(1)}%
                              </span>
                            )}
                            {typeof displaySystem.diagnostics?.feasibility === "number" && (
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300">
                                Feas {(displaySystem.diagnostics.feasibility * 100).toFixed(1)}%
                              </span>
                            )}
                            {typeof displaySystem.diagnostics?.uncertaintyPenalty === "number" && (
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300">
                                Unc {(displaySystem.diagnostics.uncertaintyPenalty * 100).toFixed(1)}%
                              </span>
                            )}
                            {displaySystem.diagnostics?.compatibilitySource && (
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-300">
                                {displaySystem.diagnostics.compatibilitySource}
                              </span>
                            )}
                          </div>

                          {/* Components Grid */}
                          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-5 gap-2 mb-3">
                            <div className="bg-muted/40 rounded p-2 text-center">
                              <div className="text-[9px] uppercase tracking-wider text-muted-foreground mb-1">Cathode</div>
                              <div className="text-xs font-medium truncate" title={displaySystem.components.cathode}>
                                {displaySystem.components.cathode}
                              </div>
                            </div>
                            <div className="bg-muted/40 rounded p-2 text-center">
                              <div className="text-[9px] uppercase tracking-wider text-muted-foreground mb-1">Anode</div>
                              <div className="text-xs font-medium truncate" title={displaySystem.components.anode}>
                                {displaySystem.components.anode}
                              </div>
                            </div>
                            <div className="bg-muted/40 rounded p-2 text-center">
                              <div className="text-[9px] uppercase tracking-wider text-muted-foreground mb-1">Electrolyte</div>
                              <div className="text-xs font-medium truncate" title={displaySystem.components.electrolyte}>
                                {displaySystem.components.electrolyte}
                              </div>
                            </div>
                            <div className="bg-muted/40 rounded p-2 text-center">
                              <div className="text-[9px] uppercase tracking-wider text-muted-foreground mb-1">Separator</div>
                              <div className="text-[11px] font-medium break-words whitespace-normal leading-snug" title={displaySystem.components.separator}>
                                {displaySystem.components.separator}
                              </div>
                            </div>
                            <div className="bg-muted/40 rounded p-2 text-center">
                              <div className="text-[9px] uppercase tracking-wider text-muted-foreground mb-1">Additives</div>
                              <div className="text-[11px] font-medium break-words whitespace-normal leading-snug" title={displaySystem.components.additives}>
                                {displaySystem.components.additives}
                              </div>
                            </div>
                          </div>

                          {/* Analysis Section - Always visible */}
                          <div className="bg-muted/20 rounded-lg p-3 mb-3">
                            <h5 className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold mb-1.5 flex items-center gap-1">
                              <Activity className="w-3 h-3" />
                              Analysis
                            </h5>
                            <p className="text-xs text-muted-foreground leading-relaxed">
                              {displaySystem.explanation}
                            </p>
                          </div>

                          {/* Properties - Only show when selected */}
                          {isSelected && (
                            <div className="mt-2 space-y-3">
                              <div className="h-[180px]">
                                <PropertyChart data={displaySystem} />
                              </div>
                              <div className="grid grid-cols-4 gap-2">
                                {Object.entries(displaySystem.properties).slice(0, 4).map(([key, val]) => (
                                  <div key={key} className="bg-muted/30 p-2 rounded border border-border/50 text-center">
                                    <div className="text-[9px] uppercase tracking-wider text-muted-foreground font-semibold truncate">
                                      {PROPERTY_CONFIG.find(p => p.key === key)?.label?.replace(' Capacity', '')?.replace(' Energy', '')?.replace(' Stability', '') || key}
                                    </div>
                                    <div className="font-mono font-bold text-sm text-foreground">
                                      {typeof val === 'number' ? val.toFixed(1) : val}
                                    </div>
                                    <div className="text-[8px] text-muted-foreground">
                                      {PROPERTY_CONFIG.find(p => p.key === key)?.unit}
                                    </div>
                                  </div>
                                ))}
                              </div>
                              {(typeof system.cTheoretical === "number" || typeof system.capacityClipped === "number") && (
                                <div className="bg-muted/20 rounded border border-border/40 p-2">
                                  <div className="text-[9px] uppercase tracking-wider text-muted-foreground font-semibold mb-1">
                                    Physics-first checks
                                  </div>
                                  <div className="grid grid-cols-2 gap-1 text-[10px] font-mono">
                                    <div>C_theoretical</div>
                                    <div className="text-right">{typeof system.cTheoretical === "number" ? system.cTheoretical.toFixed(2) : "n/a"}</div>
                                    <div>Capacity clipped</div>
                                    <div className="text-right">{typeof system.capacityClipped === "number" ? system.capacityClipped.toFixed(2) : "no"}</div>
                                  </div>
                                </div>
                              )}
                              {(displaySystem.cellProperties || displaySystem.materialProperties) && (
                                <div className="mt-2 bg-muted/20 rounded border border-border/40 p-2">
                                  <div className="text-[9px] uppercase tracking-wider text-muted-foreground font-semibold mb-1">
                                    Cell-level (Authoritative)
                                  </div>
                                  <div className="grid grid-cols-2 gap-1 text-[10px] font-mono">
                                    <div>Cap (grav)</div>
                                    <div className="text-right">
                                      {displaySystem.cellProperties?.capacity_grav?.toFixed(1)
                                        ?? displaySystem.materialProperties?.capacity_grav?.toFixed(1)
                                        ?? "n/a"}
                                    </div>
                                    <div>Cap (vol)</div>
                                    <div className="text-right">
                                      {displaySystem.cellProperties?.capacity_vol?.toFixed(1)
                                        ?? displaySystem.materialProperties?.capacity_vol?.toFixed(1)
                                        ?? "n/a"}
                                    </div>
                                    <div>Energy (grav)</div>
                                    <div className="text-right">
                                      {displaySystem.cellProperties?.energy_grav?.toFixed(1)
                                        ?? displaySystem.materialProperties?.energy_grav?.toFixed(1)
                                        ?? "n/a"}
                                    </div>
                                    <div>Energy (vol)</div>
                                    <div className="text-right">
                                      {displaySystem.cellProperties?.energy_vol?.toFixed(1)
                                        ?? displaySystem.materialProperties?.energy_vol?.toFixed(1)
                                        ?? "n/a"}
                                    </div>
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                </div>
              )}
            </ScrollArea>
          </Card>
        </div>
      </div>
    </div>
  );
}




