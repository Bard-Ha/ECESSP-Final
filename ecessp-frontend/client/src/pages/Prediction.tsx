import { useMemo, useState } from "react";
import { useMaterialsCatalog, usePredict } from "@/hooks/use-ecessp-ml";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { MetricCard } from "@/components/MetricCard";
import { FlaskConical, ArrowRight, Activity, Zap, Battery, ShieldCheck, Sparkles } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { motion } from "framer-motion";
import { Input } from "@/components/ui/input";

type ComponentType = "cathode" | "anode" | "electrolyte" | "separator" | "additives";
type FilterKey = "formulaOnly" | "mpOnly" | "liCompatible" | "naCompatible";
type MaterialOption = { material_id: string; name: string; formula?: string };

const COMPONENT_TYPES: ComponentType[] = [
  "cathode",
  "anode",
  "electrolyte",
  "separator",
  "additives",
];
const MAX_AUTOCOMPLETE_ITEMS = 6;

const PROPERTY_ICONS = {
  average_voltage: Zap,
  capacity_grav: Battery,
  capacity_vol: Battery,
  energy_grav: Sparkles,
  energy_vol: Sparkles,
  max_delta_volume: Activity,
  stability_charge: ShieldCheck,
  stability_discharge: ShieldCheck,
};

const PROPERTY_UNITS = {
  average_voltage: "V",
  capacity_grav: "mAh/g",
  capacity_vol: "mAh/cm3",
  energy_grav: "Wh/kg",
  energy_vol: "Wh/L",
  max_delta_volume: "fraction",
  stability_charge: "V",
  stability_discharge: "V",
};

const PROPERTY_LABELS = {
  average_voltage: "Average Voltage",
  capacity_grav: "Grav. Capacity",
  capacity_vol: "Vol. Capacity",
  energy_grav: "Grav. Energy",
  energy_vol: "Vol. Energy",
  max_delta_volume: "Max Delta Volume",
  stability_charge: "Charge Stability",
  stability_discharge: "Discharge Stability",
};

export default function Prediction() {
  const [components, setComponents] = useState<Record<ComponentType, string>>({
    cathode: "",
    anode: "",
    electrolyte: "",
    separator: "",
    additives: "",
  });

  const [queries, setQueries] = useState<Record<ComponentType, string>>({
    cathode: "",
    anode: "",
    electrolyte: "",
    separator: "",
    additives: "",
  });

  const [filters, setFilters] = useState<Record<ComponentType, Record<FilterKey, boolean>>>({
    cathode: { formulaOnly: false, mpOnly: false, liCompatible: false, naCompatible: false },
    anode: { formulaOnly: false, mpOnly: false, liCompatible: false, naCompatible: false },
    electrolyte: { formulaOnly: false, mpOnly: false, liCompatible: false, naCompatible: false },
    separator: { formulaOnly: false, mpOnly: false, liCompatible: false, naCompatible: false },
    additives: { formulaOnly: false, mpOnly: false, liCompatible: false, naCompatible: false },
  });

  const catalog = useMaterialsCatalog(300);

  const prediction = usePredict();
  const { toast } = useToast();
  const loadingMaterials = catalog.isLoading;

  const allMaterials = useMemo(
    () => (catalog.data?.items ?? []).slice().sort((a, b) => (a.name || a.material_id).localeCompare(b.name || b.material_id)),
    [catalog.data?.items]
  );

  const sortedOptionsByType = useMemo(() => {
    const matchesFilters = (item: MaterialOption, active: Record<FilterKey, boolean>) => {
      const haystack = `${item.name || ""} ${item.formula || ""} ${item.material_id}`.toLowerCase();
      if (active.formulaOnly && !(item.formula && item.formula.trim())) return false;
      if (active.mpOnly && !item.material_id.toLowerCase().startsWith("mp-")) return false;
      if (active.liCompatible && !haystack.includes("li")) return false;
      if (active.naCompatible && !haystack.includes("na")) return false;
      return true;
    };

    const sortAndFilter = (items: MaterialOption[], rawQuery: string, active: Record<FilterKey, boolean>) => {
      const q = rawQuery.trim().toLowerCase();
      return [...items]
        .filter((m) => matchesFilters(m, active))
        .filter((m) => {
          if (!q) return true;
          const name = (m.name || "").toLowerCase();
          const formula = (m.formula || "").toLowerCase();
          const id = m.material_id.toLowerCase();
          return name.includes(q) || formula.includes(q) || id.includes(q);
        })
        .sort((a, b) => {
        const aName = (a.name || "").toLowerCase();
        const bName = (b.name || "").toLowerCase();
        const aStarts = aName.startsWith(q) || a.material_id.toLowerCase().startsWith(q);
        const bStarts = bName.startsWith(q) || b.material_id.toLowerCase().startsWith(q);
        if (aStarts !== bStarts) return aStarts ? -1 : 1;
          return aName.localeCompare(bName);
      });
    };

    return {
      cathode: sortAndFilter(allMaterials, queries.cathode, filters.cathode),
      anode: sortAndFilter(allMaterials, queries.anode, filters.anode),
      electrolyte: sortAndFilter(allMaterials, queries.electrolyte, filters.electrolyte),
      separator: sortAndFilter(allMaterials, queries.separator, filters.separator),
      additives: sortAndFilter(allMaterials, queries.additives, filters.additives),
    };
  }, [allMaterials, queries, filters]);

  const autocompleteOptionsByType = useMemo(
    () => ({
      cathode: sortedOptionsByType.cathode.slice(0, MAX_AUTOCOMPLETE_ITEMS),
      anode: sortedOptionsByType.anode.slice(0, MAX_AUTOCOMPLETE_ITEMS),
      electrolyte: sortedOptionsByType.electrolyte.slice(0, MAX_AUTOCOMPLETE_ITEMS),
      separator: sortedOptionsByType.separator.slice(0, MAX_AUTOCOMPLETE_ITEMS),
      additives: sortedOptionsByType.additives.slice(0, MAX_AUTOCOMPLETE_ITEMS),
    }),
    [sortedOptionsByType]
  );

  const totalByType: Record<ComponentType, number> = {
    cathode: sortedOptionsByType.cathode.length,
    anode: sortedOptionsByType.anode.length,
    electrolyte: sortedOptionsByType.electrolyte.length,
    separator: sortedOptionsByType.separator.length,
    additives: sortedOptionsByType.additives.length,
  };

  const materialDisplay = (m: MaterialOption) =>
    `${m.name}${m.formula ? ` | ${m.formula}` : ""} (${m.material_id})`;

  const handleAutocompleteSelect = (type: ComponentType, m: MaterialOption) => {
    setComponents((prev) => ({ ...prev, [type]: m.material_id }));
    setQueries((prev) => ({ ...prev, [type]: m.name || m.formula || m.material_id }));
  };

  const handlePredict = () => {
    const missing = COMPONENT_TYPES.filter((c) => !components[c]);
    if (missing.length > 0) {
      toast({
        title: "Incomplete Selection",
        description: `Please select materials for: ${missing.join(", ")}.`,
        variant: "destructive",
      });
      return;
    }

    prediction.mutate(
      { components },
      {
        onError: (err: any) => {
          toast({
            title: "Prediction Error",
            description: err.message,
            variant: "destructive",
          });
        },
      }
    );
  };

  return (
    <div className="p-8 max-w-[1600px] mx-auto min-h-screen">
      <header className="mb-10">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">
          Performance Predictor
        </h1>
        <p className="text-muted-foreground mt-2 text-lg">
          Predict battery performance using cathode, anode, electrolyte, separator, and additives.
        </p>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
        <div className="xl:col-span-4 space-y-6">
          <Card className="border-border/60 shadow-lg">
            <CardHeader className="bg-muted/20 border-b border-border/50">
              <CardTitle className="flex items-center gap-2">
                <FlaskConical className="w-5 h-5 text-primary" />
                System Configuration
              </CardTitle>
              <CardDescription>
                Search and select from a curated 300-material catalog view.
              </CardDescription>
            </CardHeader>

            <CardContent className="p-6 space-y-6">
              {COMPONENT_TYPES.map((type) => (
                <div key={type} className="space-y-2">
                  <Label className="capitalize">{type} Material</Label>
                  <Input
                    placeholder={`Search ${type} by name, formula, or material ID...`}
                    value={queries[type]}
                    onChange={(e) =>
                      setQueries((prev) => ({ ...prev, [type]: e.target.value }))
                    }
                    onKeyDown={(e) => {
                      if (e.key !== "Enter") return;
                      const top = autocompleteOptionsByType[type][0];
                      if (!top) return;
                      e.preventDefault();
                      handleAutocompleteSelect(type, top);
                    }}
                    className="h-10"
                  />
                  {queries[type].trim().length > 0 && autocompleteOptionsByType[type].length > 0 && (
                    <div className="rounded-md border border-border/60 bg-popover shadow-sm">
                      {autocompleteOptionsByType[type].map((m) => (
                        <button
                          key={`ac-${type}-${m.material_id}`}
                          type="button"
                          className="w-full px-3 py-2 text-left text-sm hover:bg-accent/50"
                          onClick={() => handleAutocompleteSelect(type, m)}
                        >
                          {materialDisplay(m)}
                        </button>
                      ))}
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-2 rounded-md border border-border/60 p-2">
                    <label className="flex items-center gap-2 text-xs">
                      <Checkbox
                        checked={filters[type].formulaOnly}
                        onCheckedChange={(checked) =>
                          setFilters((prev) => ({
                            ...prev,
                            [type]: { ...prev[type], formulaOnly: checked === true },
                          }))
                        }
                      />
                      Formula only
                    </label>
                    <label className="flex items-center gap-2 text-xs">
                      <Checkbox
                        checked={filters[type].mpOnly}
                        onCheckedChange={(checked) =>
                          setFilters((prev) => ({
                            ...prev,
                            [type]: { ...prev[type], mpOnly: checked === true },
                          }))
                        }
                      />
                      MP IDs only
                    </label>
                    <label className="flex items-center gap-2 text-xs">
                      <Checkbox
                        checked={filters[type].liCompatible}
                        onCheckedChange={(checked) =>
                          setFilters((prev) => ({
                            ...prev,
                            [type]: { ...prev[type], liCompatible: checked === true },
                          }))
                        }
                      />
                      Li-compatible
                    </label>
                    <label className="flex items-center gap-2 text-xs">
                      <Checkbox
                        checked={filters[type].naCompatible}
                        onCheckedChange={(checked) =>
                          setFilters((prev) => ({
                            ...prev,
                            [type]: { ...prev[type], naCompatible: checked === true },
                          }))
                        }
                      />
                      Na-compatible
                    </label>
                  </div>
                  <Select
                    value={components[type]}
                    onValueChange={(v) =>
                      setComponents((prev) => ({ ...prev, [type]: v }))
                    }
                    disabled={loadingMaterials}
                  >
                    <SelectTrigger className="h-10">
                      <SelectValue placeholder={`Select ${type}...`} />
                    </SelectTrigger>
                    <SelectContent>
                      {sortedOptionsByType[type].map((m) => (
                        <SelectItem key={`${type}-${m.material_id}`} value={m.material_id}>
                          {materialDisplay(m)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    {totalByType[type]} match(es) from the 300-item catalog view (alphabetically ordered).
                  </p>
                </div>
              ))}

              <Button
                size="lg"
                className="w-full h-12 text-base font-semibold shadow-lg shadow-primary/20"
                onClick={handlePredict}
                disabled={prediction.isPending || loadingMaterials}
              >
                {prediction.isPending ? "Calculating..." : "Predict Performance"}
                {!prediction.isPending && <ArrowRight className="ml-2 w-5 h-5" />}
              </Button>
            </CardContent>
          </Card>

          <div className="p-4 rounded-xl bg-accent/10 border border-accent/20 text-accent-foreground text-sm">
            <h4 className="font-semibold mb-2 flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Catalog Scale
            </h4>
            <p className="opacity-80">
              A 300-item catalog view is loaded, sorted alphabetically, then filtered by search + checkmarks.
            </p>
          </div>
        </div>

        <div className="xl:col-span-8">
          {prediction.data ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="space-y-6"
            >
              <div className="flex items-center justify-between p-6 bg-card border border-border/60 rounded-2xl shadow-sm">
                <div>
                  <h2 className="text-xl font-bold">{prediction.data.system_name}</h2>
                  <p className="text-muted-foreground">Predicted configuration profile</p>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-1">
                    Confidence Score
                  </div>
                  <div className="text-3xl font-bold text-primary">
                    {(prediction.data.confidence_score * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-6 gap-3">
                <div className="rounded-lg border border-border/50 bg-card/50 p-3">
                  <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Model Score</div>
                  <div className="font-mono text-sm font-semibold">
                    {typeof prediction.data.score === "number" ? prediction.data.score.toFixed(3) : "n/a"}
                  </div>
                </div>
                <div className="rounded-lg border border-border/50 bg-card/50 p-3">
                  <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Valid</div>
                  <div className="font-mono text-sm font-semibold">
                    {prediction.data.diagnostics?.valid === true ? "true" : prediction.data.diagnostics?.valid === false ? "false" : "n/a"}
                  </div>
                </div>
                <div className="rounded-lg border border-border/50 bg-card/50 p-3">
                  <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Speculative</div>
                  <div className="font-mono text-sm font-semibold">
                    {prediction.data.diagnostics?.speculative === true ? "true" : prediction.data.diagnostics?.speculative === false ? "false" : "n/a"}
                  </div>
                </div>
                <div className="rounded-lg border border-border/50 bg-card/50 p-3">
                  <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Uncertainty</div>
                  <div className="font-mono text-sm font-semibold">
                    {typeof prediction.data.diagnostics?.uncertainty_penalty === "number"
                      ? `${(prediction.data.diagnostics.uncertainty_penalty * 100).toFixed(1)}%`
                      : "n/a"}
                  </div>
                </div>
                <div className="rounded-lg border border-border/50 bg-card/50 p-3">
                  <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Compatibility</div>
                  <div className="font-mono text-sm font-semibold">
                    {typeof prediction.data.diagnostics?.compatibility_score === "number"
                      ? prediction.data.diagnostics.compatibility_score.toFixed(3)
                      : "n/a"}
                  </div>
                </div>
                <div className="rounded-lg border border-border/50 bg-card/50 p-3">
                  <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Guardrail</div>
                  <div className="font-mono text-sm font-semibold">
                    {prediction.data.diagnostics?.guardrail_status || "n/a"}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(prediction.data.predicted_properties).map(([key, value]) => {
                  const propKey = key as keyof typeof PROPERTY_LABELS;
                  return (
                    <MetricCard
                      key={key}
                      label={PROPERTY_LABELS[propKey]}
                      value={typeof value === "number" ? value.toFixed(2) : value}
                      unit={PROPERTY_UNITS[propKey]}
                      icon={PROPERTY_ICONS[propKey]}
                      className="bg-card/50 backdrop-blur-sm"
                    />
                  );
                })}
              </div>
            </motion.div>
          ) : (
            <div className="h-full min-h-[400px] flex flex-col items-center justify-center border-2 border-dashed border-border/50 rounded-3xl bg-muted/5">
              <div className="w-20 h-20 bg-muted rounded-full flex items-center justify-center mb-6">
                <FlaskConical className="w-10 h-10 text-muted-foreground/50" />
              </div>
              <h3 className="text-xl font-semibold text-foreground">Ready to Predict</h3>
              <p className="text-muted-foreground mt-2 max-w-md text-center">
                Select all five component materials to run a prediction.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
