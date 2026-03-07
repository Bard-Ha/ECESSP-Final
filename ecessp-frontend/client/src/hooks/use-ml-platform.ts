import { useQuery, useMutation } from "@tanstack/react-query";
import {
  api,
} from "../../../shared/routes";
import type { DiscoveryRequest, PredictionRequest } from "../../../shared/schema";
import { resolveApiUrl } from "@/lib/api-base";

// ============================================
// MATERIALS
// ============================================

export function useMaterials() {
  return useQuery({
    queryKey: ["materials"],
    queryFn: async () => {
      const res = await fetch(resolveApiUrl(api.materials.list.path));

      if (!res.ok) {
        throw new Error(`Failed to fetch materials (${res.status})`);
      }

      const json = await res.json();
      return api.materials.list.responses[200].parse(json);
    },
  });
}

// ============================================
// DISCOVERY
// ============================================

export function useDiscovery() {
  return useMutation({
    mutationFn: async (data: DiscoveryRequest) => {
      // Validate input against shared schema
      const validated = api.discover.input.parse(data);

      const res = await fetch(resolveApiUrl(api.discover.path), {
        method: api.discover.method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(validated),
      });

      const json = await res.json();

      if (!res.ok) {
        if (res.status === 400) {
          const err = api.discover.responses[400].parse(json);
          throw new Error(err.message);
        }

        throw new Error(
          json?.message || `Discovery failed (${res.status})`
        );
      }

      // Transform backend response to frontend format
      if (json.system) {
        // Backend returns single system, frontend expects array
        const transformedResponse = {
          systems: [{
            id: json.system.battery_id || "system_001",
            name: `System with ${json.system.working_ion || "Li"} ion`,
            components: {
              cathode: "Unknown",
              anode: "Unknown", 
              electrolyte: "Unknown"
            },
            properties: {
              average_voltage: json.system.average_voltage || 0,
              capacity_grav: json.system.capacity_grav || 0,
              capacity_vol: json.system.capacity_vol || 0,
              energy_grav: json.system.energy_grav || 0,
              energy_vol: json.system.energy_vol || 0,
              stability_charge: json.system.stability_charge || 0,
              stability_discharge: json.system.stability_discharge || 0
            },
            score: json.score?.score || 0.5,
            explanation: json.explanation?.summary || "System analysis complete."
          }]
        };
        return transformedResponse;
      }

      return json;
    },
  });
}

// ============================================
// PREDICTION
// ============================================

export function usePrediction() {
  return useMutation({
    mutationFn: async (data: PredictionRequest) => {
      const validated = api.predict.input.parse(data);

      const res = await fetch(resolveApiUrl(api.predict.path), {
        method: api.predict.method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(validated),
      });

      const json = await res.json();

      if (!res.ok) {
        if (res.status === 400) {
          const err = api.predict.responses[400].parse(json);
          throw new Error(err.message);
        }

        if (res.status === 404) {
          throw new Error(json?.message || "Material combination not found");
        }

        throw new Error(
          json?.message || `Prediction failed (${res.status})`
        );
      }

      return api.predict.responses[200].parse(json);
    },
  });
}
