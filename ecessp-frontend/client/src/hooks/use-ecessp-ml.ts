import { useQuery, useMutation } from "@tanstack/react-query";
import type { 
  DiscoveryRequest, 
  CifDiscoveryRequest, 
  DiscoveryResponse,
  PredictionRequest,
  PredictionResponse,
  BatterySystemResponse,
  RankedSystemsResponse,
  ExplanationResponse,
  ChatResponse,
  TargetConfiguration,
  DiscoveryParams
} from "../../../shared/ecessp-ml-schemas";
import { API_BASE, resolveApiUrl } from "@/lib/api-base";

// ============================================
// ECESSP-ML API Configuration
// ============================================

const APP_TOKEN_STORAGE_KEY = "ecessp_app_token";

export type MaterialCatalogItem = {
  material_id: string;
  name: string;
  formula?: string;
};

export type MaterialsCatalogResponse = {
  items: MaterialCatalogItem[];
  total: number;
  offset: number;
  limit: number;
  source: string;
};

export type DiscoveryCandidate = BatterySystemResponse & {
  rawSystem?: Record<string, any>;
  explanationText?: string;
  source?: string;
  valid?: boolean;
  paretoRank?: number;
  paretoScore?: number;
  objectiveAlignmentScore?: number;
  feasibilityScore?: number;
  uncertaintyPenalty?: number;
  compatibilitySource?: string;
  compatibilityHeadBlend?: number;
  uncertaintyModelWeight?: number;
  physicsFirst?: Record<string, any>;
  oxidationStates?: Record<string, number>;
  cTheoretical?: number;
  capacityClipped?: number;
};

function getAppAuthHeader(): Record<string, string> {
  if (typeof window === "undefined") return {};
  const token = window.localStorage.getItem(APP_TOKEN_STORAGE_KEY);
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
}

// ============================================
// API Client Functions
// ============================================

async function apiRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(resolveApiUrl(`${API_BASE}${endpoint}`), {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...getAppAuthHeader(),
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error: ${response.status} - ${errorText}`);
  }

  return response.json();
}

// ============================================
// DISCOVERY HOOKS
// ============================================

export function useHealthCheck() {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => apiRequest<{ status: string; runtime_ready: boolean; service: string }>("/health"),
    refetchInterval: 30000, // Check every 30 seconds
  });
}

export function useMaterialsCatalog(limit = 300) {
  return useQuery({
    queryKey: ["materials-catalog", limit],
    queryFn: () =>
      apiRequest<MaterialsCatalogResponse>(`/materials?limit=${limit}&offset=0`),
    staleTime: 5 * 60_000,
  });
}

export function useMaterialsSearch(query: string, limit = 150, offset = 0) {
  const normalized = query.trim();
  const params = new URLSearchParams({
    query: normalized,
    limit: String(limit),
    offset: String(offset),
  });

  return useQuery({
    queryKey: ["materials-search", normalized, limit, offset],
    queryFn: () => apiRequest<MaterialsCatalogResponse>(`/materials?${params.toString()}`),
    staleTime: 60_000,
  });
}

export function useDiscover() {
  return useMutation({
    mutationFn: (data: DiscoveryRequest) => 
      apiRequest<DiscoveryResponse>("/discover", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  });
}

export function useDiscoverFromCif() {
  return useMutation({
    mutationFn: (data: CifDiscoveryRequest) => 
      apiRequest<DiscoveryResponse>("/discover-from-cif", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  });
}

// ============================================
// PREDICTION HOOKS
// ============================================

export function usePredict() {
  return useMutation({
    mutationFn: (data: PredictionRequest) => 
      apiRequest<PredictionResponse>("/predict", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  });
}

// ============================================
// EXPLANATION HOOKS
// ============================================

export function useExplain() {
  return useMutation({
    mutationFn: (batteryId: string) => 
      apiRequest<ExplanationResponse>("/explain", {
        method: "POST",
        body: JSON.stringify({ battery_id: batteryId }),
      }),
  });
}

// ============================================
// CHAT HOOKS
// ============================================

export function useChat() {
  return useMutation({
    mutationFn: (data: { message: string; context?: any; contextSystemId?: number }) => 
      apiRequest<ChatResponse>("/chat", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  });
}

// ============================================
// SYSTEM MANAGEMENT HOOKS
// ============================================

export function useRankedSystems() {
  return useQuery({
    queryKey: ["ranked-systems"],
    queryFn: () => apiRequest<RankedSystemsResponse>("/systems/ranked"),
    enabled: false, // Only fetch when explicitly requested
  });
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

export function createDiscoveryRequest(
  targets: TargetConfiguration,
  application?: string,
  discoveryParams?: DiscoveryParams,
  workingIon: string = "Li",
): DiscoveryRequest {
  const ion = String(workingIon || "Li").trim() || "Li";
  const maxDeltaVolume =
    typeof targets.max_delta_volume === "number" ? targets.max_delta_volume : 0.12;
  const ionScopeRaw = discoveryParams?.working_ion_candidates;
  const ionScope =
    Array.isArray(ionScopeRaw) && ionScopeRaw.length > 0
      ? Array.from(new Set([ion, ...ionScopeRaw.map((v) => String(v).trim()).filter(Boolean)])).slice(0, 4)
      : [ion];
  const resolvedDiscoveryParams: DiscoveryParams = {
    ...(discoveryParams || {}),
    working_ion_candidates: ionScope,
  };
  return {
    system: {
      battery_id: `discovery_${Date.now()}`,
      working_ion: ion,
      elements: [ion],
      average_voltage: targets.average_voltage,
      capacity_grav: targets.capacity_grav,
      capacity_vol: targets.capacity_vol,
      energy_grav: targets.energy_grav,
      energy_vol: targets.energy_vol,
      max_delta_volume: maxDeltaVolume,
      stability_charge: targets.stability_charge,
      stability_discharge: targets.stability_discharge
    },
    objective: {
      objectives: {
        average_voltage: targets.average_voltage,
        capacity_grav: targets.capacity_grav,
        capacity_vol: targets.capacity_vol,
        energy_grav: targets.energy_grav,
        energy_vol: targets.energy_vol,
        max_delta_volume: maxDeltaVolume,
        stability_charge: targets.stability_charge,
        stability_discharge: targets.stability_discharge
      }
    },
    application: application || "general",
    explain: true,
    mode: "generative",
    discovery_params: resolvedDiscoveryParams,
  };
}

export function createPredictionRequest(
  components: Record<string, string>
): PredictionRequest {
  return {
    components
  };
}

export function extractCandidateSystems(response: DiscoveryResponse): DiscoveryCandidate[] {
  const fromHistory: DiscoveryCandidate[] = (response.history || [])
    .map((entry) => {
      const system = (entry?.system || {}) as Record<string, any>;
      const score = typeof entry?.score === "number" ? entry.score : response.score?.score;
      const constraints =
        (entry?.constraints as DiscoveryCandidate["constraints"]) ||
        response.metadata?.constraints;
      const explanationText =
        typeof entry?.explanation === "string" ? entry.explanation : undefined;
      const physicsFirst =
        entry?.physics_first && typeof entry.physics_first === "object"
          ? (entry.physics_first as Record<string, any>)
          : undefined;
      const materialLevel =
        system.material_level ??
        system.uncertainty?.material_level;
      const cellLevel =
        system.cell_level ??
        system.uncertainty?.cell_level;
      return {
        battery_id: system.battery_id || `candidate_${String(entry?.rank ?? Date.now())}`,
        working_ion: system.working_ion || "Li",
        elements: Array.isArray(system.elements) ? system.elements : ["Li"],
        properties: {
          average_voltage: cellLevel?.average_voltage ?? system.average_voltage,
          capacity_grav: cellLevel?.capacity_grav ?? system.capacity_grav,
          capacity_vol: cellLevel?.capacity_vol ?? system.capacity_vol,
          energy_grav: cellLevel?.energy_grav ?? system.energy_grav,
          energy_vol: cellLevel?.energy_vol ?? system.energy_vol,
          max_delta_volume: system.max_delta_volume,
          stability_charge: system.stability_charge,
          stability_discharge: system.stability_discharge,
          material_level: materialLevel,
          cell_level: cellLevel
        },
        constraints,
        score,
        valid: Boolean(entry?.valid),
        speculative: Boolean(entry?.speculative),
        rawSystem: system,
        explanationText,
        source: typeof entry?.source === "string" ? entry.source : undefined,
        paretoRank:
          typeof entry?.pareto_rank === "number" ? entry.pareto_rank : undefined,
        paretoScore:
          typeof entry?.pareto_score === "number" ? entry.pareto_score : undefined,
        objectiveAlignmentScore:
          typeof entry?.objective_alignment_score === "number"
            ? entry.objective_alignment_score
            : undefined,
        feasibilityScore:
          typeof entry?.feasibility_score === "number" ? entry.feasibility_score : undefined,
        uncertaintyPenalty:
          typeof entry?.uncertainty_penalty === "number" ? entry.uncertainty_penalty : undefined,
        compatibilitySource:
          typeof entry?.compatibility_source === "string" ? entry.compatibility_source : undefined,
        compatibilityHeadBlend:
          typeof entry?.compatibility_head_blend === "number"
            ? entry.compatibility_head_blend
            : undefined,
        uncertaintyModelWeight:
          typeof entry?.uncertainty_model_weight === "number"
            ? entry.uncertainty_model_weight
            : undefined,
        physicsFirst,
        oxidationStates:
          entry?.oxidation_states && typeof entry.oxidation_states === "object"
            ? (entry.oxidation_states as Record<string, number>)
            : undefined,
        cTheoretical:
          typeof entry?.C_theoretical_mAh_per_g === "number" ? entry.C_theoretical_mAh_per_g : undefined,
        capacityClipped:
          typeof entry?.capacity_grav_clipped === "number" ? entry.capacity_grav_clipped : undefined,
      };
    })
    .slice(0, 15);

  if (fromHistory.length > 0) {
    return fromHistory;
  }

  if (!response.system) {
    return [];
  }

  return [{
    battery_id: response.system.battery_id || "system_001",
    working_ion: response.system.working_ion || "Li",
    elements: response.system.elements || ["Li"],
    properties: {
      average_voltage: response.system.cell_level?.average_voltage ?? response.system.average_voltage,
      capacity_grav: response.system.cell_level?.capacity_grav ?? response.system.capacity_grav,
      capacity_vol: response.system.cell_level?.capacity_vol ?? response.system.capacity_vol,
      energy_grav: response.system.cell_level?.energy_grav ?? response.system.energy_grav,
      energy_vol: response.system.cell_level?.energy_vol ?? response.system.energy_vol,
      max_delta_volume: response.system.max_delta_volume,
      stability_charge: response.system.stability_charge,
      stability_discharge: response.system.stability_discharge,
      material_level: response.system.material_level ?? response.system.uncertainty?.material_level,
      cell_level: response.system.cell_level ?? response.system.uncertainty?.cell_level
    },
    constraints: response.metadata?.constraints,
    score: response.score?.score,
    speculative: response.score?.speculative || false,
    rawSystem: response.system,
  }];
}

export function formatSystemForDisplay(system: DiscoveryCandidate, response?: DiscoveryResponse): {
  id: string;
  name: string;
  components: {
    cathode: string;
    anode: string;
    electrolyte: string;
    separator: string;
    additives: string;
  };
  properties: Record<string, number>;
  materialProperties?: Record<string, number>;
  cellProperties?: Record<string, number>;
  score: number;
  explanation: string;
  diagnostics?: {
    paretoRank?: number;
    paretoScore?: number;
    objectiveAlignment?: number;
    feasibility?: number;
    uncertaintyPenalty?: number;
    compatibilitySource?: string;
  };
} {
  const systemRecord = ((system.rawSystem ?? response?.system) ?? {}) as Record<string, any>;
  const explanationRecord = (response?.explanation ?? {}) as Record<string, any>;
  const candidateExplanation =
    typeof system.explanationText === "string" ? system.explanationText.trim() : "";
  const safeNumber = (value: unknown): number => {
    const n = typeof value === "number" ? value : 0;
    return Object.is(n, -0) ? 0 : n;
  };
  
  // Enhanced display logic for different source types
  const isGenerated =
    system.source === "latent_generated" ||
    system.source === "generated" ||
    system.source === "staged_pipeline" ||
    system.speculative;
  const datasetUnknown = isGenerated ? "AI-generated candidate" : "Dataset not specified";

  // Cathode: try multiple field names
  const cathode =
    systemRecord.cathode ||
    systemRecord.cathode_material ||
    systemRecord.framework_formula ||
    systemRecord.battery_formula ||
    systemRecord.chemsys ||
    datasetUnknown;
    
  // Anode: try multiple field names including generated materials
  const anode =
    systemRecord.anode ||
    systemRecord.anode_material ||
    (systemRecord.working_ion || system.working_ion 
      ? `${systemRecord.working_ion || system.working_ion} + ${systemRecord.anode_material || 'anode material'}`
      : datasetUnknown);
      
  // Electrolyte
  const electrolyte = 
    systemRecord.electrolyte || 
    systemRecord.electrolyte_material ||
    (isGenerated ? "Optimized for target stability" : datasetUnknown);
    
  // Separator  
  const separator = 
    systemRecord.separator || 
    systemRecord.separator_material ||
    (isGenerated ? "Standard polymer separator" : datasetUnknown);
    
  // Additives
  const additives =
    systemRecord.additives ||
    systemRecord.additive ||
    systemRecord.additives_material ||
    systemRecord.additive_material ||
    (isGenerated ? "Performance-enhancing additives" : datasetUnknown);

  const summary =
    candidateExplanation
      ? candidateExplanation
      : (typeof explanationRecord.summary === "string" && explanationRecord.summary.trim())
      ? explanationRecord.summary.trim()
      : (typeof systemRecord.explanation === "string" && systemRecord.explanation.trim())
        ? systemRecord.explanation.trim()
        : "No model explanation was returned for this candidate.";

  const labelBasis =
    systemRecord.battery_formula ||
    systemRecord.framework_formula ||
    systemRecord.chemsys ||
    (Array.isArray(system.elements) && system.elements.length > 0 ? system.elements.join("-") : null) ||
    `${system.working_ion || "Li"} ion system`;

  const materialProps = system.properties.material_level;
  const cellProps = system.properties.cell_level;
  const displayProps = {
    average_voltage: safeNumber(system.properties.average_voltage),
    capacity_grav: safeNumber(system.properties.capacity_grav),
    capacity_vol: safeNumber(system.properties.capacity_vol),
    energy_grav: safeNumber(system.properties.energy_grav),
    energy_vol: safeNumber(system.properties.energy_vol),
    max_delta_volume: safeNumber(system.properties.max_delta_volume),
    stability_charge: safeNumber(system.properties.stability_charge),
    stability_discharge: safeNumber(system.properties.stability_discharge),
  };

  const normalizeProps = (props?: {
    capacity_grav?: number;
    capacity_vol?: number;
    energy_grav?: number;
    energy_vol?: number;
  }) =>
    props
      ? {
          capacity_grav: safeNumber(props.capacity_grav),
          capacity_vol: safeNumber(props.capacity_vol),
          energy_grav: safeNumber(props.energy_grav),
          energy_vol: safeNumber(props.energy_vol),
        }
      : undefined;

  return {
    id: system.battery_id,
    name: `${labelBasis}`,
    components: {
      cathode,
      anode,
      electrolyte,
      separator,
      additives
    },
    properties: displayProps,
    materialProperties: normalizeProps(materialProps),
    cellProperties: normalizeProps(cellProps),
    score: typeof system.score === "number" ? system.score : 0.5,
    explanation: summary,
    diagnostics: {
      paretoRank: system.paretoRank,
      paretoScore: system.paretoScore,
      objectiveAlignment: system.objectiveAlignmentScore,
      feasibility: system.feasibilityScore,
      uncertaintyPenalty: system.uncertaintyPenalty,
      compatibilitySource: system.compatibilitySource,
    },
  };
}
