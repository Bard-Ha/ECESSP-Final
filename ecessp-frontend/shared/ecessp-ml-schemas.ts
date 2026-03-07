// ECESSP-ML API Schemas
// ============================================
// TypeScript definitions matching the Python backend API
// ============================================

export interface MaterialInput {
  material_id?: string;
  formula?: string;
  elements?: string[];
  cif_text?: string;
}

export interface ObjectiveSchema {
  objectives: Record<string, number>;
}

export interface BatterySystemSchema {
  battery_id?: string;
  average_voltage?: number;
  capacity_grav?: number;
  capacity_vol?: number;
  energy_grav?: number;
  energy_vol?: number;
  max_delta_volume?: number;
  stability_charge?: number;
  stability_discharge?: number;
  elements?: string[];
  working_ion?: string;
}

export interface DiscoveryParams {
  num_candidates?: number;
  diversity_weight?: number;
  novelty_weight?: number;
  extrapolation_strength?: number;
  interpolation_enabled?: boolean;
  extrapolation_enabled?: boolean;
  optimize_steps?: number;
  working_ion_candidates?: string[];
  material_source_mode?: "existing" | "generated" | "hybrid";
  component_source_mode?: "existing" | "generated" | "hybrid";
  separator_options_count?: number;
  additive_options_count?: number;
}

export interface DiscoveryRequest {
  system: BatterySystemSchema;
  objective: ObjectiveSchema;
  application?: string;
  explain: boolean;
  mode: string;
  discovery_params?: DiscoveryParams;
}

export interface SystemGenerationRequest extends DiscoveryRequest {}

export interface CifDiscoveryRequest {
  cif_text: string;
  objective: ObjectiveSchema;
  application?: string;
}

export interface ScoringRequest {
  objectives: Record<string, number>;
  top_k?: number;
}

export interface ExplanationRequest {
  battery_id: string;
  application?: string;
}

export interface ChatRequest {
  message: string;
  context?: Record<string, any>;
  contextSystemId?: number;
}

export interface PredictionRequest {
  components: Record<string, string>;
}

export interface PredictionResponse {
  system_name: string;
  predicted_properties: Record<string, number>;
  confidence_score: number;
  score?: number;
  diagnostics?: {
    valid?: boolean;
    speculative?: boolean;
    uncertainty_penalty?: number;
    compatibility_score?: number;
    role_probabilities?: Record<string, number>;
    guardrail_status?: string;
    raw_objective_score?: number;
  };
}

export interface ConstraintReport {
  valid: boolean;
  violations: string[];
  score_penalty: number;
  speculative: boolean;
}

export interface ConstraintSummary {
  overall_valid: boolean;
  physical: ConstraintReport;
  chemical: ConstraintReport;
  performance: ConstraintReport;
}

export interface SystemProperties {
  average_voltage?: number;
  capacity_grav?: number;
  capacity_vol?: number;
  energy_grav?: number;
  energy_vol?: number;
  max_delta_volume?: number;
  stability_charge?: number;
  stability_discharge?: number;
  material_level?: {
    capacity_grav?: number;
    capacity_vol?: number;
    energy_grav?: number;
    energy_vol?: number;
  };
  cell_level?: {
    capacity_grav?: number;
    capacity_vol?: number;
    energy_grav?: number;
    energy_vol?: number;
  };
}

export interface BatterySystemResponse {
  battery_id: string;
  working_ion?: string;
  elements: string[];
  properties: SystemProperties;
  constraints?: ConstraintSummary;
  score?: number;
  speculative: boolean;
}

export interface DiscoveryResponse {
  system: Record<string, any>;
  score: Record<string, any>;
  explanation?: Record<string, any>;
  history: Record<string, any>[];
  metadata: Record<string, any>;
}

export interface RankedSystemsResponse {
  systems: BatterySystemResponse[];
  objectives: Record<string, number>;
}

export interface ExplanationResponse {
  battery_id: string;
  application?: string;
  valid: boolean;
  speculative: boolean;
  summary: string;
  strengths: string[];
  weaknesses: string[];
  tradeoffs: string[];
  constraint_notes: string[];
  material_roles: Record<string, any>;
}

export interface ChatResponse {
  response: string;
  references?: string[];
  relatedSystems?: any[];
}

// Battery property types for type safety
export type BatteryProperty = 
  | "average_voltage"
  | "capacity_grav" 
  | "capacity_vol"
  | "energy_grav"
  | "energy_vol"
  | "max_delta_volume"
  | "stability_charge"
  | "stability_discharge";

// Component types for prediction
export type ComponentType = 
  | "cathode"
  | "anode" 
  | "electrolyte"
  | "separator"
  | "additive";

export interface ComponentSelection {
  [key: string]: string;
}

// Target configuration for discovery
export interface TargetConfiguration {
  average_voltage: number;
  capacity_grav: number;
  capacity_vol: number;
  energy_grav: number;
  energy_vol: number;
  max_delta_volume?: number;
  stability_charge: number;
  stability_discharge: number;
}
