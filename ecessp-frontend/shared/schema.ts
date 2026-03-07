import { pgTable, text, serial, integer, boolean, jsonb, real } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// === TABLE DEFINITIONS ===
export const materials = pgTable("materials", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  type: text("type").notNull(), // 'cathode', 'anode', 'electrolyte', 'separator'
  properties: jsonb("properties").$type<Record<string, number>>().notNull(),
});

// === SCHEMAS ===
export const insertMaterialSchema = createInsertSchema(materials).omit({ id: true });
export type Material = typeof materials.$inferSelect;
export type InsertMaterial = z.infer<typeof insertMaterialSchema>;

// === API CONTRACT TYPES ===

// Updated property list as per user request
export const BATTERY_PROPERTIES = [
  "average_voltage",
  "capacity_grav",
  "capacity_vol",
  "energy_grav",
  "energy_vol",
  "max_delta_volume",
  "stability_charge",
  "stability_discharge"
] as const;

export type BatteryProperty = typeof BATTERY_PROPERTIES[number];

// Discovery
export const discoveryRequestSchema = z.object({
  system: z.object({
    battery_id: z.string().optional(),
    average_voltage: z.number().optional(),
    capacity_grav: z.number().optional(),
    capacity_vol: z.number().optional(),
    energy_grav: z.number().optional(),
    energy_vol: z.number().optional(),
    max_delta_volume: z.number().optional(),
    stability_charge: z.number().optional(),
    stability_discharge: z.number().optional(),
    elements: z.array(z.string()).optional(),
    working_ion: z.string().optional(),
  }),
  objective: z.object({
    objectives: z.record(z.enum(BATTERY_PROPERTIES), z.number()),
  }),
  application: z.string().optional(),
  explain: z.boolean().default(true),
  mode: z.string().default("predictive"),
});

export type DiscoveryRequest = z.infer<typeof discoveryRequestSchema>;

export interface CandidateSystem {
  id: string;
  name: string;
  components: {
    cathode: string;
    anode: string;
    electrolyte: string;
  };
  properties: Record<BatteryProperty, number>;
  score: number;
  explanation: string;
}

export type DiscoveryResponse = {
  system: Record<string, any>;
  score: Record<string, any>;
  explanation: Record<string, any> | null;
  history: Record<string, any>[];
  metadata: Record<string, any>;
};

// Prediction
export const predictionRequestSchema = z.object({
  components: z.record(z.string()),
});

export type PredictionRequest = z.infer<typeof predictionRequestSchema>;

export type PredictionResponse = {
  system_name: string;
  predicted_properties: Record<BatteryProperty, number>;
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
};

// Materials List
export type MaterialsListResponse = Material[];
