// client/api.ts
import axios from "axios";
import type {
  CandidateSystem,
  DiscoveryRequest,
  DiscoveryResponse,
  PredictionRequest,
  PredictionResponse,
  BatteryProperty,
} from "@shared/schema";

// ----------------------------
// Axios instance
// ----------------------------
const apiClient = axios.create({
  baseURL: "/api", // assumes same Express server serves frontend
  timeout: 10000,
});

// ----------------------------
// Target Battery Properties
// ----------------------------
export const BATTERY_PROPERTIES: BatteryProperty[] = [
  "average_voltage",
  "capacity_grav",
  "capacity_vol",
  "energy_grav",
  "energy_vol",
  "max_delta_volume",
  "stability_charge",
  "stability_discharge",
];

// ----------------------------
// Discover Systems
// ----------------------------
export async function discoverSystems(
  targets: Record<BatteryProperty, number>,
  weights?: Record<BatteryProperty, number>
): Promise<CandidateSystem[]> {
  try {
    const payload: DiscoveryRequest = { targets };
    if (weights) payload.weights = weights;

    const response = await apiClient.post<DiscoveryResponse>("/discover", payload);
    return response.data.systems;
  } catch (err) {
    console.error("Discover API failed:", err);
    return [];
  }
}

// ----------------------------
// Predict Properties
// ----------------------------
export async function predictSystem(components: {
  cathode: string | number;
  anode: string | number;
  electrolyte: string | number;
}): Promise<PredictionResponse | null> {
  try {
    const payload: PredictionRequest = { components };
    const response = await apiClient.post<PredictionResponse>("/predict", payload);
    return response.data;
  } catch (err) {
    console.error("Predict API failed:", err);
    return null;
  }
}

// ----------------------------
// Get list of available materials
// ----------------------------
export async function listMaterials(): Promise<any[]> {
  try {
    const response = await apiClient.get("/materials");
    return response.data;
  } catch (err) {
    console.error("List materials API failed:", err);
    return [];
  }
}

// ----------------------------
// Health check
// ----------------------------
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await apiClient.get("/health");
    return response.data.status === "ok";
  } catch {
    return false;
  }
}
