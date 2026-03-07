import { drizzle } from "drizzle-orm/node-postgres";
import pg from "pg";
import * as schema from "@shared/schema";

const { Pool } = pg;

// Optional database - if not provided, use a mock for development
let db: any;
let pool: any;

if (!process.env.DATABASE_URL) {
  console.warn("DATABASE_URL not set. Using mock database for development.");
  
  // Create a mock database object for development
  const mockMaterials = [
    { id: 1, name: "LCO (LiCoO2)", type: "cathode", properties: { average_voltage: 3.9, capacity_grav: 155, capacity_vol: 700, stability_charge: 4.2, stability_discharge: 3.0 } },
    { id: 2, name: "LFP (LiFePO4)", type: "cathode", properties: { average_voltage: 3.2, capacity_grav: 160, capacity_vol: 580, stability_charge: 3.6, stability_discharge: 2.5 } },
    { id: 3, name: "NMC 811", type: "cathode", properties: { average_voltage: 3.8, capacity_grav: 200, capacity_vol: 800, stability_charge: 4.3, stability_discharge: 2.8 } },
    { id: 4, name: "Graphite", type: "anode", properties: { average_voltage: 0.1, capacity_grav: 372, capacity_vol: 700, stability_charge: 0.2, stability_discharge: 0.0 } },
    { id: 5, name: "Silicon", type: "anode", properties: { average_voltage: 0.4, capacity_grav: 3579, capacity_vol: 8000, stability_charge: 0.5, stability_discharge: 0.1 } },
    { id: 6, name: "Lithium Metal", type: "anode", properties: { average_voltage: 0.0, capacity_grav: 3860, capacity_vol: 2000, stability_charge: 0.1, stability_discharge: 0.0 } },
    { id: 7, name: "Standard Liquid", type: "electrolyte", properties: { stability_charge: 4.5, stability_discharge: 0.0 } },
    { id: 8, name: "Solid State (Sulfide)", type: "electrolyte", properties: { stability_charge: 5.0, stability_discharge: 0.0 } },
    { id: 9, name: "Polymer", type: "electrolyte", properties: { stability_charge: 4.8, stability_discharge: 0.0 } },
  ];

  db = {
    select: () => ({
      from: (table: any) => {
        if (table === schema.materials) {
          return Promise.resolve(mockMaterials);
        }
        return Promise.resolve([]);
      }
    }),
    insert: (table: any) => ({
      values: (values: any) => ({
        returning: () => {
          if (table === schema.materials) {
            const newMaterial = { id: mockMaterials.length + 1, ...values };
            mockMaterials.push(newMaterial);
            return Promise.resolve([newMaterial]);
          }
          return Promise.resolve([]);
        }
      })
    })
  };
} else {
  pool = new Pool({ connectionString: process.env.DATABASE_URL });
  db = drizzle(pool, { schema });
}

export { db, pool };
