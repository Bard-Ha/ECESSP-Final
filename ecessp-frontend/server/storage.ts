import { db } from "./db";
import { materials, type Material, type InsertMaterial } from "@shared/schema";
import { eq } from "drizzle-orm";

export interface IStorage {
  // Material CRUD
  getMaterials(): Promise<Material[]>;
  getMaterial(id: number): Promise<Material | undefined>;
  createMaterial(material: InsertMaterial): Promise<Material>;
  seedMaterials(): Promise<void>;
}

export class DatabaseStorage implements IStorage {
  async getMaterials(): Promise<Material[]> {
    return await db.select().from(materials);
  }

  async getMaterial(id: number): Promise<Material | undefined> {
    const results = await db.select().from(materials).where(eq(materials.id, id));
    return results[0];
  }
  
  // Re-implementing correctly with drizzle syntax
  async createMaterial(material: InsertMaterial): Promise<Material> {
    const [newMaterial] = await db.insert(materials).values(material).returning();
    return newMaterial;
  }

  async seedMaterials(): Promise<void> {
    const count = await db.select().from(materials);
    if (count.length > 0) return;

    const seedData: InsertMaterial[] = [
      // Cathodes
      { name: "LCO (LiCoO2)", type: "cathode", properties: { average_voltage: 3.9, capacity_grav: 155, capacity_vol: 700, stability_charge: 4.2, stability_discharge: 3.0 } },
      { name: "LFP (LiFePO4)", type: "cathode", properties: { average_voltage: 3.2, capacity_grav: 160, capacity_vol: 580, stability_charge: 3.6, stability_discharge: 2.5 } },
      { name: "NMC 811", type: "cathode", properties: { average_voltage: 3.8, capacity_grav: 200, capacity_vol: 800, stability_charge: 4.3, stability_discharge: 2.8 } },
      
      // Anodes
      { name: "Graphite", type: "anode", properties: { average_voltage: 0.1, capacity_grav: 372, capacity_vol: 700, stability_charge: 0.2, stability_discharge: 0.0 } },
      { name: "Silicon", type: "anode", properties: { average_voltage: 0.4, capacity_grav: 3579, capacity_vol: 8000, stability_charge: 0.5, stability_discharge: 0.1 } },
      { name: "Lithium Metal", type: "anode", properties: { average_voltage: 0.0, capacity_grav: 3860, capacity_vol: 2000, stability_charge: 0.1, stability_discharge: 0.0 } },

      // Electrolytes
      { name: "Standard Liquid", type: "electrolyte", properties: { stability_charge: 4.5, stability_discharge: 0.0 } },
      { name: "Solid State (Sulfide)", type: "electrolyte", properties: { stability_charge: 5.0, stability_discharge: 0.0 } },
      { name: "Polymer", type: "electrolyte", properties: { stability_charge: 4.8, stability_discharge: 0.0 } },
    ];

    await db.insert(materials).values(seedData);
  }
}

export const storage = new DatabaseStorage();
