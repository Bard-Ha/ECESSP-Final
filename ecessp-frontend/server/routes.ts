import type { Express } from "express";
import type { Server } from "http";
import { storage } from "./storage";
import { api } from "@shared/routes";

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  await storage.seedMaterials();

  // Local health is retained for the Express app itself.
  // ML backend health is proxied by backendProxy at /api/health.
  app.get(api.health.path, (_req, res) => {
    res.json({ status: "ok" });
  });

  // Materials are frontend-local (seed/mock/DB) and used by prediction UI selectors.
  app.get(api.materials.list.path, async (_req, res) => {
    const materials = await storage.getMaterials();
    res.json(materials);
  });

  return httpServer;
}

