import { z } from 'zod';
import { 
  discoveryRequestSchema, 
  predictionRequestSchema, 
  materials,
  type CandidateSystem,
  type PredictionResponse 
} from './schema';

// ============================================
// SHARED ERROR SCHEMAS
// ============================================
export const errorSchemas = {
  validation: z.object({
    message: z.string(),
    field: z.string().optional(),
  }),
  notFound: z.object({
    message: z.string(),
  }),
  internal: z.object({
    message: z.string(),
  }),
};

// ============================================
// API CONTRACT
// ============================================
export const api = {
  health: {
    method: 'GET' as const,
    path: '/api/health' as const,
    responses: {
      200: z.object({ status: z.string() }),
    },
  },
  materials: {
    list: {
      method: 'GET' as const,
      path: '/api/materials' as const,
      responses: {
        200: z.array(z.custom<typeof materials.$inferSelect>()),
      },
    },
  },
  discover: {
    method: 'POST' as const,
    path: '/api/discover' as const,
    input: discoveryRequestSchema,
    responses: {
      200: z.object({
        systems: z.array(z.custom<CandidateSystem>())
      }),
      400: errorSchemas.validation,
    },
  },
  predict: {
    method: 'POST' as const,
    path: '/api/predict' as const,
    input: predictionRequestSchema,
    responses: {
      200: z.custom<PredictionResponse>(),
      400: errorSchemas.validation,
      404: errorSchemas.notFound,
    },
  },
};

// ============================================
// HELPER
// ============================================
export function buildUrl(path: string, params?: Record<string, string | number>): string {
  let url = path;
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (url.includes(`:${key}`)) {
        url = url.replace(`:${key}`, String(value));
      }
    });
  }
  return url;
}
