import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config({ path: '.env' });

export const config = {
  databaseUrl: process.env.DATABASE_URL,
  port: Number.parseInt(process.env.PORT || '5000', 10),
  host: process.env.HOST || '127.0.0.1',
  nodeEnv: process.env.NODE_ENV || 'development',
  ecesspMlUrl: process.env.ECESSP_ML_URL || 'http://127.0.0.1:8000',
  ecesspMlApiKey: process.env.ECESSP_API_KEY || '',
  ecesspMlBearerToken: process.env.ECESSP_BEARER_TOKEN || '',
  proxyTimeoutMs: Number.parseInt(process.env.PROXY_TIMEOUT_MS || '30000', 10),
  proxyRetries: Number.parseInt(process.env.PROXY_RETRIES || '2', 10),
  authEnabled: process.env.AUTH_ENABLED === '1',
  authUsername: process.env.AUTH_USERNAME || 'admin',
  authPassword: process.env.AUTH_PASSWORD || 'change-me',
  authTokenTtlSec: Number.parseInt(process.env.AUTH_TOKEN_TTL_SEC || '43200', 10),
  forwardClientAuthHeader: process.env.FORWARD_CLIENT_AUTH_HEADER === '1',
  forwardClientApiKey: process.env.FORWARD_CLIENT_API_KEY === '1',
};
