# ECESSP Deployment on Vercel + Supabase

This project can use the same high-level stack pattern as NexaraLabs, but not as a single-hosted Vercel app.

## Recommended Architecture

- `ecessp.online` -> Vercel-hosted frontend
- `api.ecessp.online` -> container-hosted FastAPI ML backend
- `Supabase Postgres` -> app database for frontend data, sessions, and future metadata tables
- `Supabase Auth` -> optional replacement for the current in-memory app auth

## Why a Hybrid Deployment Is Required

The frontend side already fits a modern Vercel/Supabase workflow:

- the frontend server already supports `DATABASE_URL`
- the frontend already uses `drizzle-orm` with PostgreSQL
- the frontend can be adapted to serverless API routes

The ML side does not fit Vercel well:

- the backend is a long-running `FastAPI` service
- it loads model checkpoints and graph artifacts at runtime
- discovery requests can take tens of seconds
- this is a better fit for a container host such as `Railway`, `Render`, `Fly.io`, or an OCI VM

## Best Deployment Split

### 1. Frontend on Vercel

Deploy the web app to Vercel and bind:

- `ecessp.online`
- `www.ecessp.online`

Frontend responsibilities:

- serve the UI
- call `api.ecessp.online`
- store app data in Supabase Postgres
- optionally use Supabase Auth instead of the current custom token login

### 2. Database on Supabase

Use Supabase Postgres for:

- materials metadata tables
- user and session-related tables
- saved discoveries, reports, favorites, or future collaboration data

This aligns naturally with the existing `DATABASE_URL` usage in the frontend server.

### 3. ML Backend on a Container Host

Deploy `ecessp-ml` separately as a Dockerized FastAPI app.

Bind it to:

- `api.ecessp.online`

This service should keep:

- model checkpoints
- graph artifacts
- full runtime initialization
- long-running `/api/discover` and `/api/predict` requests

## What Must Change in This Repo

To make ECESSP work well with Vercel + Supabase, the project should be migrated in this order:

1. Keep the current Python backend separate and deploy it to a container host.
2. Point the frontend to the hosted backend with an environment variable such as:
   - `ECESSP_ML_URL=https://api.ecessp.online`
3. Replace the current Express in-memory auth with Supabase Auth, or disable app auth at the frontend layer and protect only the backend.
4. Move proxy/auth routes from the custom Express server into Vercel-compatible API routes, or refactor the frontend into a framework that Vercel supports more directly for server-side handlers.

## Minimal-Change Option

If you want the fastest production path with the fewest code changes:

- keep the current frontend server on a container host
- use Supabase Postgres only for the database
- keep Vercel out of the request path for now

This is the least invasive option.

## Recommended Option

If you want the same developer experience style as NexaraLabs:

- Vercel for the web frontend
- Supabase for Postgres and Auth
- separate ML backend on `api.ecessp.online`

This is the correct architecture for ECESSP.

## Suggested Domain Layout

- `ecessp.online` -> Vercel frontend
- `www.ecessp.online` -> redirect to `ecessp.online`
- `api.ecessp.online` -> FastAPI backend

## Environment Variables

### Frontend

- `DATABASE_URL=<supabase pooled postgres url>`
- `ECESSP_ML_URL=https://api.ecessp.online`
- `AUTH_ENABLED=0` if migrating to Supabase Auth

### Backend

- `ECESSP_API_KEY=<strong secret>`
- `REQUIRE_API_KEY=1`
- optional CORS allowlist for `https://ecessp.online`

## Recommendation

Use the NexaraLabs stack pattern where it fits:

- `Supabase Postgres`: yes
- `Vercel frontend`: yes
- `FastAPI ML backend on Vercel`: no

For ECESSP, the right production architecture is `Vercel + Supabase + separate ML API`.
