-- Initial schema for QR Art Studio.
-- Booleans are stored as INTEGER (0/1); timestamps as ISO 8601 TEXT.

CREATE TABLE jobs (
  id                 TEXT PRIMARY KEY,
  created_at         TEXT NOT NULL,
  started_at         TEXT,
  finished_at        TEXT,
  status             TEXT NOT NULL,           -- queued | running | completed | failed | cancelled
  error              TEXT,
  elapsed_s          REAL,

  -- request
  data               TEXT NOT NULL,
  prompt             TEXT NOT NULL,
  negative_prompt    TEXT,
  style              TEXT NOT NULL,
  model              TEXT NOT NULL,
  composition        TEXT NOT NULL,

  -- diffusion settings
  candidates         INTEGER NOT NULL,
  steps              INTEGER NOT NULL,
  guidance           REAL NOT NULL,
  controlnet_scale   REAL NOT NULL,
  tile_scale         REAL NOT NULL,
  control_end        REAL NOT NULL,
  refine             INTEGER NOT NULL,
  refine_strength    REAL NOT NULL,
  refine_steps       INTEGER NOT NULL,
  size               INTEGER NOT NULL,
  seed               INTEGER,
  require_scan       INTEGER NOT NULL,
  fast_mode          INTEGER NOT NULL,

  -- finishing
  hires_fix          INTEGER NOT NULL,
  hires_target       INTEGER NOT NULL,
  hires_strength     REAL NOT NULL,
  adetailer          INTEGER NOT NULL,
  adetailer_strength REAL NOT NULL,

  -- output rollup
  best_candidate_id  TEXT,
  scans              INTEGER,                 -- 0/1, NULL while running
  decoded            TEXT,
  qr_image_path      TEXT,

  -- bookkeeping
  client_ip          TEXT,
  user_agent         TEXT,
  parent_job_id      TEXT REFERENCES jobs(id) ON DELETE SET NULL
);
CREATE INDEX idx_jobs_created  ON jobs(created_at DESC);
CREATE INDEX idx_jobs_status   ON jobs(status);
CREATE INDEX idx_jobs_model    ON jobs(model);
CREATE INDEX idx_jobs_scans    ON jobs(scans);

CREATE TABLE candidates (
  id                 TEXT PRIMARY KEY,
  job_id             TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  idx                INTEGER NOT NULL,
  seed               INTEGER NOT NULL,
  controlnet_scale   REAL NOT NULL,
  refine_strength    REAL,
  scans              INTEGER NOT NULL,
  decoded            TEXT,
  image_path         TEXT NOT NULL,
  pass1_image_path   TEXT
);
CREATE INDEX idx_candidates_job ON candidates(job_id);

-- Streamed progress events (phase 3 will write here; schema lands now).
CREATE TABLE job_events (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id    TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  ts        TEXT NOT NULL,
  type      TEXT NOT NULL,
  payload   TEXT NOT NULL                    -- JSON
);
CREATE INDEX idx_events_job ON job_events(job_id, ts);

-- Recent + favorited prompts (phase 5 reads this; we'll write on every job).
CREATE TABLE prompts (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  text         TEXT NOT NULL UNIQUE,
  used_count   INTEGER NOT NULL DEFAULT 0,
  last_used_at TEXT,
  favorited    INTEGER NOT NULL DEFAULT 0,
  notes        TEXT
);
CREATE INDEX idx_prompts_last_used ON prompts(last_used_at DESC);
CREATE INDEX idx_prompts_favorited ON prompts(favorited);
