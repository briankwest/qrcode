-- Per-request QR Monster ControlNet version: 'v1' or 'v2'. Both are
-- loaded at warm time and swapped on the multi-controlnet without
-- rebuilding the pipe.
ALTER TABLE jobs ADD COLUMN qr_monster_version TEXT NOT NULL DEFAULT 'v1';
