-- A2: auto-escalation toggle. When 1 (default), all-fail jobs spawn a
-- follow-up with controlnet_scale +0.1 up to a 1.5 cap.
ALTER TABLE jobs ADD COLUMN auto_escalate INTEGER NOT NULL DEFAULT 1;
