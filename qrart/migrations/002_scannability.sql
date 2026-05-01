-- Per-candidate scannability score (0.0-1.0): fraction of QR modules
-- correctly resolved on the rendered image. NULL for pre-existing rows.
ALTER TABLE candidates ADD COLUMN scannability REAL;
