#!/usr/bin/env bash
set -euo pipefail

# SQLite non-destructive recovery helper
# - Backs up original DB (+ WAL/SHM)
# - Attempts `.recover` (preferred), falls back to `.dump`
# - Rebuilds into a new database file
# - Runs integrity checks and lists tables
#
# Usage:
#   scripts/sqlite_recover.sh [INPUT_DB] [OUTPUT_DB]
# Defaults:
#   INPUT_DB = data/mwi.db
#   OUTPUT_DB = data/mwi_repaired.db

IN_DB=${1:-data/mwi.db}
OUT_DB=${2:-data/mwi_repaired.db}

TS=$(date +%Y%m%d_%H%M%S)
WORK_DIR="$(dirname "$OUT_DB")/sqlite_repair_${TS}"
DUMP_DIR="$WORK_DIR/dump"
LOG_DIR="$WORK_DIR/logs"
BACKUP_DIR="$WORK_DIR/backup"

echo "[i] sqlite3 version: $(sqlite3 -version || echo 'sqlite3 not found')"

echo "[i] Creating work directories under: $WORK_DIR"
mkdir -p "$DUMP_DIR" "$LOG_DIR" "$BACKUP_DIR"

echo "[i] Backing up original DB files"
cp -p "$IN_DB" "$BACKUP_DIR/" || { echo "[!] Cannot copy $IN_DB"; exit 1; }
[ -f "${IN_DB}-wal" ] && cp -p "${IN_DB}-wal" "$BACKUP_DIR/" || true
[ -f "${IN_DB}-shm" ] && cp -p "${IN_DB}-shm" "$BACKUP_DIR/" || true

echo "[i] Attempting WAL checkpoint (best-effort)"
set +e
sqlite3 "$IN_DB" "PRAGMA wal_checkpoint(FULL);" 1>"$LOG_DIR/wal_checkpoint.out" 2>"$LOG_DIR/wal_checkpoint.err"
set -e

echo "[i] Running integrity_check on source"
set +e
echo "PRAGMA integrity_check;" | sqlite3 "$IN_DB" 1>"$LOG_DIR/integrity_source.out" 2>"$LOG_DIR/integrity_source.err"
set -e

echo "[i] Trying .recover (preferred)"
set +e
sqlite3 "$IN_DB" ".recover" 1>"$DUMP_DIR/recover.sql" 2>"$LOG_DIR/recover.err"
set -e

REC_SIZE=$(wc -c < "$DUMP_DIR/recover.sql" 2>/dev/null || echo 0)
USE_FILE=""
if [ "$REC_SIZE" -gt 0 ]; then
  echo "[i] .recover produced $REC_SIZE bytes"
  USE_FILE="$DUMP_DIR/recover.sql"
else
  echo "[!] .recover failed or empty; trying .dump as fallback"
  set +e
  sqlite3 "$IN_DB" ".dump" 1>"$DUMP_DIR/dump.sql" 2>"$LOG_DIR/dump.err"
  set -e
  DMP_SIZE=$(wc -c < "$DUMP_DIR/dump.sql" 2>/dev/null || echo 0)
  if [ "$DMP_SIZE" -gt 0 ]; then
    echo "[i] .dump produced $DMP_SIZE bytes"
    USE_FILE="$DUMP_DIR/dump.sql"
  else
    echo "[!] Neither .recover nor .dump produced output. See $LOG_DIR for errors. Aborting."
    exit 2
  fi
fi

echo "[i] Building new database at: $OUT_DB"
rm -f "$OUT_DB"
sqlite3 "$OUT_DB" < "$USE_FILE"

echo "[i] Running integrity_check on rebuilt DB"
echo "PRAGMA integrity_check;" | sqlite3 "$OUT_DB" | tee "$LOG_DIR/integrity_rebuilt.out"

echo "[i] Listing tables in rebuilt DB"
echo ".tables" | sqlite3 "$OUT_DB" | tee "$LOG_DIR/tables_rebuilt.out"

echo "[✓] Recovery attempt complete"
echo "    - Original backed up under: $BACKUP_DIR"
echo "    - Dumps/logs under: $WORK_DIR"
echo "    - Repaired DB: $OUT_DB"
echo "
Next steps:
  1) Inspect $LOG_DIR/integrity_rebuilt.out for 'ok'.
  2) Confirm tables exist (Land, Domain, Expression, ExpressionLink, Word, LandDictionary, Media, Tag, TaggedContent).
  3) Test with the app by pointing MWI_DATA_LOCATION to a folder where this repaired file is named 'mwi.db'.
"

