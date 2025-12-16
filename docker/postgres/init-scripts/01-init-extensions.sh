#!/bin/bash
set -e

# Perform all actions on the database defined in environment variables
echo "Loading extensions into database '$POSTGRES_DB'..."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- 1. Create Extensions
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS age;

    -- 2. Load AGE for the current session (sanity check)
    LOAD 'age';

    -- 3. Persist the search path
    -- This ensures 'ag_catalog' is always available for every new connection
    -- without the Python client needing to manually run SET search_path.
    ALTER DATABASE "$POSTGRES_DB" SET search_path = ag_catalog, "\$user", public;
EOSQL

echo "Extensions initialized and search_path configured successfully."