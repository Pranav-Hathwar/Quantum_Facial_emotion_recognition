# Database migrations (Alembic)

The schema is defined by the ORM models in `backend/app/models/tables.py`. Two ways to create it:

* **Dev / tests** — `Database.create_all()` (called on app startup in `backend/app/main.py`) issues
  `CREATE TABLE` for every model. Fast, no migration history. This is what the 61 tests use.
* **Managed migrations (Alembic)** — versioned schema changes, the right choice for a real
  PostgreSQL deployment where you cannot drop and recreate tables.

Alembic is configured in `alembic.ini` + `backend/alembic/`. `env.py` pulls the target metadata from
the app's `Base` and the database URL from `DATABASE_URL` (via `get_app_settings()`), so one config
works for both SQLite and PostgreSQL.

## Commands

Run from the project root with the venv active (the URL comes from `DATABASE_URL`; override per-command
with `-x db_url=...`):

```bash
alembic upgrade head                      # apply all migrations (create/upgrade the schema)
alembic current                           # show the DB's current revision
alembic history                           # list revisions
alembic downgrade -1                      # roll back one revision
alembic revision --autogenerate -m "msg"  # create a migration from model changes
```

Examples:

```bash
# apply to a specific Postgres database
alembic -x db_url=postgresql+psycopg://user:pass@host:5432/quantumvision upgrade head
# apply to a throwaway sqlite file
alembic -x db_url=sqlite:///./quantumvision.db upgrade head
```

## Current state

* `20242b6c676d_initial_schema` — creates `users`, `cameras`, `emotion_predictions`, `alerts`,
  `model_runs` and their indexes. Verified: `upgrade head` on an empty database produces exactly the
  `create_all` schema, and a follow-up `--autogenerate` detects **no changes** (no drift).

## Workflow for a schema change

1. Edit the models in `backend/app/models/tables.py`.
2. `alembic revision --autogenerate -m "describe the change"`.
3. Review the generated file in `backend/alembic/versions/` (autogenerate is a draft — check it,
   especially data migrations and SQLite `batch` operations).
4. `alembic upgrade head`.
