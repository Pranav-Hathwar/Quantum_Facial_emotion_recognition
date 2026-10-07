# Deployment

* **Local**: see README quick start (uvicorn + `npm run dev`).
* **Docker**: `docker compose up --build` → frontend on :8080 (nginx proxies `/api` and `/ws`), backend, PostgreSQL.
  Place trained checkpoints and `comparison.json` in `./ml/models` (mounted into the backend).
* **Before any real use**: change `JWT_SECRET` and `ADMIN_PASSWORD`, serve over HTTPS (browsers require it for webcam
  outside localhost), keep `STORE_IMAGES=false`, and run with a trained, evaluated model.
* **GPU serving**: the image uses CPU torch; for GPU use a CUDA base image and set `DEVICE=cuda`.
* **Responsible use**: this is an academic prototype. Facial-emotion inference is unreliable and biased in places; do not
  use it to make decisions about individuals. Consider consent and local surveillance/privacy law.
