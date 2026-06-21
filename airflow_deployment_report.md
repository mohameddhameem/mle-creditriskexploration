# Airflow Deployment Report & Open Issues

This report documents the efforts, findings, and remaining issues regarding the deployment of Apache Airflow alongside the FastAPI serving endpoint on DigitalOcean App Platform.

---

## 1. What Has Been Tried

To maintain the budget of **$22/month**, several configurations were attempted to run the FastAPI endpoint (`creditrisk-api`), Airflow Webserver, Airflow Scheduler, and a managed PostgreSQL database:

1. **Unified Multi-Component Deployment (512MB RAM)**
   - Configured separate components in the App Spec (`do-app-spec.yaml`) running off a single unified `Dockerfile`.
   - **Result**: Both the Airflow Webserver and Scheduler suffered Out-Of-Memory (OOM) kills (exiting with code `128`) during Gunicorn worker initialization because 512MB RAM was insufficient.
2. **Timing Adjustments & Probes**
   - Configured custom HTTP health checks with extended timing:
     - `creditrisk-api`: `initial_delay_seconds: 45` to allow model pickles to load.
     - `airflow-webserver`: `initial_delay_seconds: 120` to allow Gunicorn web server workers to start.
3. **Database Pre-Deployment Migrations**
   - Added a `PRE_DEPLOY` job (`airflow-init`) running `airflow db migrate` and `airflow users create` to automate database schema preparation and provision a default admin user (`admin` / `admin`).
4. **Unified Single-Container Deployment (1GB RAM)**
   - To avoid paying for two separate containers (which would cost $32/mo), we combined `airflow webserver` and `airflow scheduler` into a single container named `airflow` running on a **1GB RAM (`basic-xs`)** instance.
   - **Result**: The container still crashed with exit code `128` (OOM kill) shortly after booting.

---

## 2. Root Cause of Remaining Issues

The unified container runs out of memory and is terminated by the OS due to:

- **Parallel Process Footprint**: Running the scheduler and the webserver inside the same container launches:
  1. `airflow scheduler` parent loop process
  2. `airflow scheduler` DAG file processor child process
  3. `airflow scheduler` log server Gunicorn master
  4. `airflow scheduler` log server Gunicorn workers (2 by default)
  5. `airflow webserver` parent process
  6. `airflow webserver` Gunicorn master
  7. `airflow webserver` Gunicorn worker (1 configured)
- **Startup Memory Spikes**: When the container boots, the scheduler and webserver initialize **at the exact same time**. The concurrent imports of heavy modules (Airflow core, Flask, WTForms, SQLAlchemy) across 6-8 processes cause a memory usage peak that exceeds 1GB, triggering an OOM-kill within 20 seconds.

---

## 3. What Can Be Done Later (Recommendations)

To deploy Airflow successfully in the future under low resources, the following steps are recommended:

1. **Staggered Sequential Startup Script**
   - Create a script (such as [start_airflow.sh](file:///workspaces/CreditRiskProject/scripts/start_airflow.sh)) that starts the webserver first, waits for port 8080 to be open, and then executes the scheduler. This splits the memory peak in half.
2. **Limit Log Server Gunicorn Workers**
   - Set the environment variable `GUNICORN_CMD_ARGS="--workers 1"` in the App Spec. This forces both the webserver and the scheduler log server to use only one Gunicorn worker, saving about `70MB` RAM.
3. **Upgrade to 2GB RAM for Airflow**
   - If 1GB is still too constrained for the parallel execution of the scheduler and webserver, upgrade the unified `airflow` component to a **basic-s (2GB RAM, 1 vCPU - $15.00/mo)** tier. 
   - This increases the total monthly cost to **$27.00/mo** (which is still cheaper than separate 1GB instances at $32.00/mo).
4. **Use SequentialExecutor**
   - Switch the executor to `SequentialExecutor` to run tasks sequentially in a single process rather than spawning multiple workers.
