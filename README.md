# PEC — Project Setup & Run Guide

This README explains how to start the three main components in this workspace:

- Plant Disease Detection (Flask app)
- LSTM (time-series model)
- Development/backend (Node.js API)

Follow each section in a separate terminal tab/window.

**Important**: these instructions assume a POSIX shell on Linux and Python 3 installed on the system. Adjust commands for other OSes.

## Plant Disease Detection (Flask)

1. Open a terminal and navigate to the project folder:

```bash
cd plant_disease_detection
```

2. Create and activate a Python virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```

3. Upgrade installer tools and install dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

4. Start the Flask app:

```bash
python app.py
```

Notes:
- The Flask app serves the prediction endpoint (see `app.py`). By default it runs on port 5001 (as set in `app.py`). Ensure `trained_model.keras` exists in this folder.
- If `pip` fails building packages, upgrade `pip`, `setuptools`, and `wheel` (step 3) and consider using the Python version that matches available wheels.

## LSTM (Time-series model)

1. Open a new terminal/tab and navigate to the `LSTM` folder:

```bash
cd LSTM
```

2. Create and activate a Python virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```

3. Upgrade installer tools and install dependencies (if `requirements.txt` exists here):

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

4. Run or inspect the LSTM project:

```bash
python app.py      # if app.py is a runnable service
# or open LSTM.ipynb in Jupyter / Colab for notebook experiments
```

Notes:
- The `LSTM` folder includes `best_model.keras` and `plant_health_data.csv`. Activate the environment before running training or inference.

## Development/backend (Node.js)

1. Open a third terminal/tab and navigate to the Node backend:

```bash
cd Development/backend
```

2. Install Node dependencies and run locally:

```bash
npm install
npm run local
```

Notes:
- If `npm run local` is not defined in `package.json`, run `npm start` or inspect `package.json` for available scripts.
- The backend serves API endpoints used by the frontend in `Development/frontend`.

## Frontend

- Static frontend files are in `Development/frontend`. You can serve them via the backend `public/` directory or open `index.html` directly for testing.

## Useful tips

- Always activate the matching virtual environment before running Python apps in a folder.
- If you face dependency resolution errors, upgrade `pip`/`setuptools`/`wheel` and prefer prebuilt binary wheels.
- On servers, ensure required ports are open for Flask and Node services.

---

Project structure (overview):

- `Development/` — Node backend and static frontend
- `plant_disease_detection/` — Flask app, trained model, datasets
- `LSTM/` — LSTM experiments and model

If you want, I can also create start scripts or run the install commands and report errors.
