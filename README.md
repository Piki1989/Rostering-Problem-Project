# Rostering-Problem-Project

Hospital doctor rostering with a CP-SAT model (OR-Tools) and a Streamlit app.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## Run

```bash
uv run streamlit run app/streamlit_new.py
```

The app opens in the browser. The first load runs the optimization and can take several minutes.

## Options

- **Pandemic mode** is switched on/off with `pandemic_mode=True/False` in `run_model()` in `app/streamlit_new.py`.
- **Input data** (doctors, shifts, unavailability, preferences) is selected in `run_model_and_get_results()` in `model/model_ref.py` by uncommenting the desired files from `data/`.
