# Credit Default Prediction API

A production-ready **FastAPI** service for serving a trained **credit default prediction model**, fully containerized with **Docker** and published to **Docker Hub** for easy deployment.

---

## Overview

This API exposes a machine learning model that predicts the **probability of credit default** based on customer financial and repayment features.

The service is designed to be:

* Lightweight and fast (FastAPI)
* Reproducible (Docker)
* Deployable anywhere (Docker Hub image)

---

## Tech Stack

* **FastAPI** – API framework
* **Uvicorn** – ASGI server
* **scikit-learn / XGBoost** – Trained ML model
* **Pandas / NumPy** – Data processing
* **Docker** – Containerization
* **Docker Hub** – Image registry

---

## Project Structure

```text
credit-default-api/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── model.py             # Model loading & prediction logic
│   ├── preprocessing.py     # Feature engineering pipeline
│   ├── schemas.py           # Pydantic request/response schemas
│   └── artifacts/
│       ├── XGB.pkl          # Trained XGBoost model
│       ├── scaler.pkl       # Fitted StandardScaler
│       └── feature_columns.pkl
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## API Endpoints

### Health Check

```http
GET /
```

**Response**

```json
{
  "status": "API is running"
}
```

---

### Predict Credit Default

```http
POST /predict
```

**Request Body (JSON)**
Example:

```json
{
  "LIMIT_BAL": 200000,
  "AGE": 35,
  "PAY_0": 0,
  "PAY_2": -1,
  "PAY_3": 0,
  "PAY_4": 0,
  "PAY_5": 0,
  "PAY_6": 0,
  "PAY_AMT1": 5000,
  "PAY_AMT2": 4000,
  "PAY_AMT3": 3000,
  "PAY_AMT4": 2000,
  "PAY_AMT5": 1000,
  "PAY_AMT6": 500,
  "BILL_AMT1": 10000,
  "BILL_AMT2": 9000,
  "BILL_AMT3": 8000,
  "BILL_AMT4": 7000,
  "BILL_AMT5": 6000,
  "BILL_AMT6": 5000
}
```

**Response**

```json
{
  "default_probability": 0.27
}
```

---

## Running Locally (Without Docker)

### 1. Create & Activate Environment

```bash
conda create -n credit-api python=3.10 -y
conda activate credit-api
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run API

```bash
uvicorn app.main:app --reload
```

### 4. Open Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## Docker Usage

### Build Docker Image

```bash
docker build -t credit-default-api .
```

### Run Container Locally

```bash
docker run -p 8000:8000 credit-default-api
```

Visit:

```
http://localhost:8000/docs
```

---

## Docker Hub Image

The API image is published on Docker Hub:

```
joschj/credit-default-api
```

### Pull Image

```bash
docker pull joschj/credit-default-api:latest
```

### Run from Docker Hub

```bash
docker run -p 8000:8000 joschj/credit-default-api:latest
```

---

## Dockerfile (Summary)

* Python slim base image
* Copies app and artifacts
* Installs dependencies
* Runs FastAPI with Uvicorn

The container is **self-contained** and does not require external services.

---

## Notes on Docker Hub Status

* **Inactive**: No recent vulnerability scans triggered yet
* This is normal for newly pushed images
* Does **not** affect functionality or usability

---

## Production Considerations

For real-world deployment:

* Add authentication (JWT / API keys)
* Add logging & monitoring
* Use a reverse proxy (NGINX)
* Deploy on AWS ECS, Azure Container Apps, or GCP Cloud Run

---

## Author

**Joshua Chukwuma**
Machine Learning / AI Engineer

