# Credit Score Deployment

Struktur ini memisahkan training pipeline, inferencing, backend API, dan frontend.

## File

- `credit_score_mlflow_pipeline.py` — pipeline training OOP + MLflow.
- `inference.py` — class `CreditScorePredictor` untuk load model bundle dan prediksi.
- `main.py` — FastAPI backend.
- `app.py` — Streamlit frontend.
- `requirements.txt` — dependency deployment.

## Training

Jalankan dari folder project:

```bash
python credit_score_mlflow_pipeline.py --data_path data_B.csv --output_dir outputs
```

Output penting:

```text
outputs/best_credit_score_model.pkl
```

## Backend FastAPI

```bash
uvicorn main:app --reload
```

Buka dokumentasi endpoint:

```text
http://127.0.0.1:8000/docs
```

## Frontend Streamlit

Di terminal lain:

```bash
streamlit run app.py
```

Default endpoint frontend:

```text
http://127.0.0.1:8000/predict
```

## Screenshot testing

Gunakan tiga test case di sidebar Streamlit:

1. Poor-like customer
2. Standard-like customer
3. Good-like customer

Ambil screenshot hasil prediksi untuk masing-masing test case dan screenshot halaman FastAPI `/docs`.
