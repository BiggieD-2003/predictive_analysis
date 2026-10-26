# Smart Predictive Maintenance Dashboard

This project analyzes machine performance using sensor data to predict maintenance requirements and display insights in an interactive Streamlit dashboard.

## Features
- Data Preprocessing and Feature Engineering
- Machine Health Score calculation (Healthy/Warning/Critical)
- Random Forest Classifier for failure prediction
- Streamlit interactive UI dashboard

## Project Structure
- `data/` : Dataset will be stored here.
- `models/` : Saved machine learning models.
- `utils/` : Helper modules for preprocessing and health scores.
- `app.py` : Streamlit dashboard file.
- `train_model.py` : Model training script.
- `generate_data.py` : Script to create a custom mockup dataset.

## How to Run

1. **Install requirements:**
Ensure you have the following packages installed:
```bash
pip install pandas numpy scikit-learn streamlit plotly joblib
```
2. **Generate the mock dataset:**
This will generate `data/machine_data.csv`.
```bash
python generate_data.py
```

3. **Train the ML model:**
This will create and save the model in `models/random_forest_model.pkl`.
```bash
python train_model.py
```

4. **Run the Streamlit Dashboard:**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.
