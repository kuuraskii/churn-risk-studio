# Churn Risk Studio

A Streamlit web app for our Statistical Machine Learning project, **Customer Churn Prediction in Subscription-Based Streaming Services**, built on real KKBOX subscriber data from the WSDM Cup 2018. KKBOX is a music-streaming service in Taiwan. This is not Netflix data.

The app answers three questions: who is likely to leave, why, and whether a retention offer is worth sending.

| Page | What it shows |
|---|---|
| Overview | The problem, the data and headline results |
| Member risk checker | Churn probability, risk band and the main reasons (SHAP) for a member you describe |
| Retention simulator | Net benefit of a retention campaign for any threshold, offer cost, member value and success rate |
| Model performance | Random split vs out-of-time results, cross-validation, calibration, ROC and precision-recall curves |
| What drives churn | Overall SHAP importance, churn rate by segment, February vs March drift |
| About and limits | Data source, assumptions and limitations |

Team: Atul, Aditi and Riya (B.Tech, Bennett University).

## Folder layout

```
app.py              the app
requirements.txt    Python packages
packages.txt        system package needed by LightGBM on Streamlit Cloud
.streamlit/         colours and theme
artifacts/          model and summary files from the Milestone 3 notebook
```

The `artifacts/` folder holds only the trained model, column names, medians and summary tables. **No raw KKBOX data is stored here**, because the competition rules do not allow sharing the data.

## Step 1: get the artifacts

1. Run `KKBOX_Churn_Milestone3.ipynb` on Kaggle with the competition data attached.
2. From the output panel download `app_artifacts.zip`.
3. Unzip it into the `artifacts/` folder, so that `artifacts/model_lgbm.txt` exists.

## Step 2: run locally (the backup)

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (Mac or Linux)
pip install -r requirements.txt
streamlit run app.py
```

The app opens in the browser at `http://localhost:8501`. Stop it with Ctrl+C in the terminal.

## Step 3: deploy on Streamlit Community Cloud

1. Push this folder, including `artifacts/`, to the GitHub repository.
2. Go to share.streamlit.io and sign in with the GitHub account that owns the repository.
3. Click **Create app**, choose the repository, the branch `main` and the file `app.py`.
4. Click **Deploy**. The first start takes a few minutes while packages install.
5. Open the app link once on the morning of the viva, because free apps fall asleep when nobody uses them.

## Updating the app

Change the files, then:

```
git add .
git commit -m "Describe the change"
git push
```

Streamlit Cloud redeploys automatically after a push.

## License and data

The code is released under the MIT License (see `LICENSE`). The KKBOX data is **not** covered by this licence and is not included. It belongs to KKBOX and the WSDM Cup 2018 competition, and must be downloaded from Kaggle under the competition rules.

This project was built for a university course. The simulator figures (offer cost, member value, success rate) are assumptions, not measured business values.
