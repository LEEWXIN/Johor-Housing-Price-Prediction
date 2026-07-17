# Johor Property Price Advisor — BS23124 Data Mining Mini Project

## Run locally
```
pip install -r requirements.txt
streamlit run app/app.py
```

## Rebuild the pipeline (optional)
```
cd pipeline
python build_database.py   # cleans johor_final.csv -> johor_final_clean.csv, johor_property.db
python train_model.py      # trains models -> model_scores.json, house_model.pkl
```
Copy the four generated files (`johor_final_clean.csv`, `johor_property.db`, `model_scores.json`, `house_model.pkl`) into `app/` if you re-run this.

## Deploy to Streamlit Community Cloud (free, public link)
1. Push this whole folder to your GitHub repo (must include `requirements.txt` at the root and everything inside `app/`):
   ```
   git init
   git add .
   git commit -m "Johor Property DSS"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app** → pick this repo/branch → set **Main file path** to `app/app.py` → **Deploy**.
4. Wait ~1-2 minutes for the build. You'll get a public `*.streamlit.app` link — that's what to send your lecturer.

If the build fails on Streamlit Cloud, check the log for a missing package and add it to `requirements.txt`.
