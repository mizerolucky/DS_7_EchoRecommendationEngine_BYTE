# Echo — Recommendation Engine

<img width="947" height="460" alt="image" src="https://github.com/user-attachments/assets/e6ab249e-12bd-4486-90af-7c9a19929c93" />


An interactive movie recommender built for **AVIP 2026 Data Science Task 7**. It uses collaborative filtering through truncated sparse SVD on user–movie ratings, returns five unseen titles for any MovieLens viewer ID, and displays measured performance on held-out ratings. [Explore the source dataset](https://grouplens.org/datasets/movielens/latest/).

## Results

| Held-out test metric | Bias-only baseline | SVD recommender |
| --- | ---: | ---: |
| RMSE (lower is better) | 0.8777 | **0.8687** |
| MAE (lower is better) | 0.6753 | **0.6660** |

The source is **MovieLens latest-small** from GroupLens Research, downloaded September 23, 2026. It contains 100,836 ratings by 610 users and 9,742 movies. The dataset is fetched by the training script and not committed to this repository. The evaluation snapshot and recommendations under `public/` are generated from it. Full run metadata, split sizes, and validation search are in [`public/metrics.json`](public/metrics.json).

## Reproduce the model

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
python src/example.py
```

Or run [`notebooks/echo_recommender.ipynb`](notebooks/echo_recommender.ipynb) from the repository root. The script downloads `ratings.csv` and `movies.csv` on first run. A fixed seed splits each user's ratings into roughly 70% train, 15% validation, and 15% test. A regularized global/user/item bias is the baseline. Sparse SVD factorizes residuals on observed ratings; validation RMSE selects rank and shrinkage. After tuning, the chosen model is refit on train + validation and compared to the bias-only baseline on the untouched test set. Finally, the five highest-scoring unseen titles per user are exported to the demo as a static snapshot. The [`src/example.py`](src/example.py) script prints top-five predictions for users 1, 42, and 100.

The interface serves precomputed recommendations for all 610 known users; switching user IDs does not retrain the model. Scores are predicted ratings on the dataset's 0.5–5 scale, not probabilities. New users need rating history for personalization. RMSE and MAE assess rating prediction; they do not measure ranking quality or viewer satisfaction.

## Run the demo

```bash
npm install
npm run dev
```

Open the local URL printed by Vite. Deploy the repository as a **Vite** project on Vercel using `npm run build` and output directory `dist`. The app needs no database, secrets, or always-on Python server because the reproducible Python pipeline exports the demo's JSON data. Rerun `python src/train.py` and commit the changed `public/` files if you change the model.

<img width="948" height="475" alt="image" src="https://github.com/user-attachments/assets/8e68510e-b60a-49b1-86cc-cb12bb951690" />


## Deliverables

- `notebooks/echo_recommender.ipynb` — reproducible training, validation, evaluation, and three-user example.
- `src/train.py` — dataset acquisition, split, matrix factorization, tuning, test evaluation, and demo export.
- `src/example.py` — top-five predictions for three user IDs.
- `public/metrics.json` and `public/recommendations.json` — measured results and predictions used by the React demo.
- `src/main.jsx` and `src/style.css` — interactive Vercel-ready demo.

MovieLens data are provided by [GroupLens Research](https://grouplens.org/datasets/movielens/latest/). This is an educational demonstration; movie IDs are anonymous and no personal user accounts are involved.
