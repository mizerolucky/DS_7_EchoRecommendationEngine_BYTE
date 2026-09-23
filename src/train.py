"""Train and evaluate a regularized baseline plus low-rank SVD on MovieLens."""
from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
RNG_SEED = 42


def load_data():
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    if not (data / 'ratings.csv').exists() or not (data / 'movies.csv').exists():
        archive = data / 'ml-latest-small.zip'
        urllib.request.urlretrieve(SOURCE, archive)
        with zipfile.ZipFile(archive) as z:
            for name in ('ratings.csv', 'movies.csv'):
                (data / name).write_bytes(z.read('ml-latest-small/' + name))
        archive.unlink()
    return pd.read_csv(data / 'ratings.csv'), pd.read_csv(data / 'movies.csv')


def split_ratings(ratings):
    """Hold out 15% test and 15% validation per user; keep training history."""
    rng = np.random.default_rng(RNG_SEED)
    train, valid, test = [], [], []
    for _, group in ratings.groupby('userId', sort=True):
        ids = rng.permutation(group.index.to_numpy())
        n = len(ids)
        n_test = max(1, int(n * .15))
        n_valid = max(1, int(n * .15))
        test.extend(ids[:n_test])
        valid.extend(ids[n_test:n_test + n_valid])
        train.extend(ids[n_test + n_valid:])
    return (ratings.loc[ids].reset_index(drop=True) for ids in (train, valid, test))


class Recommender:
    def __init__(self, rank=24, shrink=0.8, reg=12):
        self.rank, self.shrink, self.reg = rank, shrink, reg

    def fit(self, train, users, items):
        self.users = np.asarray(users)
        self.items = np.asarray(items)
        self.user_index = {int(v): i for i, v in enumerate(self.users)}
        self.item_index = {int(v): i for i, v in enumerate(self.items)}
        u = train.userId.map(self.user_index).to_numpy()
        i = train.movieId.map(self.item_index).to_numpy()
        y = train.rating.to_numpy(dtype=float)
        self.mean = y.mean()
        self.ub = np.zeros(len(users))
        self.ib = np.zeros(len(items))
        uc = np.bincount(u, minlength=len(users))
        ic = np.bincount(i, minlength=len(items))
        for _ in range(12):
            self.ub = np.bincount(u, weights=y - self.mean - self.ib[i], minlength=len(users)) / (uc + self.reg)
            self.ib = np.bincount(i, weights=y - self.mean - self.ub[u], minlength=len(items)) / (ic + self.reg)
        residual = y - self.mean - self.ub[u] - self.ib[i]
        matrix = coo_matrix((residual, (u, i)), shape=(len(users), len(items))).tocsr()
        left, singular, right = svds(matrix, k=self.rank, random_state=RNG_SEED)
        order = np.argsort(singular)[::-1]
        scale = np.sqrt(singular[order] * self.shrink)
        self.uf = left[:, order] * scale
        self.mf = right[order, :].T * scale
        self.seen = {int(uid): set(group.movieId) for uid, group in train.groupby('userId')}
        return self

    def predict(self, user_ids, movie_ids, baseline=False):
        u = np.fromiter((self.user_index.get(int(x), -1) for x in user_ids), dtype=int)
        i = np.fromiter((self.item_index.get(int(x), -1) for x in movie_ids), dtype=int)
        scores = np.full(len(u), self.mean)
        known_u, known_i = u >= 0, i >= 0
        scores[known_u] += self.ub[u[known_u]]
        scores[known_i] += self.ib[i[known_i]]
        both = known_u & known_i
        if not baseline:
            scores[both] += (self.uf[u[both]] * self.mf[i[both]]).sum(axis=1)
        return np.clip(scores, .5, 5)

    def recommend(self, user_id, movies, count=5):
        if int(user_id) not in self.user_index:
            raise ValueError('Unknown user ID')
        eligible = movies[~movies.movieId.isin(self.seen[int(user_id)])].copy()
        eligible['score'] = self.predict(np.full(len(eligible), user_id), eligible.movieId.to_numpy())
        eligible = eligible.sort_values(['score', 'movieId'], ascending=[False, True]).head(count)
        return eligible[['movieId', 'title', 'genres', 'score']]


def evaluate(model, subset, baseline=False):
    prediction = model.predict(subset.userId.to_numpy(), subset.movieId.to_numpy(), baseline)
    actual = subset.rating.to_numpy()
    return {'rmse': round(float(np.sqrt(np.mean((actual - prediction) ** 2))), 4),
            'mae': round(float(np.mean(np.abs(actual - prediction))), 4)}


def run():
    ratings, movies = load_data()
    train, valid, test = split_ratings(ratings)
    users, items = np.sort(ratings.userId.unique()), np.sort(movies.movieId.unique())
    candidates = [(rank, shrink) for rank in (12, 24, 40) for shrink in (.5, .8, 1.0)]
    attempts = []
    for rank, shrink in candidates:
        model = Recommender(rank, shrink).fit(train, users, items)
        attempts.append({'rank': rank, 'shrink': shrink, **evaluate(model, valid)})
    best = min(attempts, key=lambda x: x['rmse'])
    # Refit after selecting hyperparameters; the test split is never used for tuning.
    fit_data = pd.concat([train, valid], ignore_index=True)
    model = Recommender(best['rank'], best['shrink']).fit(fit_data, users, items)
    metrics = {'dataset': 'MovieLens latest-small', 'source': SOURCE,
               'seed': RNG_SEED, 'counts': {'train': len(train), 'validation': len(valid),
                                           'test': len(test), 'users': len(users), 'movies': len(items)},
               'selected': {'rank': best['rank'], 'shrink': best['shrink'], 'bias_regularization': 12},
               'validation_search': attempts, 'test_baseline': evaluate(model, test, True),
               'test_model': evaluate(model, test)}
    output = ROOT / 'public'
    output.mkdir(exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    profiles = {}
    for uid in users:
        recs = model.recommend(int(uid), movies, count=5)
        profiles[str(uid)] = {'ratedCount': len(model.seen[int(uid)]),
                              'recommendations': [{'movieId': int(r.movieId), 'title': r.title,
                                                   'genres': r.genres.split('|'), 'score': round(r.score, 2)}
                                                  for r in recs.itertuples()]}
    (output / 'recommendations.json').write_text(json.dumps(profiles, separators=(',', ':')))
    print(json.dumps({'selected': metrics['selected'], 'baseline': metrics['test_baseline'],
                      'model': metrics['test_model']}, indent=2))
    for uid in (1, 42, 100):
        print(f'\nUser {uid}:\n{model.recommend(uid, movies).to_string(index=False)}')
    return metrics


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
