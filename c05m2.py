import argparse
import csv
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import FunctionType, SimpleNamespace

import lightgbm
import numpy as np
from PIL import Image

import c04
from features import extract_row

CONFIG = {
    "j": (0.15873688188386165, 180, "81d1dc42364593a4c0db9260b050544c2341d01b8e7882b91fec8d85676d089e"),
    "final": (0.16461504711798983, 176, "2afb51af8fc5571bcab3ebc389291a16aa3d4ab73c4edd79c365cf8152d3a454"),
    "f": (0.09902811548177957, 180, "18e0fa2cdc45b511b7a36361c933ee67e8a8bbe4e9bcdcc8833bab3b727d7388"),
    "e5": (0.1937088038619091, 170, "90b864474fabb9a0c8ae1412ee1279d234668ca1be766d336e9fb500a69cde56"),
}


def augmented_row(job):
    index, path = job
    # Give the original extractors a fresh copy of each in-memory image.
    def features(image):
        proxy = SimpleNamespace(open=lambda _: image.copy(), Resampling=Image.Resampling)
        bound = {
            name: FunctionType(func.__code__, dict(func.__globals__, Image=proxy))
            for name in ("image_features", "blocks", "targeted_features")
            for func in (extract_row.__globals__[name],)
        }
        return FunctionType(extract_row.__code__, dict(extract_row.__globals__, **bound))(path)

    with Image.open(path) as image:
        values = np.arange(256) / 255
        gamma, gain, offset = (.94, 1.04, 2) if index % 2 else (1.06, .96, -2)
        lut = np.round(np.clip((values**gamma - .5) * gain * 255 + 127.5 + offset,
                               0, 255)).astype(int).tolist()
        operation = (Image.Transpose.ROTATE_90, Image.Transpose.FLIP_LEFT_RIGHT,
                     Image.Transpose.FLIP_TOP_BOTTOM)[index % 3]
        return features(image.convert("RGB").point(lut * 3)), features(image.transpose(operation))


def augment(paths, signature):
    order = np.random.default_rng(42).permutation(len(paths))
    result = np.empty((len(paths), 2, 465), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs = ((int(index), c04.IMAGES / paths[index]) for index in order)
        for count, (index, row) in enumerate(zip(order, pool.map(augmented_row, jobs, chunksize=12)), 1):
            result[index] = row
            if count % 2000 == 0:
                print(f"Augmented {count}/{len(paths)}", flush=True)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite augmented features")
    temporary = c04.WORK / "augmented.tmp.npz"
    np.savez(temporary, signature=signature, matrix=result)
    temporary.replace(c04.WORK / "augmented.npz")


def fit_fold(key, matrix, y, augmented, photos):
    def boost(X, labels, weights=None):
        return c04.estimator().set_params(n_jobs=1).fit(X, labels, sample_weight=weights)

    clean = boost(matrix, y)
    negative = np.flatnonzero(y == 0)
    scores = clean.booster_.predict(matrix[negative], num_threads=1)
    weights = np.ones(len(y))
    weights[negative[np.argsort(scores)[-max(1, len(negative) // 50):]]] = 3.
    if key == "f":
        for label in (0, 1):
            mask = y == label
            _, inverse, sizes = np.unique(photos[mask], return_inverse=True, return_counts=True)
            part = weights[mask] / sizes[inverse] ** .25
            weights[mask] = part * (weights[mask].sum() / part.sum())
        return [(1., boost(matrix, y, weights))]
    if key == "j":
        data = np.concatenate([matrix, augmented[:, 0], augmented[:, 1]])
        labels = np.tile(y, 3)
        view_weights = np.concatenate([.5 * weights, .25 * weights, .25 * weights])
        return [(1., boost(data, labels, view_weights))]
    return [(.8, boost(matrix, y, weights)), (.2, clean)]


def train(key, matrix, y, splits, augmented=None, photos=None):
    fits = [np.arange(len(y))] if key == "f" else [fit for fit, _ in splits]
    for fold, fit in enumerate(fits):
        views = augmented[fit] if augmented is not None else None
        groups = photos[fit] if photos is not None else None
        for member, (_, model) in enumerate(fit_fold(key, matrix[fit], y[fit], views, groups)):
            path = c04.WORK / f"{key}_{fold}_{member}.txt"
            path.write_text(model.booster_.model_to_string(), encoding="utf-8")
        print(f"Trained {key} fold {fold}", flush=True)


def prediction(key, matrix):
    weights = (.8, .2) if key == "final" else (1.,)
    count = 1 if key == "f" else 5
    scores = []
    for fold in range(count):
        for member, weight in enumerate(weights):
            name = f"fold{fold}.txt" if key == "e5" else f"{key}_{fold}_{member}.txt"
            model = lightgbm.Booster(model_str=(c04.WORK / name).read_text(encoding="utf-8"))
            scores.append((weight / count) * model.predict(matrix, num_threads=1))
    result = sum(scores)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite scores")
    return result


def infer(key, matrix, test):
    threshold, _, _ = CONFIG[key]
    path = c04.ROOT / f"submission_c05m2_{'c05' + key if key != 'e5' else key}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(zip(test, (prediction(key, matrix) >= threshold).astype(int)))


def verify(key, test):
    _, positives, expected = CONFIG[key]
    path = c04.ROOT / f"submission_c05m2_{'c05' + key if key != 'e5' else key}.csv"
    rows = c04.read_csv(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if (len(rows) != 8059 or any(len(row) != 2 or row[1] not in {"0", "1"} for row in rows)
            or [row[0] for row in rows] != test or sum(row[1] == "1" for row in rows) != positives
            or digest != expected):
        raise ValueError(f"{key} CSV does not match the original: {digest}")
    print(f"PASS {key}: 8059 rows, {positives} positives, SHA-256 {digest}")


def main():
    parser = argparse.ArgumentParser(description="Reproduce C05M2-J, Final, F, or E5")
    parser.add_argument("variant", choices=CONFIG)
    parser.add_argument("phase", choices=("augment", "train", "infer", "verify"))
    args = parser.parse_args()
    key, phase = args.variant, args.phase
    if phase == "augment" and key != "j":
        parser.error("Only J uses augmentation")
    if phase == "train" and key == "e5":
        parser.error("E5 reuses the five fold models from 'c04.py train'")
    paths, y, photos, signature = c04.inventory()
    splits = c04.fixed_splits(paths, y, photos)
    c04.WORK.mkdir(exist_ok=True)
    if phase == "augment":
        augment(paths[:len(y)], signature)
    elif phase == "train":
        augmented = None
        if key == "j":
            with np.load(c04.WORK / "augmented.npz", allow_pickle=False) as data:
                if str(data["signature"]) != signature:
                    raise ValueError("Input files changed. Run augment again.")
                augmented = data["matrix"]
        train(key, c04.load_features(paths, signature)[:len(y)], y, splits, augmented, photos)
    elif phase == "infer":
        infer(key, c04.load_features(paths, signature)[len(y):], paths[len(y):])
    else:
        verify(key, paths[len(y):])


if __name__ == "__main__":
    main()
