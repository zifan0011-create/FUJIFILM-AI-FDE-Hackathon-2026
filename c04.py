import argparse
import csv
import hashlib
import io
from pathlib import Path, PurePosixPath

import lightgbm
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from features import extract_row

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "DATA_Maguire_20180517_ALL" / "SDNET2018"
WORK = ROOT / "work"
THRESHOLD = 0.20427700824353937
SPLIT_SHA = "e730b29021490128f28bc753fa92eb4343606c89f94873a94dd5442629e6efb5"
CSV_SHA = "7ea929eb932ab7466ae4b868b48dd2b352147f51d40406d38f6af7c44b8e14c8"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.reader(handle))


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    temporary.replace(path)


def inventory():
    train = read_csv(ROOT / "train_label.csv")
    test = (ROOT / "test.txt").read_text(encoding="utf-8-sig").splitlines()
    sample = read_csv(ROOT / "sample_submission.csv")
    if any(len(row) != 2 or row[1] not in {"0", "1"} for row in train):
        raise ValueError("Invalid training label rows")
    if any(len(row) != 2 for row in sample) or [row[0] for row in sample] != test:
        raise ValueError("Sample submission path order differs from test.txt")
    paths = [row[0] for row in train] + test
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate path in the train/test inventory")
    signature = hashlib.sha256()
    photo_ids = []
    for path in paths:
        parts = PurePosixPath(path).parts
        if (len(parts) != 3 or parts[0] != "W" or parts[1] not in {"CW", "UW"}
                or not parts[2].endswith(".jpg")):
            raise ValueError(f"Unexpected image path: {path}")
        photo, tile = PurePosixPath(path).stem.rsplit("-", 1)
        if not photo.isdecimal() or not all(part.isdecimal() for part in tile.split("_")):
            raise ValueError(f"Unexpected photo/tile ID: {path}")
        photo_ids.append(photo)
        stat = (IMAGES / path).stat()
        if stat.st_size == 0:
            raise ValueError(f"Empty image file: {path}")
        signature.update(f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    n_train = len(train)
    if set(photo_ids[:n_train]) & set(photo_ids[n_train:]):
        raise ValueError("A source photo appears in both train and test")
    return paths, np.array([int(row[1]) for row in train]), np.array(photo_ids), signature.hexdigest()


def fixed_splits(paths, y, photos):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = photos[:len(y)]
    splits = list(splitter.split(y, y, groups))
    folds = np.full(len(y), -1)
    for fold, (fit, valid) in enumerate(splits):
        if set(groups[fit]) & set(groups[valid]) or np.any(folds[valid] >= 0):
            raise ValueError("Invalid photo split")
        folds[valid] = fold
    manifest = io.StringIO(newline="")
    writer = csv.writer(manifest, lineterminator="\n")
    writer.writerow(("path", "photo_id", "fold", "label"))
    writer.writerows(zip(paths, groups, folds, y))
    if hashlib.sha256(manifest.getvalue().encode("utf-8")).hexdigest() != SPLIT_SHA:
        raise ValueError("The training inputs do not reproduce the original C04 split")
    return splits


def extract(paths, signature):
    matrix = np.empty((len(paths), 465), dtype=np.float32)
    for index, path in enumerate(paths):
        matrix[index] = extract_row(IMAGES / path)
        if (index + 1) % 2000 == 0:
            print(f"Features {index + 1}/{len(paths)}", flush=True)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite features")
    temporary = WORK / "features.tmp.npz"
    np.savez(temporary, paths=np.array(paths), signature=signature, matrix=matrix)
    temporary.replace(WORK / "features.npz")


def load_features(paths, signature):
    with np.load(WORK / "features.npz", allow_pickle=False) as data:
        if data["paths"].tolist() != paths or str(data["signature"]) != signature:
            raise ValueError("Input files changed. Run extract again.")
        return data["matrix"]


def estimator():
    return lightgbm.LGBMClassifier(
        n_estimators=240, learning_rate=.035, num_leaves=15,
        min_child_samples=25, colsample_bytree=.85, scale_pos_weight=1,
        reg_lambda=2, random_state=42, n_jobs=6, verbosity=-1,
    )


def fit_fold(X, y):
    initial = estimator().fit(X, y)
    negatives = np.flatnonzero(y == 0)
    scores = initial.booster_.predict(X[negatives], num_threads=6)
    count = max(1, len(negatives) // 50)
    weights = np.ones(len(y))
    weights[negatives[np.argsort(scores)[-count:]]] = 3.
    return estimator().fit(X, y, sample_weight=weights)


def train(matrix, y, splits):
    for fold, (fit, _) in enumerate(splits):
        model = fit_fold(matrix[fit], y[fit])
        (WORK / f"fold{fold}.txt").write_text(model.booster_.model_to_string(), encoding="utf-8")
        print(f"Trained fold {fold}", flush=True)


def infer(matrix, test):
    scores = [
        lightgbm.Booster(model_str=(WORK / f"fold{fold}.txt").read_text(encoding="utf-8"))
        .predict(matrix, num_threads=6)
        for fold in range(5)
    ]
    mean = np.mean(scores, axis=0)
    if not np.isfinite(mean).all():
        raise ValueError("Non-finite scores")
    write_csv(ROOT / "submission_c04.csv", zip(test, (mean >= THRESHOLD).astype(int)))


def verify(test):
    path = ROOT / "submission_c04.csv"
    raw = path.read_bytes()
    rows = read_csv(path)
    if len(rows) != 8059 or any(len(row) != 2 or row[1] not in {"0", "1"} for row in rows):
        raise ValueError("Invalid C04 CSV format")
    if [row[0] for row in rows] != test or sum(row[1] == "1" for row in rows) != 168:
        raise ValueError("C04 paths or positive count differ")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != CSV_SHA:
        raise ValueError(f"C04 SHA-256 differs: {digest}")
    print(f"PASS: {len(rows)} rows, 168 positives, {len(raw)} bytes, SHA-256 {digest}")


def main():
    parser = argparse.ArgumentParser(description="Reproduce the original C04 submission")
    parser.add_argument("phase", choices=("extract", "train", "infer", "verify"))
    phase = parser.parse_args().phase
    paths, y, photos, signature = inventory()
    splits = fixed_splits(paths, y, photos)
    WORK.mkdir(exist_ok=True)
    if phase == "extract":
        extract(paths, signature)
    elif phase == "train":
        train(load_features(paths, signature), y, splits)
    elif phase == "infer":
        infer(load_features(paths, signature)[len(y):], paths[len(y):])
    else:
        verify(paths[len(y):])


if __name__ == "__main__":
    main()
