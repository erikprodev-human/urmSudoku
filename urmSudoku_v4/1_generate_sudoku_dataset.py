"""Generate Sudoku Dataset for Universal Grid-to-Grid URM Training.

Downloads Sudoku-Extreme from HuggingFace and converts to universal format:
    train_inputs.npy  (N, 9, 9) uint8 — 0=blank, 1-9=digits
    train_labels.npy  (N, 9, 9) uint8
    test_inputs.npy   (M, 9, 9) uint8
    test_labels.npy   (M, 9, 9) uint8

Usage:
    python 1_generate_sudoku_dataset.py --data_path data/sudoku --subsample 1000
"""
import os, csv, argparse, urllib.request
import numpy as np
from tqdm import tqdm


def download_file(url, path):
    print(f"Downloading {os.path.basename(path)}...")
    with urllib.request.urlopen(url) as r, open(path, 'wb') as f:
        total = int(r.getheader('Content-Length') or 0)
        with tqdm(total=total, unit='B', unit_scale=True) as pb:
            while (buf := r.read(1024 * 1024)):
                f.write(buf); pb.update(len(buf))


def prepare_sudoku(data_path, subsample=1000):
    os.makedirs(data_path, exist_ok=True)
    repo = "sapientinc/sudoku-extreme"
    for split in ["train", "test"]:
        cp = os.path.join(data_path, f"{split}.csv")
        if not os.path.exists(cp):
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{split}.csv?download=true"
            download_file(url, cp)
        inputs, labels = [], []
        with open(cp, newline='') as f:
            rd = csv.reader(f); next(rd)
            for _, q, a, _ in rd:
                inputs.append(np.array([int(c) if c != '.' else 0 for c in q], dtype=np.uint8).reshape(9, 9))
                labels.append(np.array([int(c) for c in a], dtype=np.uint8).reshape(9, 9))
        inputs, labels = np.array(inputs), np.array(labels)
        if split == "train" and subsample and subsample < len(inputs):
            idx = np.random.choice(len(inputs), subsample, replace=False)
            inputs, labels = inputs[idx], labels[idx]
        np.save(os.path.join(data_path, f"{split}_inputs.npy"), inputs)
        np.save(os.path.join(data_path, f"{split}_labels.npy"), labels)
        print(f"  {split}: {len(inputs)} samples, shape {inputs.shape}")
    print(f"Sudoku data ready in {data_path}/")


def main():
    p = argparse.ArgumentParser(description="Generate Sudoku Dataset")
    p.add_argument("--data_path", type=str, default="data/sudoku")
    p.add_argument("--subsample", type=int, default=1000)
    args = p.parse_args()
    prepare_sudoku(args.data_path, args.subsample)


if __name__ == "__main__":
    main()