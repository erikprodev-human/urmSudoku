"""Generate Sudoku Dataset"""
import os, csv, json, argparse, urllib.request
from dataclasses import dataclass, asdict
from typing import Optional, List
import numpy as np
from tqdm import tqdm

@dataclass
class PuzzleDatasetMetadata:
    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    sets: List[str]

@dataclass
class DataProcessConfig:
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-extreme-1k-aug-1000"
    subsample_size: Optional[int] = 1000
    min_difficulty: Optional[int] = None
    num_aug: int = 1000

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    with urllib.request.urlopen(url) as response, open(target_path, 'wb') as out_file:
        total_size = int(response.getheader('Content-Length').strip())
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(target_path)) as pbar:
            while (buffer := response.read(1024 * 1024)):
                out_file.write(buffer)
                pbar.update(len(buffer))

def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    transpose_flag = np.random.rand() < 0.5
    bands, stacks = np.random.permutation(3), np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        if transpose_flag:
            x = x.T
        return digit_map[x.flatten()[mapping].reshape(9, 9).copy()]
    return apply_transformation(board), apply_transformation(solution)

def convert_subset(set_name: str, config: DataProcessConfig):
    csv_path = os.path.join(config.output_dir, f"{set_name}.csv")
    os.makedirs(config.output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        url = f"https://huggingface.co/datasets/{config.source_repo}/resolve/main/{set_name}.csv?download=true"
        try:
            download_file(url, csv_path)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return

    inputs, labels = [], []
    print(f"Processing {set_name}...")
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for source, q, a, rating in reader:
            if config.min_difficulty is None or int(rating) >= config.min_difficulty:
                assert len(q) == 81 and len(a) == 81
                inputs.append(np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
                labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))

    if set_name == "train" and config.subsample_size and config.subsample_size < len(inputs):
        print(f"Subsampling {config.subsample_size} from {len(inputs)}...")
        indices = np.random.choice(len(inputs), size=config.subsample_size, replace=False)
        inputs, labels = [inputs[i] for i in indices], [labels[i] for i in indices]

    num_augments = config.num_aug if set_name == "train" else 0
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id, example_id = 0, 0
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for orig_inp, orig_out in zip(tqdm(inputs), labels):
        for aug_idx in range(1 + num_augments):
            inp, out = (orig_inp, orig_out) if aug_idx == 0 else shuffle_sudoku(orig_inp, orig_out)
            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
        results["group_indices"].append(puzzle_id)

    def _seq_to_numpy(seq):
        return np.concatenate(seq).reshape(len(seq), -1) + 1

    if results["inputs"]:
        results = {"inputs": _seq_to_numpy(results["inputs"]), "labels": _seq_to_numpy(results["labels"]),
                   "group_indices": np.array(results["group_indices"], dtype=np.int32),
                   "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
                   "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32)}
    else:
        results = {k: np.array([], dtype=np.int32) for k in results}

    metadata = PuzzleDatasetMetadata(seq_len=81, vocab_size=11, pad_id=0, ignore_label_id=0, blank_identifier_id=0,
                                     num_puzzle_identifiers=1, total_groups=max(0, len(results["group_indices"]) - 1),
                                     mean_puzzle_examples=1.0, sets=["all"])

    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(asdict(metadata), f)
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    print(f"Finished {set_name}. Output in {save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate Sudoku Dataset")
    parser.add_argument("--output-dir", type=str, default="data/sudoku-extreme-1k-aug-1000")
    parser.add_argument("--subsample-size", type=int, default=1000)
    parser.add_argument("--num-aug", type=int, default=1000)
    parser.add_argument("--min-difficulty", type=int, default=None)
    args = parser.parse_args()

    config = DataProcessConfig(output_dir=args.output_dir, subsample_size=args.subsample_size,
                               num_aug=args.num_aug, min_difficulty=args.min_difficulty)
    print(f"Generating dataset with config: {config}")
    convert_subset("train", config)
    convert_subset("test", config)

if __name__ == "__main__":
    main()