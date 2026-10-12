# final/rollout_loader.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional
import pickle
import re


@dataclass
class Rollout:
    idx: int
    path: str
    time: List[float]
    q_mes_all: List[List[float]]
    qd_mes_all: List[List[float]]
    q_d_all: List[List[float]]   
    qd_d_all: List[List[float]]   
    tau_mes_all: List[List[float]]
    cart_pos_all: List[List[float]]
    cart_ori_all: List[List[float]]


    @classmethod
    def from_dict(cls, idx: int, path: Path, d: Dict[str, Any]) -> "Rollout":
        required = ["time", "q_mes_all", "qd_mes_all", "q_d_all", "qd_d_all",
                    "tau_mes_all", "cart_pos_all", "cart_ori_all"]
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"{path} is missing keys: {missing}")
        return cls(
            idx=idx,
            path=str(path),
            time=d["time"],
            q_mes_all=d["q_mes_all"],
            qd_mes_all=d["qd_mes_all"],
            q_d_all=d["q_d_all"],
            qd_d_all=d["qd_d_all"],
            tau_mes_all=d["tau_mes_all"],
            cart_pos_all=d["cart_pos_all"],
            cart_ori_all=d["cart_ori_all"],
        )



# --------- helper(s) ----------
def _default_final_dir() -> Path:
    # Resolve to the absolute path of the 'final' package containing this file
    return Path(__file__).resolve().parent


def _find_by_index(directory: Path, i: int) -> Optional[Path]:
    # Accept both "data_{i}.pkl" and "{i}.pkl"
    for name in (f"data_{i}.pkl", f"{i}.pkl"):
        p = directory / name
        if p.is_file():
            return p
    return None


def _discover_all(directory: Path) -> List[tuple[int, Path]]:
    """Return all (index, path) pairs discovered in directory."""
    items: List[tuple[int, Path]] = []
    # Match data_123.pkl or 123.pkl
    pat = re.compile(r"^(?:data_)?(\d+)\.pkl$")
    for p in directory.iterdir():
        if p.suffix == ".pkl":
            m = pat.match(p.name)
            if m:
                items.append((int(m.group(1)), p))
    items.sort(key=lambda t: t[0])
    return items


# --------- public API ----------
def load_rollouts(
    count: Optional[int] = None,
    indices: Optional[Iterable[int]] = None,
    directory: Optional[str | Path] = None,
    strict_missing: bool = False,
) -> List[Rollout]:
    """
    Load multiple rollout .pkl files.

    Parameters
    ----------
    count : int, optional
        Load indices 1..count (useful for 'first N').
    indices : iterable of int, optional
        Load these specific indices (e.g., [1, 3, 7]).
    directory : str | Path, optional
        Folder containing the .pkl files. Defaults to the 'final' package dir.
    strict_missing : bool
        If True, raise if any requested index is not found. If False, skip silently.

    Returns
    -------
    List[Rollout]
    """
    dir_path = Path(directory) if directory is not None else _default_final_dir()

    targets: List[tuple[int, Path]] = []
    if indices is not None:
        for i in sorted(set(indices)):
            p = _find_by_index(dir_path, i)
            if p is None:
                if strict_missing:
                    raise FileNotFoundError(f"Missing rollout index {i} in {dir_path}")
            else:
                targets.append((i, p))
    elif count is not None:
        for i in range(1, count + 1):
            p = _find_by_index(dir_path, i)
            if p is None:
                if strict_missing:
                    raise FileNotFoundError(f"Missing rollout index {i} in {dir_path}")
            else:
                targets.append((i, p))
    else:
        # Load everything we can find
        targets = _discover_all(dir_path)
        if not targets:
            raise FileNotFoundError(f"No rollout .pkl files found in {dir_path}")

    rollouts: List[Rollout] = []
    for i, p in targets:
        with p.open("rb") as f:
            data = pickle.load(f)
        rollouts.append(Rollout.from_dict(i, p, data))

    # Always return in ascending index order
    rollouts.sort(key=lambda r: r.idx)
    return rollouts
