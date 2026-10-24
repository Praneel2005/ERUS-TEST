import yaml

MAX_SIDE = 1024      # leaderboard scoring size: longest side at most 1024 px


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def scoring_size(w, h, max_side=MAX_SIDE):
    s = max_side / max(w, h)
    return (w, h) if s >= 1 else (max(1, round(w * s)), max(1, round(h * s)))
