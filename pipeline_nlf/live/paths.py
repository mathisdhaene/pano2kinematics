from pathlib import Path


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def derive_side_outputs(main_out: Path):
    stem = main_out.with_suffix("")
    out_tracked = stem.with_name(stem.name + "_tracked").with_suffix(".mp4")
    out_pseudo = stem.with_name(stem.name + "_pseudo").with_suffix(".mp4")
    out_trc = stem.with_suffix(".trc")
    return out_tracked, out_pseudo, out_trc
