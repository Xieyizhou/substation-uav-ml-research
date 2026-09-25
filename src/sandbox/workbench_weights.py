"""Keep one private, hash-addressed initialization checkpoint per workbench."""

from pathlib import Path
import os
import shutil
import tempfile

from src.ml.artifacts import file_sha256


def pin_weights(project_root, runs_root, source):
    project = Path(project_root).resolve()
    source = Path(source).expanduser().resolve()
    if source.suffix.lower() != ".pt" or not source.is_file():
        raise ValueError("initial weights must be a local .pt checkpoint")
    digest = file_sha256(source)
    target = (Path(runs_root).resolve().parent / "weights" / f"{digest}.pt").resolve()
    relative = target.relative_to(project).as_posix()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha256(target) != digest:
            raise ValueError("pinned initialization checkpoint identity changed")
    else:
        # A hard link to user-controlled input would change when it is edited.
        fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".partial")
        os.close(fd)
        try:
            shutil.copyfile(source, temporary)
            if file_sha256(Path(temporary)) != digest:
                raise ValueError("initialization checkpoint changed during import")
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {"checkpoint_path": relative, "source_path": str(source)}, digest


def resolve_initial_weights(project_root, recipe):
    project = Path(project_root).resolve()
    source = recipe.initialization
    weights = project / "yolo11n.pt"
    if source is not None:
        weights = (project / source["checkpoint_path"]).resolve()
        if not weights.is_relative_to(project):
            raise ValueError("initialization checkpoint must stay inside the project")
    if file_sha256(weights) != recipe.pretrained_weights_sha256:
        raise ValueError("workbench pretrained weights changed")
    return weights
