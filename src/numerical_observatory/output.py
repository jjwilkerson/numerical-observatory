"""Stage complete renders and publish files only after successful computation."""
from pathlib import Path
import os
from tempfile import TemporaryDirectory


def staged_render(output, operation):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".observatory-", dir=output.parent) as directory:
        stage = Path(directory)
        result = operation(stage)
        files = sorted(stage.iterdir(), key=lambda path: path.name == "diagnostics.json")
        output.mkdir(parents=True, exist_ok=True)
        backups, installed = [], []
        try:
            for path in files:
                destination = output/path.name
                if destination.exists():
                    backup = stage/(path.name+".previous")
                    os.replace(destination, backup)
                    backups.append((backup, destination))
                os.replace(path, destination)
                installed.append(destination)
        except BaseException:
            for destination in installed:
                destination.unlink(missing_ok=True)
            for backup, destination in backups:
                os.replace(backup, destination)
            raise
    return result
