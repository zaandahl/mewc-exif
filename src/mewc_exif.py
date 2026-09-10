"""Immutable source metadata extraction and a separate Camelot JPEG adapter."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

import pandas as pd
import piexif
from PIL import Image

DEFAULTS = dict(INPUT_DIR="/images", MD_FILE="md_out.json", EN_FILE="mewc_out.pkl",
                EN_CSV="mewc_out.csv", EXIF_DIR="camelot", METADATA_DIR="metadata",
                OVERLAP=0.3, EDGE_DIST=0.02, MIN_EDGES=0, UPPER_CONF=0.9,
                LOWER_CONF=0.05, SUPPRESSION_POLICY="category-confidence-v1")


def relative_file(value):
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("source_file must be a POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in ("..", ".", "") for p in value.split("/")):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path.as_posix()


def beneath(root, relative):
    path = (root / relative_file(relative)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes its root: {relative}")
    return path


def atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".mewc-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        writer(Path(name))
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, lambda temp: temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"))


def bind_predictions(frame, images):
    """Prefer canonical keys; migrate legacy crop names only with a unique exact match."""
    lookup, sources = {}, {}
    for image in images:
        source = relative_file(image["file"])
        if source in sources:
            raise ValueError(f"duplicate detector source: {source}")
        sources[source] = image
        for index, _ in enumerate(image.get("detections") or []):
            path = PurePosixPath(source)
            crop = str(path.with_name(f"{path.stem}-{index}{path.suffix}"))
            for name in {crop, PurePosixPath(crop).name}:
                lookup.setdefault(name, set()).add((source, index))
    result = frame.copy().reset_index(drop=True)
    explicit = {"source_file", "detection_index"}.issubset(result.columns)
    if not explicit and ({"source_file", "detection_index"} & set(result.columns)):
        raise ValueError("prediction identity requires both source_file and detection_index")
    keys = []
    for row_id, row in result.iterrows():
        if explicit:
            source = relative_file(row["source_file"])
            index = row["detection_index"]
            if isinstance(index, bool) or pd.isna(index) or int(index) != index:
                raise ValueError(f"invalid detection_index at prediction row {row_id}")
            key = source, int(index)
        else:
            name = relative_file(row["filename"])
            candidates = lookup.get(name, set())
            if len(candidates) != 1:
                raise ValueError(f"ambiguous or unmatched legacy crop at row {row_id}: {name}")
            key = next(iter(candidates))
        source, index = key
        detections = sources.get(source, {}).get("detections") or []
        if index < 0 or index >= len(detections):
            raise ValueError(f"unknown source/detection identity at row {row_id}: {key}")
        if str(detections[index]["category"]) != "1":
            raise ValueError(f"classifier row refers to non-animal detection: {key}")
        keys.append(key)
    result["source_file"] = [key[0] for key in keys]
    result["detection_index"] = [key[1] for key in keys]
    result["conf"] = [sources[source]["detections"][index]["conf"] for source, index in keys]
    return result


def text_tag(value):
    return value.decode("utf-8").rstrip("\x00") if isinstance(value, bytes) else str(value)


def read_camera_metadata(source):
    with Image.open(source) as image:
        image_format = image.format
        exif = piexif.load(image.info["exif"]) if image.info.get("exif") else {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}
    tags = exif["Exif"]
    if piexif.ExifIFD.DateTimeOriginal in tags:
        timestamp = text_tag(tags[piexif.ExifIFD.DateTimeOriginal])
        offset = tags.get(36881)  # OffsetTimeOriginal; absent in many camera JPEGs.
        provenance, zone = "exif_datetime_original", text_tag(offset) if offset else "unknown"
    else:
        timestamp = datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat()
        provenance, zone = "filesystem_mtime", "UTC"
    flash = tags.get(piexif.ExifIFD.Flash)
    return exif, dict(image_format=image_format, date_time_orig=timestamp, timestamp_source=provenance,
                      timestamp_timezone=zone, flash_fired=None if flash is None else int(bool(int(flash) & 1)))


def camelot_metadata(exif, rows, count):
    """Preserve the established camera-tag mapping, refusing unrepresentable ties."""
    result = copy.deepcopy(exif)
    tags = result.setdefault("Exif", {})
    if isinstance(tags.get(41988), int):
        tags[41988] = (tags[41988], 1)  # Existing Bushnell repair, export only.
    tags[piexif.ExifIFD.FNumber] = (int(count), 1)
    for rank, tag in ((1, piexif.ExifIFD.ISOSpeedRatings), (2, piexif.ExifIFD.FocalLength)):
        group = rows.loc[rows["class_rank"] == rank] if len(rows) else rows
        if group.empty:
            tags.pop(tag, None)
            if rank == 1:
                tags.pop(piexif.ExifIFD.ExposureTime, None)
            continue
        best = group.loc[group["conf"] == group["conf"].max()]
        columns = ["class_id", "prob"] if rank == 1 else ["class_id"]
        distinct = best[columns].drop_duplicates()
        if len(distinct) != 1:
            raise ValueError(f"Camelot rank {rank} cannot represent tied highest-confidence classifications")
        winner = distinct.iloc[0]
        raw_id = winner["class_id"]
        class_id = int(raw_id)
        prob = float(winner["prob"]) if rank == 1 else 0
        canonical_id = (str(class_id) == raw_id) if isinstance(raw_id, str) else (not isinstance(raw_id, bool) and class_id == raw_id)
        if not canonical_id or not 0 <= class_id <= 65535 or not 0 <= prob <= 1:
            raise ValueError("classification outside Camelot tag range")
        tags[tag] = class_id if rank == 1 else (class_id, 1)
        if rank == 1:
            tags[piexif.ExifIFD.ExposureTime] = (min(int(round(prob * 100)), 99), 1)
    return result


def run(config, process=None):
    if process is None:
        from lib_tools import process_detections
        process = process_detections
    config = {**DEFAULTS, **config}
    root = Path(config["INPUT_DIR"]).resolve()
    export, metadata = beneath(root, config["EXIF_DIR"]), beneath(root, config["METADATA_DIR"])
    if root in (export, metadata):
        raise ValueError("output directory must differ from INPUT_DIR")
    if export == metadata or export.is_relative_to(metadata) or metadata.is_relative_to(export):
        raise ValueError("EXIF_DIR and METADATA_DIR must be separate directories")
    for directory in (export, metadata):
        if directory.is_file() or any(parent.is_file() for parent in directory.parents):
            raise ValueError("output directory conflicts with an existing file")
    outputs = {"EN_FILE": beneath(metadata, config["EN_FILE"]),
               "EN_CSV": beneath(metadata, config["EN_CSV"]),
               "metadata.json": beneath(metadata, "metadata.json")}
    for index, (name, path) in enumerate(outputs.items()):
        for other_name, other_path in list(outputs.items())[index + 1:]:
            if path.is_relative_to(other_path) or other_path.is_relative_to(path):
                raise ValueError(f"metadata output paths overlap: {name} and {other_name}")
        if path.is_dir() or any(parent.is_file() for parent in path.parents):
            raise ValueError(f"metadata output path conflicts with an existing file/directory: {name}")
    for key in ("MD_FILE", "EN_FILE"):
        source_artifact = beneath(root, config[key])
        if source_artifact.is_relative_to(export) or source_artifact.is_relative_to(metadata):
            raise ValueError("input artifact lies in an output directory")
    report = dict(schema_version=1, stage="exif", complete=False,
                  suppression_policy=config["SUPPRESSION_POLICY"], images=[], errors=[])
    report_path = outputs["metadata.json"]
    data = json.loads(beneath(root, config["MD_FILE"]).read_text())
    for item in data["images"]:
        source = beneath(root, item["file"])
        if source.is_relative_to(export) or source.is_relative_to(metadata):
            raise ValueError("source image lies in an output directory")
    try:
        frame = bind_predictions(pd.read_pickle(beneath(root, config["EN_FILE"])), data["images"])
        for key in ("image_format", "date_time_orig", "timestamp_source", "timestamp_timezone", "flash_fired", "detections", "eligible", "metadata_status"):
            frame[key] = pd.Series([None] * len(frame), dtype=object)
        for item in data["images"]:
            entry = dict(source_file=item.get("file"), status="error")
            report["images"].append(entry)
            try:
                source = beneath(root, item["file"])
                if source.is_relative_to(export) or source.is_relative_to(metadata):
                    raise ValueError("source image lies in an output directory")
                valid = process(item, config["OVERLAP"], config["EDGE_DIST"], config["MIN_EDGES"], config["UPPER_CONF"], config["LOWER_CONF"], policy=config["SUPPRESSION_POLICY"])
                exif, camera = read_camera_metadata(source)
                entry.update(camera)
                entry["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
                entry["eligible_detection_indices"] = [i for i, keep in enumerate(valid) if keep]
                rows = frame["source_file"] == item["file"]
                for key, value in {**camera, "detections": sum(valid)}.items():
                    frame.loc[rows, key] = value
                frame.loc[rows, "eligible"] = frame.loc[rows, "detection_index"].map(lambda index: valid[index])
                entry["classifications"] = json.loads(frame.loc[rows].to_json(orient="records"))
                destination = beneath(export, item["file"])
                if any(keep and str(item["detections"][i]["category"]) == "1" for i, keep in enumerate(valid)):
                    if camera["image_format"] != "JPEG":
                        raise ValueError("Camelot export requires JPEG; canonical metadata retained")
                    eligible_rows = frame.loc[rows & (frame["eligible"] == True)]
                    expected = {i for i, keep in enumerate(valid) if keep and str(item["detections"][i]["category"]) == "1"}
                    available = set(eligible_rows.loc[eligible_rows["class_rank"] == 1, "detection_index"]) if len(eligible_rows) else set()
                    if expected != available:
                        raise ValueError(f"missing rank-1 predictions for eligible animal indices: {sorted(expected - available)}")
                    adapted = piexif.dump(camelot_metadata(exif, eligible_rows, sum(valid)))
                    # Insert an EXIF segment into a copy; never decode/re-encode JPEG pixels.
                    def export_jpeg(temp):
                        piexif.insert(adapted, str(source), str(temp))
                        shutil.copystat(source, temp)
                    atomic_write(destination, export_jpeg)
                    entry["export_file"], entry["status"] = str(destination.relative_to(root)), "exported"
                else:
                    atomic_write(destination, lambda temp: shutil.copy2(source, temp))
                    entry["export_file"] = str(destination.relative_to(root))
                    entry["status"] = "no_animal"
                entry["export_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
                frame.loc[rows, "metadata_status"] = entry["status"]
            except Exception as error:
                entry["status"] = "error"
                entry["error"] = str(error)
                frame.loc[frame["source_file"] == item.get("file"), "metadata_status"] = "error"
            entry["classifications"] = json.loads(frame.loc[frame["source_file"] == item.get("file")].to_json(orient="records"))
        report["counts"] = {status: sum(e["status"] == status for e in report["images"]) for status in ("exported", "no_animal", "error")}
        atomic_write(outputs["EN_FILE"], frame.to_pickle)
        atomic_write(outputs["EN_CSV"], lambda temp: frame.to_csv(temp, index=False))
        report["complete"] = report["counts"]["error"] == 0
    except Exception as error:
        report["errors"].append(str(error))
        if not report["images"]:
            report["images"] = [dict(source_file=item.get("file"), status="not_processed",
                                     error="stage preflight failed; see errors") for item in data["images"]]
    report["counts"] = {status: sum(e["status"] == status for e in report["images"])
                        for status in ("exported", "no_animal", "error", "not_processed")}
    write_json(report_path, report)
    return report


def main():
    try:
        import yaml
        with open("config.yaml", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
        if not isinstance(loaded, dict):
            raise ValueError("config.yaml must contain a mapping")
        config = {**DEFAULTS, **loaded}
        config.update({key: os.environ[key] for key in config if key in os.environ})
        report = run(config)
        print(json.dumps(dict(complete=report["complete"], counts=report.get("counts", {}),
                              errors=report["errors"], image_errors=[
                                  {"source_file": item["source_file"], "error": item["error"]}
                                  for item in report.get("images", []) if "error" in item])))
        return 0 if report["complete"] else 1
    except Exception as error:
        print(f"EXIF stage failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
