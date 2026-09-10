import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pandas as pd
from PIL import Image
import piexif
import pytest

spec = importlib.util.spec_from_file_location("mewc_exif", Path(__file__).parents[1] / "src/mewc_exif.py")
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)


def detection(conf=0.9, category="1"):
    return dict(conf=conf, category=category, bbox=[0.1, 0.1, 0.5, 0.5])


def process(item, overlap, edge, minimum, upper, lower, policy):
    if item.get("failure") or item.get("detections") is None:
        raise ValueError("detector failure")
    return [d["conf"] >= float(lower) for d in item["detections"]]


def jpeg(path, flash=0x18, exif=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    tags = {"0th": {piexif.ImageIFD.Make: b"Synthetic camera"}, "Exif": {
        piexif.ExifIFD.DateTimeOriginal: b"2024:01:02 03:04:05",
        piexif.ExifIFD.Flash: flash, piexif.ExifIFD.FNumber: (28, 10),
        piexif.ExifIFD.ISOSpeedRatings: 400, piexif.ExifIFD.ExposureTime: (1, 125),
        piexif.ExifIFD.FocalLength: (50, 1)}}
    image = Image.new("RGB", (32, 24), "olive")
    image.putpixel((8, 10), (243, 25, 19))
    image.save(path, **({"exif": piexif.dump(tags)} if exif else {}))
    # Synthetic APP13 segment models preservation of unrelated IPTC metadata bytes.
    payload = b"Photoshop 3.0\x00synthetic metadata sentinel"
    app13 = b"\xff\xed" + (len(payload) + 2).to_bytes(2, "big") + payload
    path.write_bytes(path.read_bytes()[:2] + app13 + path.read_bytes()[2:])
    return app13


def inputs(root, images, rows):
    (root / "md_out.json").write_text(json.dumps({"images": images}))
    pd.DataFrame(rows, columns=["source_file", "detection_index", "filename", "class_rank", "class_id", "prob"]).to_pickle(root / "mewc_out.pkl")


def row(source="site/a.jpg", index=0, rank=1, class_id=7, prob=0.82):
    return [source, index, f"{Path(source).stem}-{index}.jpg", rank, class_id, prob]


def test_lossless_source_and_camelot_mapping(tmp_path):
    source = tmp_path / "site/a.jpg"
    app13 = jpeg(source)
    original = source.read_bytes()
    original_exif = piexif.load(str(source))
    images = [{"file": "site/a.jpg", "detections": [detection(), detection(0.01, "2")]}]
    inputs(tmp_path, images, [row(), row(rank=2, class_id=11, prob=0.15)])
    input_pickle = (tmp_path / "mewc_out.pkl").read_bytes()
    report = metadata.run({"INPUT_DIR": str(tmp_path)}, process)
    assert report["complete"], report
    exported = tmp_path / "camelot/site/a.jpg"
    assert source.read_bytes() == original
    assert piexif.load(str(source)) == original_exif
    assert (tmp_path / "mewc_out.pkl").read_bytes() == input_pickle
    assert app13 in exported.read_bytes()
    assert Image.open(source).tobytes() == Image.open(exported).tobytes()
    assert source.read_bytes()[source.read_bytes().index(b"\xff\xda"):] == exported.read_bytes()[exported.read_bytes().index(b"\xff\xda"):]
    tags = piexif.load(str(exported))["Exif"]
    assert tags[piexif.ExifIFD.FNumber] == (1, 1)
    assert tags[piexif.ExifIFD.ISOSpeedRatings] == 7
    assert tags[piexif.ExifIFD.ExposureTime] == (82, 1)
    assert tags[piexif.ExifIFD.FocalLength] == (11, 1)
    entry = report["images"][0]
    assert entry["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert entry["flash_fired"] == 0
    assert entry["timestamp_source"] == "exif_datetime_original"
    assert entry["timestamp_timezone"] == "unknown"
    enriched = pd.read_csv(tmp_path / "metadata/mewc_out.csv")
    assert enriched["detections"].tolist() == [1, 1]
    assert enriched["metadata_status"].tolist() == ["exported", "exported"]


@pytest.mark.parametrize("flash,expected", [(0, 0), (0x18, 0), (0x19, 1), (1, 1)])
def test_flash_fired_bit(tmp_path, flash, expected):
    source = tmp_path / "a.jpg"
    jpeg(source, flash)
    assert metadata.read_camera_metadata(source)[1]["flash_fired"] == expected


def test_missing_exif_uses_explicit_utc_mtime(tmp_path):
    source = tmp_path / "a.jpg"
    jpeg(source, exif=False)
    os.utime(source, (0, 0))
    _, camera = metadata.read_camera_metadata(source)
    assert camera["date_time_orig"] == "1970-01-01T00:00:00+00:00"
    assert camera["timestamp_source"] == "filesystem_mtime"
    assert camera["timestamp_timezone"] == "UTC"
    assert camera["flash_fired"] is None


def test_exact_index_and_stem_joins():
    images = [{"file": name, "detections": [detection(i / 20) for i in range(11)]} for name in ("site/a.jpg", "site/ab.jpg")]
    frame = pd.DataFrame({"filename": ["a-1.jpg", "a-10.jpg", "ab-1.jpg"]})
    bound = metadata.bind_predictions(frame, images)
    assert bound["detection_index"].tolist() == [1, 10, 1]
    assert bound["source_file"].tolist() == ["site/a.jpg", "site/a.jpg", "site/ab.jpg"]
    assert bound["conf"].tolist() == [0.05, 0.5, 0.05]


def test_ambiguous_basename_rejected_but_relative_path_exact():
    images = [{"file": f"{site}/a.jpg", "detections": [detection()]} for site in ("site1", "site2")]
    with pytest.raises(ValueError, match="ambiguous"):
        metadata.bind_predictions(pd.DataFrame({"filename": ["a-0.jpg"]}), images)
    result = metadata.bind_predictions(pd.DataFrame({"filename": ["site2/a-0.jpg"]}), images)
    assert result.loc[0, "source_file"] == "site2/a.jpg"


@pytest.mark.parametrize("name", ["a-0.jpg.extra", "a-0", "aa-0.jpg"])
def test_legacy_prefix_matches_rejected(name):
    with pytest.raises(ValueError, match="unmatched"):
        metadata.bind_predictions(pd.DataFrame({"filename": [name]}), [{"file": "a.jpg", "detections": [detection()]}])


def test_ties_retained_canonical_and_export_fails(tmp_path):
    jpeg(tmp_path / "site/a.jpg")
    inputs(tmp_path, [{"file": "site/a.jpg", "detections": [detection()]}], [row(prob=0.5), row(class_id=8, prob=0.5)])
    report = metadata.run({"INPUT_DIR": str(tmp_path)}, process)
    assert not report["complete"]
    assert report["counts"]["error"] == 1
    assert len(report["images"][0]["classifications"]) == 2
    assert "tied" in report["images"][0]["error"]
    assert not (tmp_path / "camelot/site/a.jpg").exists()
    assert set(pd.read_csv(tmp_path / "metadata/mewc_out.csv")["metadata_status"]) == {"error"}


def test_failure_accounting_and_missing_prediction(tmp_path):
    jpeg(tmp_path / "site/a.jpg")
    inputs(tmp_path, [{"file": "site/a.jpg", "detections": [detection()]}, {"file": "missing.jpg", "detections": None, "failure": "broken"}], [])
    report = metadata.run({"INPUT_DIR": str(tmp_path)}, process)
    assert not report["complete"]
    assert report["counts"]["error"] == 2
    assert "missing rank-1" in report["images"][0]["error"]
    assert "detector failure" in report["images"][1]["error"]


def test_non_jpeg_canonical_survives_unsupported_export(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    inputs(tmp_path, [{"file": "a.png", "detections": [detection()]}], [row("a.png")])
    report = metadata.run({"INPUT_DIR": str(tmp_path)}, process)
    assert not report["complete"]
    assert len(report["images"][0]["classifications"]) == 1
    assert "requires JPEG" in report["images"][0]["error"]


def test_output_directory_escape_and_alias_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        metadata.run({"INPUT_DIR": str(tmp_path), "EXIF_DIR": ".."}, process)
    (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="differ"):
        metadata.run({"INPUT_DIR": str(tmp_path), "EXIF_DIR": "alias"}, process)


def test_cli_returns_failure(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "lib_common", SimpleNamespace(read_yaml=lambda _: {}))
    monkeypatch.setattr(metadata, "run", lambda _: {"complete": False, "errors": []})
    assert metadata.main() == 1


def test_no_eligible_animal_is_copied_with_mtime(tmp_path):
    jpeg(tmp_path / "blank.jpg", exif=False)
    original = (tmp_path / "blank.jpg").read_bytes()
    os.utime(tmp_path / "blank.jpg", (123, 456))
    inputs(tmp_path, [{"file": "blank.jpg", "detections": [detection(0.01)]}], [])
    report = metadata.run({"INPUT_DIR": str(tmp_path)}, process)
    assert report["complete"], report
    assert report["images"][0]["status"] == "no_animal"
    assert (tmp_path / "camelot/blank.jpg").read_bytes() == original
    assert (tmp_path / "camelot/blank.jpg").stat().st_mtime == 456


def test_per_image_export_write_failure_reaches_report(tmp_path, monkeypatch):
    jpeg(tmp_path / "site/a.jpg")
    inputs(tmp_path, [{"file": "site/a.jpg", "detections": [detection()]}], [row()])
    def fail_insert(*args):
        raise OSError("injected export failure")
    monkeypatch.setattr(piexif, "insert", fail_insert)
    report = metadata.run({"INPUT_DIR": str(tmp_path)}, process)
    assert not report["complete"]
    assert "injected export failure" in report["images"][0]["error"]
    assert report["images"][0]["classifications"][0]["metadata_status"] == "error"


def test_source_inside_output_is_rejected_before_any_write(tmp_path):
    inputs(tmp_path, [{"file": "metadata/metadata.json", "detections": []}], [])
    (tmp_path / "metadata").mkdir()
    sentinel = tmp_path / "metadata/metadata.json"
    sentinel.write_bytes(b"original sentinel")
    with pytest.raises(ValueError, match="source image"):
        metadata.run({"INPUT_DIR": str(tmp_path)}, process)
    assert sentinel.read_bytes() == b"original sentinel"


def test_explicit_camera_offset_is_preserved(tmp_path):
    source = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8)).save(source, exif=piexif.dump({"Exif": {
        piexif.ExifIFD.DateTimeOriginal: b"2024:01:02 03:04:05", 36881: b"+10:00"}}))
    assert metadata.read_camera_metadata(source)[1]["timestamp_timezone"] == "+10:00"


def test_ambiguous_join_records_unprocessed_images(tmp_path):
    images = [{"file": f"{site}/a.jpg", "detections": [detection()]} for site in ("one", "two")]
    (tmp_path / "md_out.json").write_text(json.dumps({"images": images}))
    pd.DataFrame({"filename": ["a-0.jpg"]}).to_pickle(tmp_path / "mewc_out.pkl")
    report = metadata.run({"INPUT_DIR": str(tmp_path)}, process)
    assert not report["complete"]
    assert report["counts"]["not_processed"] == 2
    assert "ambiguous" in report["errors"][0]


def test_final_export_verification_failure_is_not_success(tmp_path, monkeypatch):
    jpeg(tmp_path / "site/a.jpg")
    inputs(tmp_path, [{"file": "site/a.jpg", "detections": [detection()]}], [row()])
    original_hash = hashlib.sha256
    count = 0
    def fail_second_hash(value):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("injected final verification failure")
        return original_hash(value)
    monkeypatch.setattr(hashlib, "sha256", fail_second_hash)
    report = metadata.run({"INPUT_DIR": str(tmp_path)}, process)
    assert not report["complete"]
    assert report["counts"]["error"] == 1


@pytest.mark.parametrize("changes", [
    {"EN_CSV": "mewc_out.pkl"},
    {"EN_FILE": "metadata.json"},
    {"EN_CSV": "metadata.json"},
    {"EN_CSV": "mewc_out.pkl/table.csv"},
    {"EN_FILE": "tables/output.pkl", "EN_CSV": "tables"},
    {"EN_CSV": "metadata.json/table.csv"},
])
def test_output_path_collisions_fail_before_any_write(tmp_path, monkeypatch, changes):
    monkeypatch.setattr(metadata, "atomic_write", lambda *_: pytest.fail("invalid configuration wrote output"))
    with pytest.raises(ValueError, match="metadata output paths overlap"):
        metadata.run({"INPUT_DIR": str(tmp_path), **changes}, process)
    assert list(tmp_path.iterdir()) == []


def test_resolved_output_alias_collision_fails_before_any_write(tmp_path, monkeypatch):
    directory = tmp_path / "metadata"
    directory.mkdir()
    (directory / "alias.csv").symlink_to(directory / "mewc_out.pkl")
    monkeypatch.setattr(metadata, "atomic_write", lambda *_: pytest.fail("invalid configuration wrote output"))
    with pytest.raises(ValueError, match="metadata output paths overlap"):
        metadata.run({"INPUT_DIR": str(tmp_path), "EN_CSV": "alias.csv"}, process)
    assert not (directory / "mewc_out.pkl").exists()


def test_output_symlink_to_input_is_rejected_before_writes(tmp_path, monkeypatch):
    original = tmp_path / "mewc_out.pkl"
    original.write_bytes(b"immutable input sentinel")
    directory = tmp_path / "metadata"
    directory.mkdir()
    (directory / "alias.csv").symlink_to(original)
    monkeypatch.setattr(metadata, "atomic_write", lambda *_: pytest.fail("invalid configuration wrote output"))
    with pytest.raises(ValueError, match="escapes"):
        metadata.run({"INPUT_DIR": str(tmp_path), "EN_CSV": "alias.csv"}, process)
    assert original.read_bytes() == b"immutable input sentinel"


@pytest.mark.parametrize("existing", ["directory", "parent_file"])
def test_existing_output_file_directory_conflicts_fail_before_writes(tmp_path, monkeypatch, existing):
    directory = tmp_path / "metadata"
    directory.mkdir()
    output = "table.csv"
    if existing == "directory":
        (directory / output).mkdir()
    else:
        (directory / "parent").write_bytes(b"existing sentinel")
        output = "parent/table.csv"
    monkeypatch.setattr(metadata, "atomic_write", lambda *_: pytest.fail("invalid configuration wrote output"))
    with pytest.raises(ValueError, match="existing file/directory"):
        metadata.run({"INPUT_DIR": str(tmp_path), "EN_CSV": output}, process)


@pytest.mark.parametrize("key,value", [("METADATA_DIR", "metadata"), ("EXIF_DIR", "blocked/camelot")])
def test_output_directory_blocked_by_file_fails_before_writes(tmp_path, monkeypatch, key, value):
    blocker = tmp_path / value.split("/")[0]
    blocker.write_bytes(b"existing sentinel")
    monkeypatch.setattr(metadata, "atomic_write", lambda *_: pytest.fail("invalid configuration wrote output"))
    with pytest.raises(ValueError, match="output directory conflicts"):
        metadata.run({"INPUT_DIR": str(tmp_path), key: value}, process)
    assert blocker.read_bytes() == b"existing sentinel"
