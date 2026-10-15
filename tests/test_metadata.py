import copy
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
    with Image.open(source) as source_image, Image.open(exported) as exported_image:
        assert source_image.tobytes() == exported_image.tobytes()
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


@pytest.mark.parametrize("probabilities,percent", [((.822, .824), 82), ((.995, 1.0), 99)])
def test_equivalent_encoded_ties_export_without_changing_predictions(tmp_path, probabilities, percent):
    source = tmp_path / "site/a.jpg"
    app13 = jpeg(source)
    original = source.read_bytes()
    images = [{"file": "site/a.jpg", "detections": [detection(), detection(), detection(.8)]}]
    predictions = [row(index=0, prob=probabilities[0]), row(index=1, prob=probabilities[1]),
                   row(index=2, class_id=8, prob=.99),
                   row(index=0, rank=2, class_id=11, prob=.15),
                   row(index=1, rank=2, class_id=11, prob=.2),
                   row(index=2, rank=2, class_id=12, prob=.9)]
    inputs(tmp_path, images, predictions)
    original_pickle = (tmp_path / "mewc_out.pkl").read_bytes()
    original_frame = pd.read_pickle(tmp_path / "mewc_out.pkl")

    report = metadata.run({"INPUT_DIR": str(tmp_path)}, process)

    assert report["complete"], report
    exported = tmp_path / "camelot/site/a.jpg"
    tags = piexif.load(str(exported))["Exif"]
    assert tags[piexif.ExifIFD.FNumber] == (3, 1)
    assert tags[piexif.ExifIFD.ISOSpeedRatings] == 7
    assert tags[piexif.ExifIFD.ExposureTime] == (percent, 1)
    assert tags[piexif.ExifIFD.FocalLength] == (11, 1)
    assert source.read_bytes() == original
    assert (tmp_path / "mewc_out.pkl").read_bytes() == original_pickle
    assert app13 in exported.read_bytes()
    assert original[original.index(b"\xff\xda"):] == exported.read_bytes()[exported.read_bytes().index(b"\xff\xda"):]
    for enriched in (pd.read_pickle(tmp_path / "metadata/mewc_out.pkl"),
                     pd.read_csv(tmp_path / "metadata/mewc_out.csv")):
        pd.testing.assert_frame_equal(enriched[original_frame.columns], original_frame, check_exact=True)
        assert enriched["metadata_status"].tolist() == ["exported"] * len(predictions)
    assert len(report["images"][0]["classifications"]) == len(predictions)


@pytest.mark.parametrize("rank,codes,probabilities", [
    (1, [7, 7], [.824, .826]),
    (1, [7, 8], [.822, .824]),
    (2, [7, 8], [.822, .824]),
])
def test_different_encoded_tags_still_fail_in_either_order(rank, codes, probabilities):
    records = [dict(class_rank=rank, class_id=code, prob=prob, conf=.9)
               for code, prob in zip(codes, probabilities)]
    for ordered in (records, records[::-1]):
        with pytest.raises(ValueError, match=f"Camelot rank {rank}.*tied highest-confidence"):
            metadata.camelot_metadata({"Exif": {}}, pd.DataFrame(ordered), 2)


@pytest.mark.parametrize("rank", [1, 2])
def test_canonical_class_codes_share_projection_without_mutating_inputs(rank):
    frame = pd.DataFrame([dict(class_rank=rank, class_id=code, prob=.82, conf=.9)
                          for code in (7, "7", 7.0)])
    exif = {"Exif": {41988: 5}, "0th": {piexif.ImageIFD.Make: b"Original camera"}}
    before_frame, before_exif = frame.copy(deep=True), copy.deepcopy(exif)
    first = metadata.camelot_metadata(exif, frame, 3)
    second = metadata.camelot_metadata(exif, frame.iloc[::-1], 3)
    assert first == second
    tag = piexif.ExifIFD.ISOSpeedRatings if rank == 1 else piexif.ExifIFD.FocalLength
    assert first["Exif"][tag] == (7 if rank == 1 else (7, 1))
    assert first["Exif"][41988] == (5, 1)
    assert first["0th"] == exif["0th"]
    assert exif == before_exif
    pd.testing.assert_frame_equal(frame, before_frame, check_exact=True)


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("invalid", [True, False, "07", "+7", "7.0", " 7", "7 ",
                                    -1, 65536, 7.5, float("nan"), float("inf"), None])
def test_every_selected_class_code_is_validated(rank, invalid):
    valid = int(invalid) if isinstance(invalid, bool) else 7
    frame = pd.DataFrame([dict(class_rank=rank, class_id=code, prob=.82, conf=.9)
                          for code in (valid, invalid)], dtype=object)
    for ordered in (frame, frame.iloc[::-1]):
        with pytest.raises(ValueError, match="classification outside Camelot tag range"):
            metadata.camelot_metadata({"Exif": {}}, ordered, 2)


@pytest.mark.parametrize("invalid", [-.001, 1.001, float("nan"), float("inf"), "invalid", None])
def test_every_selected_probability_is_validated_before_encoding(invalid):
    frame = pd.DataFrame([dict(class_rank=1, class_id=7, prob=prob, conf=.9)
                          for prob in (.82, invalid)], dtype=object)
    for ordered in (frame, frame.iloc[::-1]):
        with pytest.raises(ValueError, match="classification outside Camelot tag range"):
            metadata.camelot_metadata({"Exif": {}}, ordered, 2)


@pytest.mark.parametrize("probability,percent", [(0, 0), (.005, 0), (.125, 12), (.135, 14),
                                               (.985, 98), (1, 99)])
def test_camelot_percentage_rounding_and_cap_are_unchanged(probability, percent):
    frame = pd.DataFrame([dict(class_rank=1, class_id=7, prob=probability, conf=.9)])
    result = metadata.camelot_metadata({"Exif": {}}, frame, 1)
    assert result["Exif"][piexif.ExifIFD.ExposureTime] == (percent, 1)


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


def test_cli_returns_failure(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "lib_common", None)
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


def test_original_decimal_string_class_code_exports_without_axis_recoding():
    frame = pd.DataFrame([dict(class_rank=1, class_id='999', prob=.83, conf=.9)])
    result = metadata.camelot_metadata({'Exif': {}}, frame, 1)
    assert result['Exif'][piexif.ExifIFD.ISOSpeedRatings] == 999


def test_shipped_config_matches_stage_defaults():
    import yaml
    shipped = yaml.safe_load(Path(metadata.__file__).with_name("config.yaml").read_text())
    assert shipped == metadata.DEFAULTS
    assert all(value is not None for value in shipped.values())


def test_main_shipped_config_exports_without_flow_lib_common(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    jpeg(tmp_path / "site/a.jpg")
    original = (tmp_path / "site/a.jpg").read_bytes()
    inputs(tmp_path, [{"file": "site/a.jpg", "detections": [detection()]}], [row()])
    (tmp_path / "config.yaml").write_text(Path(metadata.__file__).with_name("config.yaml").read_text())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "lib_common", None)
    monkeypatch.setitem(sys.modules, "lib_tools", SimpleNamespace(process_detections=process))
    assert metadata.main() == 0
    assert json.loads(capsys.readouterr().out)["complete"] is True
    assert (tmp_path / "camelot/site/a.jpg").is_file()
    assert (tmp_path / "site/a.jpg").read_bytes() == original
    assert json.loads((tmp_path / "metadata/metadata.json").read_text())["complete"] is True


@pytest.mark.parametrize("document", ["", "- INPUT_DIR\n", "scalar\n"])
def test_main_rejects_nonmapping_yaml_before_processing(tmp_path, monkeypatch, capsys, document):
    (tmp_path / "config.yaml").write_text(document)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "lib_common", None)
    monkeypatch.setattr(metadata, "run", lambda _: pytest.fail("invalid YAML started processing"))
    assert metadata.main() == 1
    assert "config.yaml must contain a mapping" in capsys.readouterr().out
