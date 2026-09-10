<img src="mewc_logo_hex.png" alt="MEWC Hex Sticker" width="200" align="right"/>

# mewc-exif

The integrity changes in this checkout require the source builds described in [BUILDING.md](BUILDING.md). Existing DockerHub examples do not provide these fixes until a maintainer publishes a compatible release. Use the tested image ID or digest from the generated image lock.

## Introduction
This repository contains code to build a Docker container for running mewc-exif. This tool extracts canonical metadata and creates separate Camelot export copies from classifier and MegaDetector results. Source images and input prediction files remain unchanged. This can be useful when importing camera trap images into organisational tools like [Camelot](https://camelotproject.org). Because EXIF data is specific to camera brands and the specific EXIF tags do not support every data type this is an experimental image and should be treated as such. Eligible animal JPEGs receive the Camelot metadata mapping in the export tree. Other images are copied unchanged so the export tree remains complete. An unsupported format for an eligible animal, ambiguous classification, or failed image makes the stage exit nonzero; canonical metadata remains available for review.

You can supply arguments via an environment file where the contents of that file are in the following format with one entry per line:
```
VARIABLE=VALUE
```

## Usage

After installing Docker you can run the container using a command similar to the following. Substitute `"$IN_DIR"` for your image directory and create a text file `"$ENV_FILE"` with any config options you wish to override. 

```
docker pull zaandahl/mewc-exif
docker run --env-file "$ENV_FILE" \
    --interactive --tty --rm \
    --volume "$IN_DIR":/images \
    zaandahl/mewc-exif
```

## Config Options

The following environment variables are supported for configuration (and their default values are shown). Simply omit any variables you don't need to change and if you want to just use all defaults you can leave `--env-file $ENV_FILE` out of the command alltogether. The last four options are designed to reduce a common effect where multiple spurious detection boxes are cascaded over a single animal in an effect similar to [Matryoshka](https://en.wikipedia.org/wiki/Matryoshka_doll) nesting dolls. 

| Variable | Default | Description |
| ---------|---------|------------ |
| INPUT_DIR | "/images/" | A mounted point containing images to process - must match the Docker command above |
| MD_FILE | "md_out.json" | MegaDetector output file, must be located in INPUT_DIR |
| EN_FILE | "mewc_out.pkl" | PKL file from en-predict Efficient Net output. Must be located in INPUT_DIR |
| EN_CSV | "mewc_out.csv" | CSV file from en-predict Efficient Net output. Must be located in INPUT_DIR  |
| EXIF_DIR | "camelot" | Separate export tree under INPUT_DIR; preserves every original relative image path |
| METADATA_DIR | "metadata" | Canonical metadata.json and enriched EN_FILE/EN_CSV; input tables are never overwritten |
| SUPPRESSION_POLICY | "category-confidence-v1" | Shared, versioned detection eligibility; legacy-matryoshka-v1 is an explicit historical alternative |
| LOWER_CONF | 0.05 | The lowest detection confidence threshold to accept |
| OVERLAP | 0.3 | Matryoshka reduction - minimum proportional shared area for two boxes to be considered overlapping  |
| EDGE_DIST | 0.02 | Matryoshka reduction - minimum proportional edge distance for two boxes to share a 'close' edge |
| MIN_EDGES | 0 | Matryoshka reduction - minimum number of 'close' edges to consider removing smaller overlapped box |
| UPPER_CONF | 0.9 | Matryoshka reduction - upper detection confidence to give a 'free pass' for detection boxes|


## Identity, metadata and data preservation

Prediction rows use `source_file` (the full POSIX path relative to `INPUT_DIR`, including extension) and `detection_index` (zero-based index in the original MegaDetector detection list). `crop_id` is retained when supplied. A legacy `filename` is accepted only if its complete relative crop name or basename has exactly one match to `<source-stem>-<original-index><extension>`. Ambiguous duplicate basenames, stem-prefix matches, and index-prefix matches fail instead of guessing.

`METADATA_DIR/metadata.json` is the canonical stage record: it records every image's status, source SHA-256, original detection indices retained by the configured policy, all prediction rows (including ties), camera timestamp provenance, flash state, and any export failure. Enriched pickle/CSV files are saved after image processing with `metadata_status`. `complete=false` or a nonzero exit means the output is partial; do not consume partial or stale files from an earlier run as successful output. Use a fresh output directory for each run.

The enriched `EN_FILE`, `EN_CSV` and reserved `metadata.json` output paths must be distinct, with none a parent of another. Their resolved paths must remain inside `METADATA_DIR`; path collisions, symlink escapes and existing file/directory conflicts fail before any output is written.

`date_time_orig` preserves camera `DateTimeOriginal` text when present. `timestamp_source=exif_datetime_original` and `timestamp_timezone=unknown` explicitly record a missing camera offset; an existing `OffsetTimeOriginal` is retained. Missing camera timestamps use filesystem modification time formatted in UTC, with `timestamp_source=filesystem_mtime` and `timestamp_timezone=UTC`. That fallback is file metadata, not an inferred capture time. Missing flash metadata is null; otherwise only EXIF Flash bit 0 determines whether flash fired (`0x18` is false; `0x19` is true).

Original JPEG bytes and camera EXIF remain immutable. The Camelot adapter inserts an EXIF segment into a new JPEG without re-encoding image data; unrelated JPEG metadata segments are retained. Source modification times are preserved in export copies. The adapter intentionally repurposes camera fields only in these copies:

| EXIF field | Camelot export value |
| --- | --- |
| FNumber (33437) | Count of all eligible detector boxes, including any person or vehicle boxes; this is not animal abundance |
| ISOSpeedRatings (34855) | Rank-1 class ID from the eligible animal crop with highest detector confidence |
| ExposureTime (33434) | That class probability rounded to percent and capped at 99, stored as a rational with denominator 1 |
| FocalLength (37386) | Rank-2 class ID selected by the same highest-detector-confidence rule; omitted when rank 2 is unavailable |

The mapping preserves the established Camelot camera-field convention. Tied highest-confidence candidates with different class/probability values cannot fit a single camera tag and produce an explicit export failure; all ties survive in the canonical rows. Missing rank-1 predictions for any eligible animal also fail. No eligible animal means a byte-identical copy, without repurposing camera fields. Eligible non-JPEG animals retain canonical metadata but fail Camelot export, which currently supports JPEG only.

New runs use `category-confidence-v1`: lower-confidence filtering followed by higher-confidence-first suppression within each detector category, with the configured overlap/edge/confidence thresholds. The shared detector helper defines this policy. Historical `legacy-matryoshka-v1` remains explicit for reproduction; it may depend on order and suppress across categories. This release does not rewrite historical results. Metadata, snipping and boxing should receive the same policy and numeric thresholds. Crop counts include eligible animals only; FNumber counts all eligible categories.

## Verification

Run `python -m pytest -q tests` in an environment containing pytest, pandas, Pillow and piexif. Synthetic fixtures verify exact path/index joins, ambiguity rejection, immutable originals/input tables, lossless JPEG scan and pixel preservation, unrelated APP13 metadata retention, the Camelot tag mapping, flash decoding, timestamp provenance, tie/failure accounting and unsupported-format handling. Runtime eligibility comes from the matching `mewc-detect` shared helper.
