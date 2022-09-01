# mewc-exif

## Introduction
This repository contains code to build a Docker container for running mewc-exif. This is a tool used to write classifier data from en-predict and MegaDetector detection data to the EXIF fields of camera trap images. This can be useful when importing camera trap images into organisational tools like [Camelot](https://camelotproject.org). Because EXIF data is specific to camera brands and the specific EXIF tags do not support every data type this is an experimental image and should be treated as such. EXIF data will be written to an image if it contains an animal and has valid detections.

You can supply arguments via an environment file where the contents of that file are in the following format with one entry per line:
```
VARIABLE=VALUE
```

## Usage

After installing Docker you can run the container using a command similar to the following. Substitute `"$IN_DIR"` for your image directory and create a text file `"$ENV_FILE"` with any config options you wish to override. 

```
docker pull zaandahl/mewc-exif:v1.0
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
| EN_FILE | "en_out.pkl" | PKL file from en-predict Efficient Net output. Must be located in INPUT_DIR |
| EN_CSV | "en_out.csv" | CSV file from en-predict Efficient Net output. Must be located in INPUT_DIR  |
| LOWER_CONF | 0.05 | The lowest detection confidence threshold to accept |
| OVERLAP | 0.3 | Matryoshka reduction - minimum proportional shared area for two boxes to be considered overlapping  |
| EDGE_DIST | 0.02 | Matryoshka reduction - minimum proportional edge distance for two boxes to share a 'close' edge |
| MIN_EDGES | 0 | Matryoshka reduction - minimum number of 'close' edges to consider removing smaller overlapped box |
| UPPER_CONF | 0.9 | Matryoshka reduction - upper detection confidence to give a 'free pass' for detection boxes|

