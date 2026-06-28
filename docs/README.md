# Panoramic Stereo Mosaicing

A faithful replication of stereo mosaicing from a single moving video camera, after Peleg, Ben-Ezra & Pritch (2001). Given a video panned steadily across a scene, it reconstructs a panorama whose perspective shifts as the viewpoint moves.

> Shmuel Peleg, Michael Ben-Ezra, and Yael Pritch. "Stereo mosaicing from a single moving video camera." *Proc. SPIE 4297, Stereoscopic Displays and Virtual Reality Systems VIII* (2001). https://doi.org/10.1117/12.430806

## The result

<video src="demo.mp4" autoplay loop muted playsinline width="720"></video>

![Mosaiced result](demo.gif)

<sub>Demo generated from the "boat" example sequence from the Hebrew University Image Processing course.</sub>

## What this is

this is a from-scratch replication of the paper's method, written for coursework.

## From input to result

The full source clip and the mosaiced output:

### Original video
<iframe width="560" height="315" src="https://www.youtube.com/embed/Vwe2E89m5x0" title="Original video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### After mosaicing
<iframe width="560" height="315" src="https://www.youtube.com/embed/sPMCly6xqLI" title="After mosaicing" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## How it works

The pipeline deconstructs the video into frames, then:

1. Detects Harris corner features in each frame and builds descriptors.
2. Matches features between consecutive frames and estimates the alignment (homography, translation-only by default) with RANSAC.
3. Assembles new panoramic frames from strips of many aligned source frames — turning lateral camera motion into a synthetic change of viewpoint across the output.

## Reproduce

Requires Python 3.12 and `ffmpeg` on PATH.

```bash
python -m venv .venv && source .venv/bin/activate   # or: uv venv --python 3.12 .venv
pip install -r requirements.txt
python make_panorama.py
```

This runs the bundled example (`videos/boat.mp4`) and writes the mosaic video to the repository root. To try your own clip, drop a video in `videos/` and point `make_panorama.py` at it.

## Limitations

The method assumes a specific capture: steady camera height, slow lateral pan, no rotation, subject near the horizon.

## Improved version

[TBA]

Code and full details: [github.com/tamirelazar/dynamic_panoramator](https://github.com/tamirelazar/dynamic_panoramator)
