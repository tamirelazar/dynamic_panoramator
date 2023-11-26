# Panoramic Stereo Mosaicing
Implementation of Shmuel Peleg's work on stereo mosaicing from video, e.g. in the article cited below:

Shmuel Peleg, Michael Ben-Ezra, and Yael Pritch "Stereo mosaicing from a single moving video camera", Proc. SPIE 4297, Stereoscopic Displays and Virtual Reality Systems VIII, (22 June 2001); https://doi.org/10.1117/12.430806

## Description
Given a video with consistent vertical movement, creates a static perspective-changing panoramic video.

## Demonstration

### Original Video:
<iframe width="560" height="315" src="https://www.youtube.com/embed/Vwe2E89m5x0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### After Mosaicing:
<iframe width="560" height="315" src="https://www.youtube.com/embed/sPMCly6xqLI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Details

The program deconstructs the video into frames, and then:
1. Identifies features for each frame.
2. Calculate the alignment matrix between each frame and the next.
3. Create new, panoramic frames from batches of matching frames, thus transforming a video showing lateral movement of camera, into a panoramic video showing perspective change of the same view.

### Usage

Just run make_panorama.py to see my example of the program's function.
If you want to try your own, put a video in the video directory and run the file the same way.

**Side Note:** This won't work with any video. You have to hold your camera in a consistent height, move it rather slowly and stick to lateral movement. To get best results, shoot a subject in the horizon - this program does not handle rotation amazingly.
