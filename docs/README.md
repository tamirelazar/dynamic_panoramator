# Dynamic Panoramator

## Turns This:
<iframe width="560" height="315" src="https://www.youtube.com/embed/Vwe2E89m5x0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Into This:
<iframe width="560" height="315" src="https://www.youtube.com/embed/sPMCly6xqLI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Details

The program deconstructs the video into frames, and then:
- Identifies features for each frame.
- Calculate the alignment matrix between each frame and the next.
- Create new, panoramic frames from batches of matching frames, thus transforming a video showing lateral movement of camera, into a panoramic video showing perspective change of the same view.

It is extremely cool.

## Usage

Just run make_panorama.py to see my example of the program's function.
If you want to try your own, put a video in the video directory and run the file the same way.

**Side Note:** This won't work with any video. You have to hold your camera in a consistent height, move it rather slowly and stick to lateral movement. To get best results, shoot a subject in the horizon - this program does not handle rotation amazingly.
