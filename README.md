# dynamic_panoramator
Course assignment for IMPR course at HUJI.
Given a video with consistent vertical movement, creates a static perspective-changing panoramic video.

### Details

The program deconstructs the video into frames, and then:
1. Identifies features for each frame.
2. Calculate the alignment matrix between each frame and the next.
3. Create new, panoramic frames from batches of matching frames, thus transforming a video showing lateral movement of camera, into a panoramic video showing perspective change of the same view.

It is extremely cool.


### Usage

Just run make_panorama.py to see my example of the program's function.
If you want to try your own, put a video in the video directory and run the file the same way.
**Side Note:** This won't work with any video. You have to hold your camera in a consistent height, move it rather slowly and stick to lateral movement. To get best results, shoot a subject in the horizon - this program does not handle rotation amazingly.
