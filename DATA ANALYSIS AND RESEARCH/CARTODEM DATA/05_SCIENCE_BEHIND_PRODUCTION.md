# The Science Behind Producing CartoDEM

## 1. Photogrammetry and Parallax
The science behind generating a DEM from space is **Photogrammetry**.
Hold your finger in front of your face and close your left eye, then your right eye. Your finger appears to jump back and forth against the background. This is **Parallax**.
Because your eyes are spaced apart, your brain uses the difference between the two images to calculate depth (how far away your finger is).

## 2. Stereoscopic Calculation
Cartosat-1 does the exact same thing. Because it takes two photos of a mountain from two different angles (separated by hundreds of kilometers in space), the mountain peak appears shifted relative to the flat ground beneath it.
Supercomputers measure the exact pixel distance of this shift. Using complex trigonometry, orbital mechanics, and the known distance between the two camera shots, the computer calculates the exact height of the mountain peak down to the meter.
