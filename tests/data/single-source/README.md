# Description

Data for testing single source DOA estimation.

## Naming

`single-source_<source frequency in Hz>Hz_<DOA in degree>deg_<duration in s>s.npy`

## Content

The content are serialized numpy array containing 4 channel data in as complex values. The antenna array has the following
geometry (x, y): [(0.113, 0.0) (-0.036, 0.0), (-0.076, 0.0), (-0.113, 0.0)], which is the geometry provided by the microphone
array of the kinect. Sampling rate is 16kHz.

