First, run this below to extract the planes and lines of the layout. The result will be a 3D mesh file.

```
python extract_plane.py
```



Then, run the code from ScanNet to extract the poses in the ScanNet dataset, then use

```
python pose_loader.py
```

to select the poses



Then, put the poses into the OpenGLRender base dir, and run the C++ program to get the pictures