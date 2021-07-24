This the repository used to generate the data used in our Layout Estimation code based on [ScanNet](https://github.com/ScanNet/ScanNet) and [ScanNet-Planes](https://github.com/skanti/SceneCAD).



Before running the pipeline, please remember two things:

1. Please remember setting the directories of the shell files and python files before running them.
2. The running process is very time consuming, please use nohup or tmux or other methods to keep it running at background.



First, you should get the ScanNet data and code from [here](https://github.com/ScanNet/ScanNet), and ScanNet-Planes data from [here](https://github.com/skanti/SceneCAD).

Second, you should build the python environment used in this code.

```
pip install -r requirements.txt
```

Third, you should load the ScanNet data from the downloaded place, 

```
cd load_data
sh load_data.sh
cd ..
```

Fourth, you should modify the ScanNet-Planes data used in further work.

```
cd background_annotation
python get_background_mesh.py
python get_background_pointcloud.py
cd ..
```

Fifth, you should generate the segmentation file.

```
cd load_data
python get_seg.py
cd ..
```

Sixth, you should render the depth and normal data based on PyRender. In this step, for faster speed, you 'd better change the lib of PyRender to the one in [AdversarialTexture](https://github.com/hjwdzh/AdversarialTexture).

```
cd render/lib
sh compile.sh
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
cd ..
sh render_basic.sh
cd ..
```

Seventh, you should render the layout data based on PyRender. In this step, you must use the lib of PyRender provided by my repository.

```
cd render/lib
sh compile.sh
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
cd ..
sh render_layout.sh
cd ..
```

Eighth, you should clear the data and split the training and validation data.

```
cd load_data
python split_data.py --clear=1
cd ..
```

After the eight steps, the whole dataset is built, and can be used in our Layout Estimation task.