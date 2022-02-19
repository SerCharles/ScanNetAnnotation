#!/bin/bash
#Render the basic info of depth and norms of our ScanNet Dataset

base_code=/home/shenguanlin/ScanNetAnnotation/render
base_data=/home1/sgl/scannet_mine

cd $base_data

scenes=$(ls)
for scene in $scenes
do 
    if [ -d $base_data/$scene/depth ]
    then
        echo $scene
    else
        cd $base_code
        python render_basic.py --base_dir=$base_data --scene_id=$scene
        cd $base_data/$scene 
    fi
done 
