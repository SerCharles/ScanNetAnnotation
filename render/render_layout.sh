#Render the basic info of layout seg, depth and norms of our ScanNet Dataset

base_code=/home/shenguanlin/ScanNetAnnotation/render
base_data_scannet=/home1/sgl/scannet_mine
base_data_plane=/home1/sgl/scannet_planes_mine

cd $base_data_scannet

scenes=$(ls)
for scene in $scenes
do 
    if [ -d $base_data_scannet/$scene/layout_depth ]
    then
        echo $scene
    else
        cd $base_code
        python render_layout.py --base_dir_scannet=$base_data_scannet --base_dir_plane=$base_data_plane --scene_id=$scene
    fi
done 
