
base_code=/home/shenguanlin/ScanNetAnnotation/render
base_data_scannet=/home1/shenguanlin/scannet_pretrain
base_data_plane=/home1/shenguanlin/scannet_planes

cd $base_data_scannet

scenes=$(ls)
for scene in $scenes
do 
    cd $base_code
    python render_layout.py --base_dir_scannet=$base_data_scannet --base_dir_plane=$base_data_plane --scene_id=$scene

done 
