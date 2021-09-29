#Render the vanishing point annotation result

base_code=/home/shenguanlin/ScanNetAnnotation/vanishing_point_annotation
base_data_scannet=/home1/shenguanlin/scannet_pretrain
base_data_plane=/home1/shenguanlin/scannet_planes_mine


cd $base_data_scannet

scenes=$(ls)
for scene in $scenes
do 
    cd $base_code
    python main.py --base_dir_scannet=$base_data_scannet --base_dir_plane=$base_data_plane --scene_id=$scene
done 

