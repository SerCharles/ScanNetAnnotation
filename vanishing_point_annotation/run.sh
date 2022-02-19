#Render the vanishing point annotation result

base_code=/home/shenguanlin/ScanNetAnnotation/vanishing_point_annotation
base_data_scannet=/home1/sgl/scannet_mine
base_data_plane=/home1/sgl/scannet_planes_mine


cd $base_data_scannet

scenes=$(ls)
for scene in $scenes
do 
    cd $base_data_scannet/$scene/color
    picture_num=$(ls -l |grep "^-"|wc -l)
    cd $base_code
    i=0
    while [ $i -lt $picture_num ];
    do 
        python main.py --base_dir_scannet=$base_data_scannet --base_dir_plane=$base_data_plane --scene_id=$scene --start_num=$i --number_per_round=100
        i=`expr $i + 100`;
    done
done 

