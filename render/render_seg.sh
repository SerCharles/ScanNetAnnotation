
base_code=/home/shenguanlin/ScanNetAnnotation/render
base_data=/home1/shenguanlin/scannet_pretrain

cd $base_data

scenes=$(ls)
for scene in $scenes
do 
    cd $base_code
    python render_seg.py --base_dir=$base_data --scene_id=$scene
done 
