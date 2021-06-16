
base_code=/home/shenguanlin/render
base_data=/home1/shenguanlin/scannet_pretrain

cd $base_data

scenes=$(ls)
for scene in $scenes
do 
    cd $base_code
    python render_basic.py --base_dir=$base_data --scene_id=$scene
    cd $base_data/$scene 
    rm -rf depth 
    mv new_depth depth
done 
