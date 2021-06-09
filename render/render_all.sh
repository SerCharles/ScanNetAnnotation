
base_code=/home/shenguanlin/pyrender
base_data=/data/sgl/geolayout_pretrain

cd $base_data

scenes=$(ls *)
cd $base_code
for scene in $scenes
do 
    python main.py --base_dir=$base_data --scene_id=$scene
done 