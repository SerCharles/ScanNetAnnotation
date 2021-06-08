
base_code=/home/shenguanlin/pyrender
base_data=/data/sgl/geolayout_pretrain

cd $base_data/data_list 

confs=$(ls *)
cd $base_code
for conf in $confs 
do 
    python main.py --base_dir=$base_data --conf_name=$conf
done 