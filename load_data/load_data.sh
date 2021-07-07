scannet_code=/home/shenguanlin/ScanNet
base_code=/home/shenguanlin/ScanNetAnnotation/load_data
base_source=/data/sgl/scannet/scans
base_target=/home/shenguanlin/scannet_mine

mkdir $base_target
cd $base_target


sudo chmod -R 777 $base_source

cd $base_source
files=$(ls $folder)
for file in $files
do
    mkdir $base_target/$file 
    mkdir $base_target/$file/ply

    ply_name_full="_vh_clean.ply"
    sens_name=".sens"
    instance_name="_2d-instance-filt.zip"
    label_name="_2d-label-filt.zip"
    ply_name="_vh_clean_2.ply"
    seg_name="_vh_clean_2.labels.ply"
    aggregation_name="_vh_clean.aggregation.json"
    json_name="_vh_clean_2.0.010000.segs.json"

    cd $scannet_code/SensReader/c++
    ./sens $base_source/$file/$file$sens_name $base_target/$file
    sudo unzip $base_source/$file/$file$instance_name -d $base_target/$file
    sudo unzip $base_source/$file/$file$label_name -d $base_target/$file
    cp $base_source/$file/$file$ply_name $base_target/$file/ply/$file$ply_name
    cp $base_source/$file/$file$seg_name $base_target/$file/ply/$file$seg_name
    cp $base_source/$file/$file$aggregation_name $base_target/$file/ply/$file$aggregation_name
    cp $base_source/$file/$file$json_name $base_target/$file/ply/$file$json_name
    cp $base_source/$file/$file$ply_name_full $base_target/$file/ply/$file$ply_name_full

    sudo chmod -R 777 $base_target
    cd $base_code
    python data_selection.py --base_dir=$base_target --scene_id=$file
done

