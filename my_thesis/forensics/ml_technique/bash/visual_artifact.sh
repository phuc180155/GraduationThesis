
######
export train_dir="/mnt/disk1/doan/phucnp/Dataset/dfdcv5/image/train"
export val_dir="/mnt/disk1/doan/phucnp/Dataset/dfdcv5/image/val"
export test_dir="/mnt/disk1/doan/phucnp/Dataset/dfdcv5/image/test"
export checkpoint="checkpoint/dfdcv5/spectrum"
for C in 2.0
do
    export checkpoint="checkpoint/dfdcv5/visual_artifact"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint visual_artifact
done