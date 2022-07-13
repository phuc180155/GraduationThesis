######
# export train_dir="/mnt/disk1/doan/phucnp/Dataset/UADFV/image/train"
# export val_dir="/mnt/disk1/doan/phucnp/Dataset/UADFV/image/val"
# export test_dir="/mnt/disk1/doan/phucnp/Dataset/UADFV/image/test"

# for C in 2.0
# do
#     export checkpoint="checkpoint/UADFV/visual_artifact"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint visual_artifact
# done


######
export train_dir="/mnt/disk1/doan/phucnp/Dataset/df_in_the_wildv6/image/train"
export val_dir="/mnt/disk1/doan/phucnp/Dataset/df_in_the_wildv6/image/val"
export test_dir="/mnt/disk1/doan/phucnp/Dataset/df_in_the_wildv6/image/test"
export checkpoint="checkpoint/df_in_the_wildv6/spectrum"
for C in 2.0
do
    export checkpoint="checkpoint/df_in_the_wildv6/visual_artifact"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint visual_artifact
done

######
export train_dir="/mnt/disk1/doan/phucnp/Dataset/Celeb-DFv6/image/train"
export val_dir="/mnt/disk1/doan/phucnp/Dataset/Celeb-DFv6/image/val"
export test_dir="/mnt/disk1/doan/phucnp/Dataset/Celeb-DFv6/image/test"
export checkpoint="checkpoint/Celeb-DFv6/spectrum"
for C in 2.0
do
    export checkpoint="checkpoint/Celeb-DFv6/visual_artifact"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint visual_artifact
done
    