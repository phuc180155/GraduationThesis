#######
# export train_dir="/mnt/disk1/doan/phucnp/Dataset/UADFV/image/train"
# export val_dir="/mnt/disk1/doan/phucnp/Dataset/UADFV/image/val"
# export test_dir="/mnt/disk1/doan/phucnp/Dataset/UADFV/image/test"
# for C in 2.0 3.0 4.0 5.0
# do
#     export checkpoint="checkpoint/UADFV/spectrum"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
#     export checkpoint="checkpoint/UADFV/headpose"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic

# done

# ######
# export train_dir="/mnt/disk1/doan/phucnp/Dataset/df_in_the_wildv6/image/train"
# export val_dir="/mnt/disk1/doan/phucnp/Dataset/df_in_the_wildv6/image/val"
# export test_dir="/mnt/disk1/doan/phucnp/Dataset/df_in_the_wildv6/image/test"
# export checkpoint="checkpoint/df_in_the_wildv6/spectrum"
# for C in 2.0 3.0 4.0 5.0
# do
#     export checkpoint="checkpoint/df_in_the_wildv6/spectrum"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
#     export checkpoint="checkpoint/df_in_the_wildv6/headpose"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic

# done

# ######
# export train_dir="/mnt/disk1/doan/phucnp/Dataset/Celeb-DFv6/image/train"
# export val_dir="/mnt/disk1/doan/phucnp/Dataset/Celeb-DFv6/image/val"
# export test_dir="/mnt/disk1/doan/phucnp/Dataset/Celeb-DFv6/image/test"
# export checkpoint="checkpoint/Celeb-DFv6/spectrum"
# for C in 2.0 3.0 4.0 5.0
# do
#     export checkpoint="checkpoint/Celeb-DFv6/spectrum"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
#     export checkpoint="checkpoint/Celeb-DFv6/headpose"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
# done

######
export train_dir="/mnt/disk1/doan/phucnp/Dataset/dfdcv5/image/train"
export val_dir="/mnt/disk1/doan/phucnp/Dataset/dfdcv5/image/val"
export test_dir="/mnt/disk1/doan/phucnp/Dataset/dfdcv5/image/test"
export checkpoint="checkpoint/dfdcv5/spectrum"
for C in 2.0 3.0 4.0 5.0
do
    export checkpoint="checkpoint/dfdcv5/headpose"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
    export checkpoint="checkpoint/dfdcv5/spectrum"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
done