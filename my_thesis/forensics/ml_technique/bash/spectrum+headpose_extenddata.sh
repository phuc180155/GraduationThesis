# ######
# export train_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/deepfake/train"
# export val_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/deepfake/val"

# for C in 2.0
# do
#     export test_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/3dmm/test"
#     export checkpoint="checkpoint/cross_dataset/deepfake/3dmm/headpose"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
#     export checkpoint="checkpoint/cross_dataset/deepfake/3dmm/spectrum"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
# done

# for C in 2.0
# do
#     export test_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/faceswap_2d/test"
#     export checkpoint="checkpoint/cross_dataset/deepfake/faceswap_2d/headpose"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
#     export checkpoint="checkpoint/cross_dataset/deepfake/faceswap_2d/spectrum"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
# done

# for C in 2.0
# do
#     export test_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/faceswap_3d/test"
#     export checkpoint="checkpoint/cross_dataset/deepfake/faceswap_3d/headpose"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
#     export checkpoint="checkpoint/cross_dataset/deepfake/faceswap_3d/spectrum"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
# done

# for C in 2.0
# do
#     export test_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/monkey/test"
#     export checkpoint="checkpoint/cross_dataset/deepfake/monkey/headpose"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
#     export checkpoint="checkpoint/cross_dataset/deepfake/monkey/spectrum"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
# done

# for C in 2.0
# do
#     export test_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/reenact/test"
#     export checkpoint="checkpoint/cross_dataset/deepfake/reenact/headpose"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
#     export checkpoint="checkpoint/cross_dataset/deepfake/reenact/spectrum"
#     python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
# done

for C in 2.0
do
    export train_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/stargan/train"
    export val_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/stargan/val"
    export test_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/stargan/test"
    # export checkpoint="checkpoint/cross_dataset/deepfake/stargan/headpose"
    # python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
    export checkpoint="checkpoint/cross_dataset/deepfake/stargan/spectrum"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
done

for C in 2.0
do
    export train_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/x2face/train"
    export val_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/x2face/val"
    export test_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/x2face/test"
    # export checkpoint="checkpoint/cross_dataset/deepfake/x2face/headpose"
    # python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
    export checkpoint="checkpoint/cross_dataset/deepfake/x2face/spectrum"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
done