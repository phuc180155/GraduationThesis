######
export train_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/3dmm/train"
export val_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/3dmm/val"
export test_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/3dmm/test"
for C in 2.0
do
    export checkpoint="checkpoint/3dmm/headpose"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
    export checkpoint="checkpoint/3dmm/spectrum"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
done

######
export train_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/deepfake/train"
export val_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/deepfake/val"
export test_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/deepfake/test"
for C in 2.0
do
    export checkpoint="checkpoint/deepfake/headpose"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
    export checkpoint="checkpoint/deepfake/spectrum"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
done

######
export train_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/faceswap_2d/train"
export val_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/faceswap_2d/val"
export test_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/faceswap_2d/test"
for C in 2.0
do
    export checkpoint="checkpoint/faceswap_2d/headpose"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
    export checkpoint="checkpoint/faceswap_2d/spectrum"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
done

######
export train_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/faceswap_3d/train"
export val_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/faceswap_3d/val"
export test_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/faceswap_3d/test"
for C in 2.0
do
    export checkpoint="checkpoint/faceswap_3d/headpose"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
    export checkpoint="checkpoint/faceswap_3d/spectrum"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
done

######
export train_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/monkey/train"
export val_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/monkey/val"
export test_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/monkey/test"
for C in 2.0
do
    export checkpoint="checkpoint/monkey/headpose"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
    export checkpoint="checkpoint/monkey/spectrum"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
done

######
export train_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/reenact/train"
export val_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/reenact/val"
export test_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/reenact/test"
for C in 2.0
do
    export checkpoint="checkpoint/reenact/headpose"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
    export checkpoint="checkpoint/reenact/spectrum"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
done

######
export train_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/stargan/train"
export val_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/stargan/val"
export test_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/stargan/test"
for C in 2.0
do
    export checkpoint="checkpoint/stargan/headpose"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
    export checkpoint="checkpoint/stargan/spectrum"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
done

######
export train_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/x2face/train"
export val_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/x2face/val"
export test_dir="/mnt/disk1/doan/phucnp/Dataset/my_extend_data/extend_data_train/x2face/test"
for C in 2.0
do
    export checkpoint="checkpoint/x2face/headpose"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint headpose_forensic
    export checkpoint="checkpoint/x2face/spectrum"
    python args_train.py --train_dir $train_dir --val_dir $val_dir --test_dir $test_dir --C $C --checkpoint $checkpoint spectrum
done