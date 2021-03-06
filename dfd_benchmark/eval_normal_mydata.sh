python eval.py --val_set /hdd/tam/dfd_benmark/extend_data_train/deepfake/test --batch_size 32 --image_size 256 --workers 16 --checkpoint wavelet_deepfake_checkpoint/  --resume model_best.pt --gpu_id 0 --adj_contrast 1.0 wavelet
python eval.py --val_set /hdd/tam/dfd_benmark/extend_data_train/3dmm/test --batch_size 32 --image_size 256 --workers 16 --checkpoint wavelet_3dmm_checkpoint/ --resume model_best.pt --gpu_id 0 --adj_contrast 1.0 wavelet
python eval.py --val_set /hdd/tam/dfd_benmark/extend_data_train/faceswap_2d/test --batch_size 32  --image_size 256 --workers 16 --checkpoint wavelet_faceswap_2d_checkpoint/  --resume model_best.pt --gpu_id 0 --adj_contrast 1.0  wavelet
python eval.py --val_set /hdd/tam/dfd_benmark/extend_data_train/faceswap_3d/test --batch_size 32 --image_size 256 --workers 16 --checkpoint wavelet_faceswap_3d_checkpoint/  --resume model_best.pt --gpu_id 0 --adj_contrast 1.0 wavelet
python eval.py --val_set /hdd/tam/dfd_benmark/extend_data_train/monkey/test --batch_size 32  --image_size 256 --workers 16 --checkpoint wavelet_monkey_checkpoint/ --gpu_id 0 --resume model_best.pt --adj_contrast 1.0  wavelet
python eval.py --val_set /hdd/tam/dfd_benmark/extend_data_train/reenact/test --batch_size 32  --image_size 256 --workers 16 --checkpoint wavelet_reenact_checkpoint/ --gpu_id 0  --resume model_best.pt  --adj_contrast 1.0  wavelet
python eval.py --val_set /hdd/tam/dfd_benmark/extend_data_train/stargan/test --batch_size 32  --image_size 256 --workers 16 --checkpoint wavelet_stargan_checkpoint/ --gpu_id 0  --resume model_best.pt  --adj_contrast 1.0  wavelet
python eval.py --val_set /hdd/tam/dfd_benmark/extend_data_train/x2face/test --batch_size 32  --image_size 256 --workers 16 --checkpoint wavelet_x2face_checkpoint/ --gpu_id 0  --resume model_best.pt  --adj_contrast 1.0  wavelet


#python eval.py --val_set /hdd/tam/dfd_benmark/extend_data_train/deepfake/test --batch_size 1 --image_size 256 --workers 1 --checkpoint wavelet_deepfake_checkpoint/  --resume model_best.pt --gpu_id 0 --adj_contrast 1.0 wavelet
