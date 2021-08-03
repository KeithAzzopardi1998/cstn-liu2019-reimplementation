export HDF5_DISABLE_VERSION_CHECK=1
python3 train.py --lr 0.001 \
                --batch_size 32 \
                --seq_len 5 \
                --num_days_test 60 \
                --dataset_path ./od_matrix_100pc_10x3.npy \
                --out_path ./testing_python3
