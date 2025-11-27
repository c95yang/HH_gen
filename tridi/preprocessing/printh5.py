import h5py

with h5py.File("data/preprocessed/behave_smplh/dataset_test_1fps.hdf5", "r") as f:
    data = f["Sub05_test/Default/second_sbj_smpl_body"][:]  # 读取数据
    print(data.shape)
    print(data[:5]) 