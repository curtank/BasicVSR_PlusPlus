# INPUT_PATH="/mnt/weka/scratch/qingyu.sui/sample/Friends"
# OUTPUT_PATH="/mnt/weka/scratch/zheyu.zhang/zheyu_hrv/basicvsr_plusplus/Friends"
# CONFIG_PATH="configs/basicvsr_plusplus_reds4.py"
# CHECKPOINT_PATH="chkpts/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"

# python3 demo/restoration_video_demo_2.py --max-seq-len=5  $CONFIG_PATH $CHECKPOINT_PATH $INPUT_PATH $OUTPUT_PATH
from multiprocessing import Pool
import os
import subprocess
def run_sample(device_num, sample_name, input_path):
    # get_pool_index

    config_path = "configs/basicvsr_plusplus_reds4.py"
    checkpoint_path = "chkpts/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"
    input_path_full = os.path.join(input_path, sample_name)
    output_path = f"/mnt/weka/scratch/zheyu.zhang/zheyu_hrv/basicvsr_plusplus/{sample_name}"
    cmd = f"python3 demo/restoration_video_demo_2.py --max-seq-len=5 --device={device_num} {config_path} {checkpoint_path} {input_path_full} {output_path}"

    print(device_num, sample_name)
    
    subprocess.run(cmd, shell=True)
    print(f"finished {sample_name}")
def run_multi_sample(device_num, sample_list, input_path):
    print(device_num, sample_list)
    for sample_name in sample_list:
        run_sample(device_num, sample_name, input_path)
    print(f"finished {sample_list}")
def main():
    device_num = [0,1,2]
    # device_num = [0]
    sample_dist = '/mnt/weka/scratch/qingyu.sui/sample'
    sample_list = [
        x for x in os.listdir(sample_dist) if os.path.isdir(os.path.join(sample_dist, x))
    ]
    print(sample_list)

    pool = Pool(len(device_num))
    for i in range(len(device_num)):
        pool.apply_async(run_multi_sample, args=(device_num[i], sample_list[i::len(device_num)], sample_dist))
    pool.close()
    pool.join()
    print('All subprocesses done.')
    

if __name__ == '__main__':
    main()