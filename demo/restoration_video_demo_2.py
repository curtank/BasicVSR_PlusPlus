# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import cv2
# import tqdm
from tqdm import tqdm
import mmcv
import numpy as np
import torch

from mmedit.apis import init_model
from mmedit.core import tensor2img
from mmedit.utils import modify_args

VIDEO_EXTENSIONS = ('.mp4', '.mov')


def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_dir', help='directory of the input video')
    parser.add_argument('output_dir', help='directory of the output video')
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='index corresponds to the first frame of the sequence')
    parser.add_argument(
        '--filename-tmpl',
        default='{:08d}.png',
        help='template of the file names')
    parser.add_argument(
        '--window-size',
        type=int,
        default=0,
        help='window size if sliding-window framework is used')
    parser.add_argument(
        '--max-seq-len',
        type=int,
        default=None,
        help='maximum sequence length if recurrent framework is used')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def init_writer(result, path):
    h, w = result.shape[-2:]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, 24, (w, h))

def run(model,src_file,dst_file,device):
    reader = mmcv.VideoReader(src_file)
    writer = None
    images = [
        np.flip(frame, axis=2)
        for frame in reader
    ]
    inputs = [
        torch.from_numpy(image / 255.0).permute(2, 0, 1).float().unsqueeze(0)
        for image in images
    ]
    stack = torch.stack(inputs, dim=1)

    with torch.no_grad():
        batch_size = 25
        for i in tqdm(range(0, stack.size(1), batch_size)):
            data = stack[:, i:i+batch_size, :, :, :].to(device)
            result = model(data, test_mode=True)["output"].cpu()
            if writer is None:
                writer = init_writer(result, dst_file)
            for j in range(0, result.size(1)):
                writer.write(tensor2img(result[:, j, :, :, :]).astype(np.uint8))

    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

def run_to_img(model,src_file,output_dir,filename_tmpl,device):
    reader = mmcv.VideoReader(src_file)
    writer = None
    images = [
        np.flip(frame, axis=2)
        for frame in reader
    ]
    inputs = [
        torch.from_numpy(image / 255.0).permute(2, 0, 1).float().unsqueeze(0)
        for image in images
    ]
    stack = torch.stack(inputs, dim=1)

    with torch.no_grad():
        batch_size = 25
        for i in tqdm(range(0, stack.size(1), batch_size)):
            data = stack[:, i:i+batch_size, :, :, :].to(device)
            result = model(data, test_mode=True)["output"].cpu()
            for j in range(0, result.size(1)):
                output_j = tensor2img(result[:, j, :, :, :])
                save_path_j = f'{output_dir}/{filename_tmpl.format(i*batch_size+j)}'
                mmcv.imwrite( output_j, save_path_j)

    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

def img2img(model,input_dir,output_dir,filename_tmpl,device):
    images = []
    image_names = []
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.png'):
            img = mmcv.imread(os.path.join(input_dir, filename))
            img = np.flip(img, axis=2)
            images.append(img)
            image_names.append(filename)
            print(filename)
    inputs = [
        torch.from_numpy(image / 255.0).permute(2, 0, 1).float().unsqueeze(0)
        for image in images
    ]
    stack = torch.stack(inputs, dim=1)
    image_name_index = 0
    with torch.no_grad():
        batch_size = 5
        for i in tqdm(range(0, stack.size(1), batch_size)):
            data = stack[:, i:i+batch_size, :, :, :].to(device)
            result = model(data, test_mode=True)["output"].cpu()
            for j in range(0, result.size(1)):
                output_j = tensor2img(result[:, j, :, :, :])
                save_path_j = f'{output_dir}/{image_names[image_name_index]}'
                image_name_index += 1
                mmcv.imwrite( output_j, save_path_j)

    cv2.destroyAllWindows()
def main():
    """ Demo for video restoration models.

    Note that we accept video as input/output, when 'input_dir'/'output_dir'
    is set to the path to the video. But using videos introduces video
    compression, which lowers the visual quality. If you want actual quality,
    please save them as separate images (.png).
    """

    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))
    img2img(model,args.input_dir,args.output_dir,args.filename_tmpl,args.device)

    # if os.path.splitext(args.output_dir)[1] in VIDEO_EXTENSIONS:
    #     run(model,args.input_dir,args.output_dir, args.device)
    # else:
    #     run_to_img(model,args.input_dir,args.output_dir,args.filename_tmpl,args.device)

if __name__ == '__main__':
    main()
