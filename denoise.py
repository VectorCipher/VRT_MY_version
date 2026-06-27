import argparse
import cv2
import glob
import os
import shutil
import torch
import numpy as np
from os import path as osp
from collections import OrderedDict
from torch.utils.data import DataLoader

from models.network_vrt import VRT as net
from utils import utils_image as util
from data.dataset_video_test import SingleVideoRecurrentTestDataset

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    print(f"Extracting frames from {video_path}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(output_dir, f"{frame_idx:05d}.png")
        cv2.imwrite(out_path, frame)
        frame_idx += 1
    cap.release()
    print(f"Extracted {frame_idx} frames.")
    return frame_idx

def stitch_frames(input_dir, output_video, fps=30.0):
    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not frames:
        print(f"No frames found in {input_dir}")
        return
    
    first_frame = cv2.imread(frames[0])
    height, width, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"Stitching {len(frames)} frames to {output_video}...")
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        out.write(frame)
        
    out.release()
    print(f"Video saved successfully to {output_video}")

def test_clip(lq, model, args):
    sf = args.scale
    window_size = args.window_size
    size_patch_testing = args.tile[1]
    
    if size_patch_testing:
        overlap_size = args.tile_overlap[1]
        not_overlap_border = True

        b, d, c, h, w = lq.size()
        c = c - 1 if args.nonblind_denoising else c
        stride = size_patch_testing - overlap_size
        h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
        w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
        E = torch.zeros(b, d, c, h*sf, w*sf)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                out_patch = model(in_patch).detach().cpu()
                out_patch_mask = torch.ones_like(out_patch)

                if not_overlap_border:
                    if h_idx < h_idx_list[-1]:
                        out_patch[..., -overlap_size//2:, :] *= 0
                        out_patch_mask[..., -overlap_size//2:, :] *= 0
                    if w_idx < w_idx_list[-1]:
                        out_patch[..., :, -overlap_size//2:] *= 0
                        out_patch_mask[..., :, -overlap_size//2:] *= 0
                    if h_idx > h_idx_list[0]:
                        out_patch[..., :overlap_size//2, :] *= 0
                        out_patch_mask[..., :overlap_size//2, :] *= 0
                    if w_idx > w_idx_list[0]:
                        out_patch[..., :, :overlap_size//2] *= 0
                        out_patch_mask[..., :, :overlap_size//2] *= 0

                E[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch_mask)
        output = E.div_(W)
    else:
        _, _, _, h_old, w_old = lq.size()
        h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
        w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

        lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
        lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq
        output = model(lq).detach().cpu()
        output = output[:, :, :, :h_old*sf, :w_old*sf]

    return output

def test_video(lq, model, args):
    num_frame_testing = args.tile[0]
    if num_frame_testing:
        sf = args.scale
        num_frame_overlapping = args.tile_overlap[0]
        not_overlap_border = False
        b, d, c, h, w = lq.size()
        c = c - 1 if args.nonblind_denoising else c
        stride = num_frame_testing - num_frame_overlapping
        d_idx_list = list(range(0, d-num_frame_testing, stride)) + [max(0, d-num_frame_testing)]
        E = torch.zeros(b, d, c, h*sf, w*sf)
        W = torch.zeros(b, d, 1, 1, 1)

        for d_idx in d_idx_list:
            lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]
            out_clip = test_clip(lq_clip, model, args)
            out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

            if not_overlap_border:
                if d_idx < d_idx_list[-1]:
                    out_clip[:, -num_frame_overlapping//2:, ...] *= 0
                    out_clip_mask[:, -num_frame_overlapping//2:, ...] *= 0
                if d_idx > d_idx_list[0]:
                    out_clip[:, :num_frame_overlapping//2, ...] *= 0
                    out_clip_mask[:, :num_frame_overlapping//2, ...] *= 0

            E[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip)
            W[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip_mask)
        output = E.div_(W)
    else:
        window_size = args.window_size
        d_old = lq.size(1)
        d_pad = (window_size[0] - d_old % window_size[0]) % window_size[0]
        lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1) if d_pad else lq
        output = test_clip(lq, model, args)
        output = output[:, :d_old, :, :, :]

    return output

def main():
    parser = argparse.ArgumentParser(description="End-to-End VRT Denoising")
    parser.add_argument('--input', type=str, required=True, help='Input MP4 video or folder of frames')
    parser.add_argument('--output', type=str, required=True, help='Output MP4 video path')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .pth model file')
    parser.add_argument('--sigma', type=int, default=20, help='Noise level for denoising (e.g., 20)')
    parser.add_argument('--tile', type=int, nargs='+', default=[6, 128, 128], help='Tile size (t, h, w)')
    parser.add_argument('--tile_overlap', type=int, nargs='+', default=[2, 20, 20], help='Tile overlap (t, h, w)')
    args = parser.parse_args()

    # VRT specific arguments (Hardcoded for 008_VRT_videodenoising_DAVIS)
    args.scale = 1
    args.window_size = [6, 8, 8]
    args.nonblind_denoising = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize Model
    print("Initializing model...")
    model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                indep_reconsts=[9,10], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2, deformable_groups=16,
                nonblind_denoising=True)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    
    pretrained_model = torch.load(args.model_path, map_location=device)
    model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model.keys() else pretrained_model, strict=True)
    model.eval()
    model = model.to(device)

    # 2. Handle Input Video Extraction
    is_video_input = os.path.isfile(args.input) and args.input.endswith(('.mp4', '.avi', '.mov'))
    tmp_in_dir = "tmp_vrt_frames/in/000"
    tmp_out_dir = "tmp_vrt_frames/out"
    
    if is_video_input:
        if os.path.exists("tmp_vrt_frames"):
            shutil.rmtree("tmp_vrt_frames")
        os.makedirs(tmp_in_dir)
        extract_frames(args.input, tmp_in_dir)
        folder_lq = "tmp_vrt_frames/in"
    else:
        folder_lq = args.input

    # 3. Setup Dataset
    test_set = SingleVideoRecurrentTestDataset({
        'dataroot_gt': None, 
        'dataroot_lq': folder_lq,
        'sigma': args.sigma, 
        'num_frame': -1, 
        'cache_data': False
    })
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
    
    # 4. Inference
    os.makedirs(tmp_out_dir, exist_ok=True)
    
    for idx, batch in enumerate(test_loader):
        lq = batch['L'].to(device)
        folder = batch['folder']
        
        with torch.no_grad():
            print(f"Running inference on {folder[0]}...")
            output = test_video(lq, model, args)
            
        for i in range(output.shape[1]):
            img = output[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if img.ndim == 3:
                img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
            img = (img * 255.0).round().astype(np.uint8)
            seq_ = osp.basename(batch['lq_path'][i][0]).split('.')[0]
            cv2.imwrite(f'{tmp_out_dir}/{seq_}.png', img)

    # 5. Stitch Video
    if args.output.endswith(('.mp4', '.avi', '.mov')):
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        stitch_frames(tmp_out_dir, args.output)
    else:
        if os.path.exists(args.output):
            shutil.rmtree(args.output)
        shutil.copytree(tmp_out_dir, args.output)
        
    # 6. Cleanup
    if os.path.exists("tmp_vrt_frames"):
        shutil.rmtree("tmp_vrt_frames")
        print("Cleaned up temporary frames.")

if __name__ == '__main__':
    main()
