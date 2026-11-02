# Downloads to delete after the CVPR 2027 submission

Downloaded 2026-11-02 for local streaming experiments (user cap: 10 GB).
Delete after submitting (deadline 2026-11-16):

| Path (under vdaquant/) | Source | Zip size |
|---|---|---|
| benchmark_data/sintel/ | MPI-Sintel-depth-training-20150305.zip + MPI-Sintel-training_images.zip (files.is.tue.mpg.de) | 1.60 + 1.78 GB |
| benchmark_data/bonn/ | rgbd_bonn_{balloon,crowd,kidnapping_box}.zip (ipb.uni-bonn.de) | 1.38 GB |
| checkpoints/video_depth_anything_vitb.pth | huggingface.co/depth-anything/Video-Depth-Anything-Base | 0.46 GB |
| checkpoints/video_depth_anything_vitl.pth | huggingface.co/depth-anything/Video-Depth-Anything-Large | 1.54 GB |

Keep: checkpoints/video_depth_anything_vits.pth (present before).

Cleanup command (run from vdaquant/):

    rm -rf benchmark_data/sintel benchmark_data/bonn checkpoints/video_depth_anything_vitb.pth checkpoints/video_depth_anything_vitl.pth
