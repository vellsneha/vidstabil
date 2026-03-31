python gen_tracks.py   --image_dir "/workspace/vidstabil/data/crowd9_scene/images_2"   --mask_dir  "/workspace/vidstabil/data/crowd9_scene/dynamic_masks"   --out_dir   "/workspace/vidstabil/data/crowd9_scene/bootscotracker_static"   --is_static   --grid_size 32

python train_entrypoint.py \
  --legacy-dynamic \
  -s "/workspace/vidstabil/data/crowd9_scene" \
  --expname my_run_tracks \
  --use_dynamic_mask \
  --iterations 10000

python render_stabilized.py \
  -s "/workspace/vidstabil/data/crowd9_scene" \
  --expname my_run_tracks \
  --output_video "/workspace/vidstabil/output/my_run_tracks/stabilized.mp4"