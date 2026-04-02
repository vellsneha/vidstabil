<!-- mkdir -p data/test_clip/images
ffmpeg -i crowd9.mp4 -frames:v 100 -q:v 1 data/test_clip/images/%05d.png
ls data2/test_clip/images/ | wc -l -->

<!-- python prepare_dataset.py --src-frames "/workspace/vidstabil/data/test_clip/images" --scene "/workspace/vidstabil/data/crowd9_scene" --gen-depth --depth-model v2 -->

<!-- python preprocess_dynamic_masks.py -s "/workspace/vidstabil/data/crowd9_scene" --backend gsam2 --text-prompt "person ." -->

<!-- python gen_tracks.py --image_dir "/workspace/vidstabil/data/crowd9_scene/images_2" --mask_dir "/workspace/vidstabil/data/crowd9_scene/dynamic_masks" --out_dir "/workspace/vidstabil/data/crowd9_scene/bootscotracker_dynamic" --grid_size 64 --max_hw 512 -->

<!-- python gen_tracks.py --image_dir "/workspace/vidstabil/data/crowd9_scene/images_2" --mask_dir  "/workspace/vidstabil/data/crowd9_scene/dynamic_masks" --out_dir "/workspace/vidstabil/data/crowd9_scene/bootscotracker_static" --is_static --grid_size 32 -->

<!-- python train_entrypoint.py -s "/workspace/vidstabil/data/crowd9_scene" --expname my_run_tracks --use_dynamic_mask --iterations 5000 -->

<!-- python render_stabilized.py -s "/workspace/vidstabil/data/crowd9_scene" --expname my_run_tracks --trajectory_scale 0.25 --output_video "/workspace/vidstabil/output/my_run_tracks/stabilized.mp4" -->