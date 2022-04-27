

# ego-view models
MODEL_NAMES=(
	6d3bfc13-1da4-49f0-bf6d-f6b1fc9647d8
	9dcfaa3f-a6af-4295-93ca-24a84d6b9c2d
	98e50a71-c7b2-411d-99f4-781826488a26
	2a3550a4-7b3d-4ab1-8165-e20d7cb069c9
	0061c32d-da98-4583-a311-8f2fc37b6655
	44e55ee6-76da-4995-8fff-f4a2a4c3a8af
	4d4f41a2-4bfe-42f2-88d0-1db253eeb9be
	17fd2c0a-fee5-47c5-92cc-8b37f4479a8b
	b3ef41e8-db72-4e12-808d-353e4cd54280
	88f42746-3374-40f7-a015-625652ca62c8
	fe460247-d73f-4519-8643-ff38f95fb3b7
	e3411e4e-87a6-4539-9ebb-1fcd6f99e601
	36b24988-5c54-46b4-9c22-cd48f70ae9f6
	9b170dcf-6ba8-41d0-9ff2-fc3faf92e514
	b5a2198c-5368-4feb-85c5-843a5646ecfa
	615683e4-8431-4b31-ba1d-3523e6165fa9
	)

SPLITS=(
	val
	test
	)

now=$(date +"%Y%m%d_%H%M%S")

for MODEL_NAME in ${MODEL_NAMES[@]}; do

	for SPLIT in ${SPLITS[@]}; do
		python -u scripts/test.py --rendering_config_name ${MODEL_NAME}.yaml \
		                       --training_config_name ${MODEL_NAME}.yaml \
		                       --ckpt_fpath /Users/johnlambert/Downloads/${MODEL_NAME}.pth \
		                       --num_workers 1 \
		                       --split ${SPLIT} \
		                       --gpu_ids -1 \
		                       --filter_eval_by_visibility True \
		                       2>&1 | tee eval_logs/${MODEL_NAME}-${SPLIT}-$now.log

	done
done