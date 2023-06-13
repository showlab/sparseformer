train-sparseformer_t:
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
	python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py \
	--cfg configs/sparseformer_t.yaml \
	--data-path /path/to/imagenet \
	--amp-opt-level O0 \
	--cache-mode no \
	--output output

train-sparseformer_s:
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
	python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py \
	--cfg configs/sparseformer_s.yaml \
	--data-path /path/to/imagenet \
	--amp-opt-level O0 \
	--cache-mode no \
	--output output

train-sparseformer_b:
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
	python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py \
	--cfg configs/sparseformer_b.yaml \
	--data-path /path/to/imagenet \
	--amp-opt-level O0 \
	--cache-mode no \
	--output output

