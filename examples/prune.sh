prune_gal() {
    poetry run python prune_30.py facebook/galactica-1.3b \
        --wandb_project new-method-compare --focus pile_codeless \
        --cripple code --run_pre_test True --svd_attn False \
        --attn_mode pre-out --token_limit 1000 --name "$@"
}
prune_opt() {
    poetry run python prune_30.py facebook/opt-1.3b \
        --wandb_project new-method-compare --focus pile_codeless \
        --cripple code --run_pre_test True --svd_attn False \
        --attn_mode pre-out --token_limit 1000 --name "$@"
}
prune_pyt() {
    poetry run python prune_30.py EleutherAI/pythia-1.4b \
        --wandb_project new-method-compare --focus pile_codeless \
        --cripple code --run_pre_test True --svd_attn False \
        --attn_mode pre-out --token_limit 1000 --name "$@"
}

prune_roberta() {
    poetry run python prune_30.py roberta-large \
        --wandb_project new-method-compare --focus pile_codeless \
        --cripple code --run_pre_test True --svd_attn False \ 
	--attn_mode pre-out --token_limit 512 --name "$@"
}

prune_vit() {
    #poetry run python /root/separability/examples/prune_30.py google/vit-base-patch16-224 \
    poetry run python /root/separability/examples/prune_30.py google/vit-base-patch16-224 \
	--focus imagenet-1k-birdless --cripple imagenet-1k-birds \
        --dtype fp32 \
	--eval_sample_size 1000 --collection_sample_size 100000 \
        --wandb_project birds \
	--recalculate_activations false \
	--name "$@"
}

prune_rocket() {
    # ViT CIFAR100
    poetry run python /root/separability/examples/prune_30.py Ahmed9275/Vit-Cifar100 \
	--focus cifar100-rocketless --cripple cifar100-rocket \
        --dtype fp32 \
	--eval_sample_size 1000 --collection_sample_size 100000 \
	--additional_datasets cifar100-rocket-mia \
        --wandb_project rockets \
	--recalculate_activations false \
	--name "$@" --n_steps 10
}

prune_mushrooms() {
    # ViT CIFAR100
    poetry run python /root/separability/examples/prune_30.py Ahmed9275/Vit-Cifar100 \
	--focus cifar100-mushroomless --cripple cifar100-mushroom \
        --dtype fp32 \
	--eval_sample_size 1000 --collection_sample_size 100000 \
	--additional_datasets cifar100-mushroom-mia \
        --wandb_project mushrooms \
	--recalculate_activations false \
	--name "$@"
}

# prune_vit "vit l 2% 2% noniter" --attn_frac 0.02 --ff_frac 0.02
# prune_vit "vit l 2% 2% random"  --attn_frac 0.02 --ff_frac 0.02 --attn_scoring random --ff_scoring random
prune_mushrooms "vit b 3% 3% noniter" --attn_frac 0.03 --ff_frac 0.03 --n_steps 1
#prune_rocket "vit b 1% 1% noniter" --attn_frac 0.01 --ff_frac 0.01
