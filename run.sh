deepspeed --module openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset Qingyisi/Alpaca-CoT@firefly \
   --input_key instruction \
   --output_key output \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain Qwen/Qwen2-0.5B \
   --save_path ./checkpoint/qwen2-0.5b-firefly-sft \
   --save_steps 20 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --packing_samples \
   --load_checkpoint \
   --use_wandb 4feae54effa30a16590278a5ae843b3a3ca69419

# 支持 HF tokenizer.apply_chat_template
# --apply_chat_template 
# --tokenizer_chat_template {HF Chat Template}

# 支持 RingAttention
# pip install ring_flash_attn
#   --ring_attn_size 2 \
#   --ring_head_stride 2 \

# 也可用于 continued pre-training
# --pretrain_mode