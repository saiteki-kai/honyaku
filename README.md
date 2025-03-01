# Response Generation

The response generation script can be used to generate responses for a dataset using an Hugging Face model. The generated responses are saved in a Parquet file named `./outputs/<dataset_name>/responses/<model_name>/<split>.parquet`.

```bash
python generate.py \
    --model-name "Qwen/Qwen2.5-72B-Instruct" \
    --dataset-name "PKU-Alignment/BeaverTails-Evaluation" \
    --split "test" \
    --field "prompt" \
    --dtype "bfloat16" \
    --max-length 2048 \
    --output-path "./outputs/" \
    --log-file "./logs/responses/Qwen__Qwen2.5-72B-Instruct.log"
```
