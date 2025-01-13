# Converting the Model
To convert an OmChat model to Hugging Face (HF) format, use the convert.py script.
```
python convert_main.py --model_name <path_to_original_model> --output_folder <path_to_output_directory>
```

## Example
```
python convert_main.py --model_name /data2/omchat_dev/omchat/checkpoints/omchat-qwen2-7b-siglip-multi-fk104_inifinity_stage4 \
                  --output_folder checkpoints/omchat_temp_multi_hf
```

This command will save the converted model and processor in the specified output_folder.

## Testing the Model
To test the converted OmChat model, use the test.py script.

```
python test.py --model_name <path_to_converted_model>
```

