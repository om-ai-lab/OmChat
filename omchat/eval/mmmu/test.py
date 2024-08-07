import yaml



def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict

config = load_yaml("/data3/ljj/proj/MLLM_evals/scripts/MMMU/eval/configs/llava1.5.yaml")
for key, value in config.items():
    if key != 'eval_params' and type(value) == list:
        assert len(value) == 1, 'key {} has more than one value'.format(key)
        config[key] = value[0]
import pdb;pdb.set_trace()