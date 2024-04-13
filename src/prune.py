from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import json
import copy
from torch import nn
import tensorboardX


def test():
    with open('./configs/config.json','r') as f:
        config = json.load(f)
    f.close()
    
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    model = AutoModelForCausalLM.from_pretrained(config['model'])

    dataset_map = {
        "mmlu":"cais/mmlu",
        "boolq":"google/boolq"
    }

    dataset = load_dataset(dataset_map[config['dataset']], split='validation')

    train_args = TrainingArguments(
        output_dir='./models/',
        overwrite_output_dir=False,
        do_eval=True,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        eval_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()


def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    assert num_layers_to_keep < (len(model.layers)-1)

    oldModuleList = model.layers
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(0, num_layers_to_keep):
        newModuleList.append(oldModuleList[i])
    newModuleList.append(oldModuleList[-1])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.layers = newModuleList

    return copyOfModel


def experiment():
    with open('./configs/config.json','r') as f:
        config = json.load(f)
    f.close()
    
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    model = AutoModelForCausalLM.from_pretrained(config['model'])

    dataset_map = {
        "mmlu":"cais/mmlu",
        "boolq":"google/boolq"
    }

    dataset = load_dataset(dataset_map[config['dataset']], split='validation')

    train_args = TrainingArguments(
        output_dir='./models/',
        overwrite_output_dir=False,
        do_eval=True,
        prediction_loss_only=True,
    )

    for i in reversed(range(len(model.layers))):
        sub_model = deleteEncodingLayers(model, i)
        num_params = sum(p.numel() for p in sub_model.parameters() if p.requires_grad)

        print(f'Layers: {i} | Params: {sub_model}')
        trainer = Trainer(
            model=sub_model,
            args=train_args,
            eval_dataset=dataset,
            tokenizer=tokenizer,
        )

        trainer.train()


if __name__=='__main__':
    test()