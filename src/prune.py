from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import json
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
        #report_to=['tensorboard'],
        #logging_dir=f'./runs/baseline_{config['dataset']}'
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        eval_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__=='__main__':
    test()