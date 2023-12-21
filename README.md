# gpt2_funtune

Fine-tuning a distilgpt2 model to generate game reviews. The training data in `dataset.csv` was from Steam review (only 1000 random reviews were used in the script).

## Usage

```bash
python finetune4gamereviews.py -i dataset.csv -o  game_review_generator -s 1000
```

the `dataset.csv` is the training dataset contains columns `app_name`, `review_score`, and `review_text`. The trained model will be stored in the directory `gamereveiw_distillgpt2`. Only random 1000 samples will be used for the fine-tuning.


Examples of using the generated model can be found in note book `game_review_generate.ipynb`.