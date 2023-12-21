# gpt2_funtune

Fine-tuning a distilgpt2 model to generate game reviews. The training data in `dataset.csv` was from Steam review (only 1000 random reviews were used in the script).

## Usage

```bash
python3 finetune4gamereviews.py

```

the trained model will be stored in the directory `gamereveiw_distillgpt2`. Examples of using the generated model can be found in note book `game_review_generate.ipynb`.