from tqdm import tqdm
import numpy as np

from model_functions import model_factory
from behav_exp_analysis import data_preprocessing, get_models

df = data_preprocessing()

model_names = get_models(df)

for model_name in model_names:
    print("loading model: ", model_name)
    model = model_factory(model_name, 0)

    df2 = df.groupby("sentence1").first().reset_index()
    sentences = list(df2["sentence1"].values)
    model_probs = list(df2[f"sentence1_{model_name}_prob"])

    df2 = df.groupby("sentence2").first().reset_index()
    sentences.extend(list(df2["sentence2"].values))
    model_probs.extend(list(df2[f"sentence2_{model_name}_prob"]))
    del df2

    for i, (sentence, model_prob) in enumerate(
        tqdm(
            zip(sentences, model_probs),
            total=len(sentences),
            desc=f"testing {model_name}",
        )
    ):
        eval_prob = model.sent_prob(sentence)
        assert np.isclose(eval_prob, model_prob), f"{eval_prob} != {model_prob}"
        if i > 20:
            break
    del model
