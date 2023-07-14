import os
import pdb

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import textwrap


from behav_exp_analysis import data_preprocessing, get_models, plot_main_results_figures, calc_binarized_accuracy, niceify
from model_functions import model_factory

def get_token_counts_for_all_models(df):

    # reduce df to unique sentence pairs
    df2 = df.drop_duplicates("sentence_pair")

    # extract model list from csv:
    model_names = get_models(df2)

    models = {}
    for model_name in model_names:
        print(f"Loading {model_name}")
        models[model_name] = model_factory(model_name, gpu_id=None, only_tokenizer=True)

    # for each model, generate a cross-validated logistic regression prediction of the probability of human preference
    for model_name, model in models.items():
        print(f"Couting tokens for {model_name}")
        # generate token counts
        df2[f"sentence1_{model_name}_token_count"] = model.count_tokens(df2["sentence1"])
        df2[f"sentence2_{model_name}_token_count"] = model.count_tokens(df2["sentence2"])

    # merge into original df (which has multiple rows per sentence pair)
    df = df.merge(df2, on="sentence_pair", how = "left", suffixes=('', '_y'))

    # drop columns with _y suffix
    df = df[df.columns.drop(list(df.filter(regex='_y')))]
    return df

def estimate_loocv_logistic_regression_predictions(df):

    # for each model and each sentence pair, genererate a cross-validated
    # logistic regression prediction of the probability of human preference
    # for the first sentence in the pair, using both the log prob difference and token count difference
    # as independent variables.

    # args:
    # df: dataframe including tokens

    df = df.copy()

    y = np.asarray(df["rating"] >= 4, dtype=np.float32)


    model_names = get_models(df)

    # for each model, generate a cross-validated logistic regression prediction of the probability of human preference
    for model_name in model_names:


        log_p1 = df[f"sentence1_{model_name}_prob"].to_numpy()
        log_p2 = df[f"sentence2_{model_name}_prob"].to_numpy()

        # replace -inf with min value
        log_p1[log_p1 == -np.inf] = np.min(log_p1[log_p1 != -np.inf])
        log_p2[log_p2 == -np.inf] = np.min(log_p2[log_p2 != -np.inf])

        n_tokens_1 = df[f"sentence1_{model_name}_token_count"]
        n_tokens_2 = df[f"sentence2_{model_name}_token_count"]

        def PenLP(logP, n_tokens):
            return logP / ((5+n_tokens)/(5+1))**0.8

        x1 = log_p2 - log_p1
        x2 = log_p2/n_tokens_2 - log_p1/n_tokens_1
        x3 = PenLP(log_p2, n_tokens_2) - PenLP(log_p1, n_tokens_1)

        x = np.stack([x1, x2, x3], axis=1)

        # estimate loocv predictions
        loocv_predictions = np.full_like(y, fill_value=np.nan)
        loocv_logproba = np.full((len(y),2), dtype=np.float32, fill_value=np.nan)

        reg_coeffs = np.full((len(y),x.shape[1]), dtype=np.float32, fill_value=np.nan)

        uq_sentence_pairs = df["sentence_pair"].unique()
        for i, sentence_pair in tqdm(enumerate(uq_sentence_pairs)):
            test_mask = df["sentence_pair"] == sentence_pair
            training_mask = np.logical_not(test_mask)

            clf = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000, fit_intercept=False)

            clf.fit(x[training_mask], y[training_mask])

            reg_coeffs[test_mask] = clf.coef_

            loocv_predictions[test_mask] = clf.predict(x[test_mask])
            loocv_logproba[test_mask] = clf.predict_log_proba(x[test_mask])

        print(f"Model: {model_name}, accuracy: {np.mean(loocv_predictions == y)}")

        mean_coefs = np.mean(reg_coeffs, axis=0)

        print(f"Model: {model_name}, mean coefficients: {mean_coefs}")

        # replace raw probabilities with logistic regression predictions
        df[f"sentence1_{model_name}_prob"] = loocv_logproba[:,0]
        df[f"sentence2_{model_name}_prob"] = loocv_logproba[:,1]
        # this allows us to reuse the same downstream data analysis code
    return df

def wrap_text_to_multiline_latex(text, width=20):
    # Split the text into multiple lines
    lines = textwrap.wrap(text, width)
    # Join the lines using the shortstack command
    return '\\shortstack{' + ' \\\\ '.join(lines) + '}'

def pandas_to_latex(df, caption, label):

    df = df.copy()

    # Convert all float values to percentages with 2 decimal places
    df = df.applymap(lambda x: f'{x*100:.2f}\\%')

    # Wrap the column headers into multiple lines
    df.columns = [wrap_text_to_multiline_latex(col) for col in df.columns]


    df.index = niceify(list(df.index))
    latex_table = df.to_latex(
        escape=False,
        multicolumn=True,
        multicolumn_format='c',
        index_names=True,
        caption=caption,
        label=label
    )

    # Add top rule and mid rule
    latex_table = latex_table.replace("\\toprule", "\\toprule\n\\textbf{Model} & \\multicolumn{2}{c}{\\textbf{Human-choice prediction accuracy}} \\\\")

    return latex_table


if __name__ == '__main__':
    data_file = "behavioral_results/contstim_Aug2021_n100_results_anon.csv"
    new_data_file = "behavioral_results/contstim_Aug2021_n100_results_anon_aligned_with_loso_with_lr_predictions.csv"

    df = data_preprocessing(data_file)
    if not os.path.exists(new_data_file):
        df_lr = get_token_counts_for_all_models(df)
        df_lr = estimate_loocv_logistic_regression_predictions(df_lr)
        df_lr.to_csv(new_data_file)
    else:
        df_lr = pd.read_csv(new_data_file)

    plot_main_results_figures(
        df,
        measure="binarized_accuracy",
        save_folder="figures/token_count_normalized_predictions/standard",
        figure_set="4",
        force_panel_letters=True,
        initial_panel_letter_index=0,
    )

    plot_main_results_figures(
        df_lr,
        measure="binarized_accuracy",
        save_folder="figures/token_count_normalized_predictions/token_count_normalized",
        figure_set="4",
        force_panel_letters=True,
        initial_panel_letter_index=1,
    )

    acc_standard = calc_binarized_accuracy(df,drop_model_prob=True).mean()
    acc_lr = calc_binarized_accuracy(df_lr, drop_model_prob=True).mean()

    # drop all columns except for the model name columns
    models = get_models(df)
    acc_standard = acc_standard[models]
    acc_lr = acc_lr[models]

    new_df = pd.DataFrame({
        "Unnormalized sentence probability estimates": acc_standard,
        "Token-count-adjusted sentence\nprobability estimates": acc_lr})

    print(pandas_to_latex(new_df, 'Accuracy Analysis', 'tab:accuracy_analysis'))
