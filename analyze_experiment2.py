import os
import json

import numpy as np
import pandas as pd

import behav_exp_analysis


# prepare behavioral data from its raw format


def preprocess_experiment2():
    fields_to_keep = [
        "Zone Type",
        "Trial Number",
        "Zone Name",
        "Response",
        "Reaction Time",
        "counterbalance-o1ql",
        "sentence1",
        "sentence2",
        "sentence1_type",
        "sentence2_type",
        "Participant Private ID",
    ]

    try:
        assert os.path.exists(
            "behavioral_results/contstim_Nov2022_30_participants_anon.csv"
        )
    except:
        csvs = [
            os.path.join(
                "behavioral_results",
                "contstim_Nov2022_data_exp_48362-v23_task-kwi1.csv",
            ),
            os.path.join(
                "behavioral_results",
                "contstim_Nov2022_data_exp_48362-v24_task-kwi1.csv",
            ),
        ]

        df = pd.concat([pd.read_csv(csv) for csv in csvs])
        df = df.loc[[str(s).startswith("resp") for s in df["Zone Name"]]]

        if "sentence1_type" in df.columns:
            df.drop("sentence1_type", axis=1, inplace=True)
        if "sentence2_type" in df.columns:
            df.drop("sentence2_type", axis=1, inplace=True)
        if "sentence1" in df.columns:
            df.drop("sentence1", axis=1, inplace=True)
        if "sentence2" in df.columns:
            df.drop("sentence2", axis=1, inplace=True)

        df = df.rename(
            columns={
                "sentence1_type_set 1": "sentence1_type",
                "sentence2_type_set 1": "sentence2_type",
                "sentence1_set 1": "sentence1",
                "sentence2_set 1": "sentence2",
                "sentence1_model_set 1": "sentence1_model",
                "sentence2_model_set 1": "sentence2_model",
            }
        )

        if "Participant Private ID" in df.columns:

            # we excluded three participants from the analysis for behavior indicating low effort.
            # remove excluded participants
            # the IDs are not included in the repo for privacy reasons
            excluded_participants = json.load("behavioral_results/contstim_Nov2022_excluded_participants.json")
            df = df[~df["Participant Private ID"].isin(excluded_participants)]

            IDs, df["subject"] = np.unique(
                df["Participant Private ID"], return_inverse=True
            )
            df = df.drop(columns=["Participant Private ID"])
            pd.DataFrame(IDs).to_csv(
                "behavioral_results/contstim_Nov2022_30_participants_subject_ID_list.csv"
            )
        else:
            assert "subject" in df.columns, "subject column not found"

        n_subjects = len(df["subject"].unique())

        print("found {} subjects".format(n_subjects))

        # leave only the selection fields:
        df = df.drop(columns=[col for col in df.columns if col not in fields_to_keep])

        model_list = [
            "bert",
            "bert_has_a_mouth",
            "electra",
            "electra_has_a_mouth",
            "roberta",
            "roberta_has_a_mouth",
        ]

        behav_exp_analysis.add_model_sentence_probabilities(
            df, model_list, remove_existing=False
        )
        df.to_csv(
            f"behavioral_results/contstim_Nov2022_{n_subjects}_participants_anon.csv"
        )

    finally:
        df = behav_exp_analysis.data_preprocessing(
            results_csv="behavioral_results/contstim_Nov2022_30_participants_anon.csv",
            experiment=2,
        )

    behav_exp_analysis.catch_trial_report(df, subject_id_column="subject")

    df.to_csv(
        "behavioral_results/contstim_Nov2022_30_participants_anon_aligned_with_loso.csv")

if __name__ == "__main__":
    try:
        df = pd.read_csv(
            "behavioral_results/contstim_Nov2022_30_participants_anon_aligned_with_loso.csv"
        )
    except:
        preprocess_experiment2()
        df = pd.read_csv(
            "behavioral_results/contstim_Nov2022_30_participants_anon_aligned_with_loso.csv"
        )

    model_combinations_to_contrast = [
        ("bert", "bert_has_a_mouth"),
        ["electra", "electra_has_a_mouth"],
        ["roberta", "roberta_has_a_mouth"],
    ]

    behav_exp_analysis.plot_main_results_figures(
        df,
        save_folder="figures/exp2/binarized_acc_by_subject",
        measure="RAE_signed_rank_cosine_similarity",
        figure_set="exp2_synthetic",
        exp="exp2",
        statistical_testing_level="subject",
        model_combinations_to_contrast=model_combinations_to_contrast,
        initial_panel_letter_index=1,
    )

    behav_exp_analysis.generate_worst_sentence_pairs_table(
        df,
        trial_type="synthetic_vs_synthetic",
        n_sentences_per_model=2,
        target_folder="has_a_mouth_exp_tables",
        models = ["bert_has_a_mouth", "electra_has_a_mouth", "roberta_has_a_mouth"],
    )

    # as a complement to the main results, we also plot the results of experiment 1 in the
    # randomly-sampled natural sentence condition using the same models, plotting, and inference
    # used in experiment 2.

    try:
        df = pd.read_csv(
            "behavioral_results/contstim_Aug2021_n100_results_anon_with_PLL_models_aligned_with_loso.csv"
        )
    except:
        df = behav_exp_analysis.data_preprocessing()
        df = behav_exp_analysis.add_model_sentence_probabilities(
            df,
            ["bert_has_a_mouth", "electra_has_a_mouth", "roberta_has_a_mouth"],
        )
        # drop irrelevant models
        models_to_drop = ["gpt2", "bigram", "trigram", "xlm", "rnn", "lstm"]
        df = df[
            [col for col in df.columns if not any([m in col for m in models_to_drop])]
        ]
        df.to_csv(
            "behavioral_results/contstim_Aug2021_n100_results_anon_with_PLL_models_aligned_with_loso.csv"
        )

    behav_exp_analysis.plot_main_results_figures(
        df,
        save_folder="figures/exp1/binarized_acc_by_subject",
        measure="RAE_signed_rank_cosine_similarity",
        figure_set="exp1_natural_sents_reanalyzed_for_exp",
        exp="exp2",
        statistical_testing_level="subject",
        model_combinations_to_contrast=model_combinations_to_contrast,
        initial_panel_letter_index=0,
    )
