import os
from batch_synthesize_controversial_pairs import (
    NaturalSentenceAssigner,
    synthesize_controversial_sentence_pair_set,
)

if __name__ == "__main__":
    model_pairs = [
        (
            "bert",
            "bert_has_a_mouth",
        ),
        ("bert_has_a_mouth", "bert"),
        (
            "electra",
            "electra_has_a_mouth",
        ),
        ("electra_has_a_mouth", "electra"),
        (
            "roberta",
            "roberta_has_a_mouth",
        ),
        ("roberta_has_a_mouth", "roberta"),
    ]

    initial_sentence_assigner = NaturalSentenceAssigner(model_pairs)
    sent_len = 8

    results_csv_folder = os.path.join(
        "synthesized_sentences",
        "bidirectional_prob_calc_exp",
        "controverisal_sentence_pairs_natural_initialization",
        "{}_word".format(sent_len),
    )

    synthesize_controversial_sentence_pair_set(
        model_pairs,
        initial_sentence_assigner,
        results_csv_folder=results_csv_folder,
        sent_len=sent_len,  # in the preprint, we used 8 word sentences
        allow_only_prepositions_to_repeat=True,  # in the preprint, this was True
        natural_initialization=True,  # sample random sentence for initialization
        max_sentence_pairs_per_run=1,  # set this to a small number (e.g. 5) if HPC job time is limited, None if you want the code to keep running until it's done
        max_non_decreasing_loss_attempts_per_word=100,
        max_replacement_attempts_per_word=100,
        max_opt_hours=12,
        verbose=3,
    )
