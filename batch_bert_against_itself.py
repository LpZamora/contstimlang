import os
from batch_synthesize_controversial_pairs import NaturalSentenceAssigner, synthesize_controversial_sentence_pair_set

from model_functions import model_factory
from sentence_optimization import (
    optimize_sentence_set,
    initialize_random_word_sentence,
    controversiality_loss_func,
)
from utils import exclusive_write_line
from task_scheduler import TaskScheduler



if __name__ == "__main__":
    all_model_names = [
        "bert",
        "bert_has_a_mouth",
    ]

    initial_sentence_assigner = NaturalSentenceAssigner(all_model_names)
    sent_len = 8

    results_csv_folder = os.path.join(
        "synthesized_sentences",
        "controverisal_sentence_pairs_natural_initialization",
        "{}_word".format(sent_len),
    )

    synthesize_controversial_sentence_pair_set(
        all_model_names,
        initial_sentence_assigner,
        results_csv_folder=results_csv_folder,
        sent_len=sent_len,  # in the preprint, we used 8 word sentences
        allow_only_prepositions_to_repeat=True,  # in the preprint, this was True
        natural_initialization=True,  # sample random sentence for initialization
        max_sentence_pairs_per_run=5,  # set this to a small number (e.g. 5) if HPC job time is limited, None if you want the code to keep running until it's done
        verbose=3,
    )
