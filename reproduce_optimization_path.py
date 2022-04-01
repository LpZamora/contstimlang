import random

import pandas as pd

from model_functions import model_factory


def dfs(
    current_sentence,
    target_sentence,
    history,
    m_accept,
    m_reject,
    m_accept_source_sentence_prob,
    m_reject_source_sentence_prob,
):
    """depth first search from current_sentence to target_sentence"""

    m_accept_prob_current = m_accept.sent_prob(current_sentence)
    m_reject_prob_current = m_reject.sent_prob(current_sentence)

    admissable = (m_accept_prob_current >= m_accept_source_sentence_prob) and (
        m_reject_prob_current <= m_reject_source_sentence_prob
    )

    if admissable:
        print("admissable:", current_sentence)
        history = history + [
            {
                "sentence": current_sentence,
                "m_accept_prob": m_accept_prob_current,
                "m_reject_prob": m_reject_prob_current,
            }
        ]
        if current_sentence == target_sentence:
            return True, history
        else:

            n_words = len(current_sentence.split(" "))

            # get potential next sentences
            word_idx_list = [
                i_word
                for i_word in range(n_words)
                if current_sentence.split(" ")[i_word]
                != target_sentence.split(" ")[i_word]
            ]
            random.shuffle(
                word_idx_list
            )  # randomize the order of the words to be considered

            for word_idx in word_idx_list:
                next_sentence = " ".join(
                    current_sentence.split(" ")[:word_idx]
                    + [target_sentence.split(" ")[word_idx]]
                    + current_sentence.split(" ")[word_idx + 1 :]
                )
                next_sentence_success, next_sentence_history = dfs(
                    next_sentence,
                    target_sentence,
                    history,
                    m_accept,
                    m_reject,
                    m_accept_source_sentence_prob,
                    m_reject_source_sentence_prob,
                )
                if next_sentence_success:
                    return True, next_sentence_history
    else:
        print("not admissable:", current_sentence)

    # does not lead to the solution
    return False, history


def reproduce_optimization_path(
    source_sentence, target_sentence, m_accept_model_name, m_reject_model_name
):
    """retrace a possible optimization path between source_sentence and target_sentence

    args:
        source_sentence: original sentence
        target_sentence: target sentence
        m_accept_model_name: the name of the model for "accepting" the target sentence
        m_reject_model_name: the name of the model for "rejecting" the target sentence
    returns:
        a dataframe representing the possible optimization path
    """

    assert len(source_sentence.split(" ")) == len(
        target_sentence.split(" ")
    ), "source and target sentences must be of the same length"

    # load the models
    m_accept = model_factory(m_accept_model_name, 0)
    m_reject = model_factory(m_reject_model_name, 1)

    m_accept_source_sentence_prob = m_accept.sent_prob(source_sentence)
    m_reject_source_sentence_prob = m_reject.sent_prob(source_sentence)

    m_accept_target_sentence_prob = m_accept.sent_prob(target_sentence)
    m_reject_target_sentence_prob = m_reject.sent_prob(target_sentence)

    assert (
        m_accept_target_sentence_prob >= m_accept_source_sentence_prob
    ), "for m_accept, target sentence must be of greater or equal probability than source sentence"
    assert (
        m_reject_target_sentence_prob < m_reject_source_sentence_prob
    ), "for m_reject target sentence must be of lower probability than source sentence"

    # start the search
    success, history = dfs(
        source_sentence,
        target_sentence,
        [],
        m_accept,
        m_reject,
        m_accept_source_sentence_prob,
        m_reject_source_sentence_prob,
    )

    if success:
        history = pd.DataFrame(history)
        history = history.rename(
            columns={
                "m_accept_prob": f"{m_accept_model_name}_prob",
                "m_reject_prob": f"{m_reject_model_name}_prob",
            }
        )
        print(history)
        history.to_csv(f"{source_sentence}_to_{target_sentence}_history.csv")
    else:
        print("no optimization path found")


if __name__ == "__main__":

    reproduce_optimization_path(
        "This is the lie you have been sold",
        "That is the narrative we have been sold",
        m_accept_model_name="gpt2",
        m_reject_model_name="bert",
    )
    reproduce_optimization_path(
        "This is the lie you have been sold",
        "This is the week you have been dying",
        m_accept_model_name="bert",
        m_reject_model_name="gpt2",
    )
