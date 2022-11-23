import os
from pyexpat import model
import subprocess
import pathlib


import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import scipy.stats

from model_functions import model_factory
from sentence_optimization import optimize_sentence_set

try:
    Lau_repo_folder = os.path.join("third_party", "acceptability_prediction_in_context")
    from third_party.acceptability_prediction_in_context.code.calc_corr import get_sentence_data
except:
    if not os.path.exists(Lau_repo_folder):
        subprocess.run(
            [
                "git",
                "submodule",
                "add",
                "--force",
                "https://github.com/jhlau/acceptability-prediction-in-context.git",
                Lau_repo_folder,
            ]
        )
    assert os.path.exists(Lau_repo_folder)
    from third_party.acceptability_prediction_in_context.code.calc_corr import get_sentence_data


def import_Lau_2020_data(context_condition="none", max_sentence_length=None):
    """Load human acceptability ratings from 'How Furiously Can Colorless Green Ideas Sleep? Sentence Acceptability in Context'

    args:
        context_condition: str, one of 'none', 'random', 'real'. Here we use only 'none'.
    returns:
        data: pandas.DataFrame
    """

    assert os.path.exists(Lau_repo_folder)

    data_dict = get_sentence_data(
        os.path.join(Lau_repo_folder,
            "human-ratings", 
            f"context-{context_condition}-ratings.csv"
        )
    )

    human_ratings_df = pd.DataFrame.from_dict(data_dict,orient='index')

    human_ratings_df["translated"] = human_ratings_df["translated"].astype(int)
    human_ratings_df["n_ratings"] = [
        len(ratings) for ratings in human_ratings_df["full-ratings"]
    ]

    # remove final period from sentences, where applicable (it is added again in model evaluation):
    human_ratings_df["sentence"] = [
        sentence[:-1] if sentence[-1] == "." else sentence
        for sentence in human_ratings_df["sentence"]
    ]

    if max_sentence_length is not None:
        human_ratings_df = human_ratings_df[
            human_ratings_df["n_words"] <= max_sentence_length
        ]

    return human_ratings_df


def get_model_loglikelihoods(human_ratings_df, model_name, cache_folder="full_ratings"):
    # first, see if we cached the results:
    cached_model_loglikelihoods_filename = os.path.join(
        "cached_model_loglikelihoods", cache_folder, f"{model_name}.csv"
    )
    if os.path.exists(cached_model_loglikelihoods_filename):
        cache_df = pd.read_csv(cached_model_loglikelihoods_filename)
        cache_df.set_index("sentence").loc[human_ratings_df.sentence]
        model_log_likelihoods = torch.tensor(
            cache_df[f"{model_name}_log_likelihood"], dtype=torch.float32
        )
    else:
        model = model_factory(model_name, 0)
        model_log_likelihoods = []
        for sentence in tqdm(human_ratings_df["sentence"]):
            model_log_likelihoods.append(model.sent_prob(sentence))
        model_log_likelihoods = torch.tensor(model_log_likelihoods, dtype=torch.float32)
        # cache the results:
        dirname = os.path.dirname(cached_model_loglikelihoods_filename)
        if not os.path.exists(dirname):
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        cache_df = pd.DataFrame(
            {
                "sentence": human_ratings_df["sentence"],
                f"{model_name}_log_likelihood": model_log_likelihoods.numpy(),
            }
        )
        cache_df.to_csv(cached_model_loglikelihoods_filename, index=False)
    return model_log_likelihoods


def get_model_list():
    model_list = [
        "gpt2",
        #'lstm','rnn',
        # "bigram",
        # "trigram",
        # "bert",
        "bert_new_implementation",
        # "bert_has_a_mouth",
        # "electra",
        "electra_new_implementation",
        # "electra_has_a_mouth",
        # "roberta",
        "roberta_new_implementation",
        # "roberta_has_a_mouth",
        # "xlm",
    ]
    return model_list

def prepare_all_rating_models(max_sentence_length=None, measure_sentences_by_words=False):

    human_ratings_df = import_Lau_2020_data(max_sentence_length=max_sentence_length)

    model_list = get_model_list()

    data_tag = f"max_sentence_length_{max_sentence_length}" if max_sentence_length is not None else "full_ratings"
    for model_name in model_list:
        # print(f"Computing log-likelihoods for {model_name}")
        model_log_likelihoods = get_model_loglikelihoods(
            human_ratings_df,
            model_name,
            cache_folder=data_tag
        )
        
        readout_model, r = calibrate_model(
            human_ratings_df,
            model_log_likelihoods,
            measure_sentences_by_words=measure_sentences_by_words,
            verbose=False,
        )

        print(f"{model_name} r = {r:.3f}")

        if measure_sentences_by_words:
            model_tag = '_n_words_normalized'
        else:
            model_tag = '_n_chars_normalized'

        model_save_folder = os.path.join("absolute_rating_models", "calibrated_on_" + data_tag + model_tag)

        if not os.path.exists(model_save_folder):
            pathlib.Path(model_save_folder).mkdir(parents=True, exist_ok=True)

        # save the response model parameters:
        response_model_filename = os.path.join(
            model_save_folder, f"{model_name}_rating_model.pth")
        torch.save(readout_model.state_dict(), response_model_filename)
    
def acquire_absolute_rating_controversiality_fun(model_accept_rating_module, model_reject_rating_module):
    """ Returns a functions that quantifies the controversiality of a sentence by contrasting absolute rating models.
    """
        
    def absolute_rating_controversiality(sentence_log_p, n_characters):
        """ args:
            sentence_log_p: (numpy.array,  M_models x 1 sentences or D_designs x M_models x 1 sentences)
            n_characters: (numpy.array, 1 sentences or D_designs x 1 sentences)
        """

        assert sentence_log_p.shape[-1] == 1, "only one sentence at a time"
        assert n_characters.shape[-1] == 1, "only one sentence at a time"

        m_accept_log_prob = sentence_log_p[..., 0, 0]
        m_reject_log_prob = sentence_log_p[..., 1, 0]
        n_characters = n_characters[..., 0]

        if not isinstance(m_accept_log_prob, torch.Tensor):
            m_accept_log_prob = torch.tensor(m_accept_log_prob, dtype=torch.float32)
        if not isinstance(m_reject_log_prob, torch.Tensor):
            m_reject_log_prob = torch.tensor(m_reject_log_prob, dtype=torch.float32)
        
        
        if n_characters is not None:
            sentence_length = n_characters                
            if not isinstance(sentence_length, torch.Tensor):
                sentence_length = torch.tensor(sentence_length, dtype=torch.float32)
            
        with torch.no_grad():
            model_accept_predicted_rating = model_accept_rating_module(
                m_accept_log_prob,
                sentence_length).to('cpu').numpy()
            model_reject_predicted_rating = model_reject_rating_module(
                m_reject_log_prob,
                sentence_length).to('cpu').numpy()

        # the ratings are on a scale from 1 to 4.
        controversiality = np.minimum(model_accept_predicted_rating - 2.5, 2.5 - model_reject_predicted_rating)
        return controversiality         
    
    return absolute_rating_controversiality

def load_rating_model(model_name, data_tag="full_ratings", model_tag="_n_chars_normalized"):
    model_save_folder = os.path.join("absolute_rating_models", "calibrated_on_" + data_tag + model_tag)
    rating_model_filename = os.path.join(
        model_save_folder, f"{model_name}_rating_model.pth")

    readout_model = LogLikelihoodToAcceptabilityRating()
    readout_model.load_state_dict(torch.load(rating_model_filename))
    return readout_model

def _test_rating_models_loading(max_sentence_length=None, measure_sentences_by_words=False):
    """ make sure that the rating models can be loaded and used to predict ratings."""

    model_list = get_model_list()
    human_ratings_df = import_Lau_2020_data(max_sentence_length=max_sentence_length)

    data_tag = f"max_sentence_length_{max_sentence_length}" if max_sentence_length is not None else "full_ratings"

    if measure_sentences_by_words:
        model_tag = '_n_words_normalized'
    else:
        model_tag = '_n_chars_normalized'

    for model_name in model_list:
        # print(f"Computing log-likelihoods for {model_name}")
        model_log_likelihoods = get_model_loglikelihoods(
            human_ratings_df,
            model_name,
            cache_folder=data_tag
        )
        
        readout_model = load_rating_model(model_name, data_tag=data_tag, model_tag=model_tag)

        if measure_sentences_by_words:
            sentence_length = torch.tensor([len(s.split()) for s in human_ratings_df.sentence], dtype=torch.float32)
        else:   
            sentence_length = torch.tensor([len(s) for s in human_ratings_df.sentence], dtype=torch.float32)

        with torch.no_grad():
            model_predicted_rating = readout_model(
                model_log_likelihoods, 
                sentence_length).to('cpu').numpy()
        
        r = scipy.stats.pearsonr(model_predicted_rating, human_ratings_df["mean-ratings"])[0]
        print(f"{model_name} r = {r:.3f}")

def torch_pearsonr(x, y):
    """Compute Pearson correlation coefficient between x and y."""
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm * xm) * torch.sum(ym * ym))
    r_val = r_num / r_den
    return r_val


class LogLikelihoodToAcceptabilityRating(torch.nn.Module):
    """A trainable mapping from log-likelihood to 4-point acceptability rating."""

    def __init__(self, slope=1.0, intercept=0.0, alpha=0.8):
        super().__init__()

        self.slope = torch.nn.Parameter(torch.tensor(slope, dtype=torch.float32), requires_grad=False)
        self.intercept = torch.nn.Parameter(
            torch.tensor(intercept, dtype=torch.float32), requires_grad=False)
        self.alpha = torch.nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=False)
        # self.B = torch.nn.Parameter(torch.tensor(B, dtype=torch.float32), requires_grad=False)

    def raw_predictions(self, LL, sentence_length):
        # denominator = ((self.B + sentence_length)/(self.B+1))**self.alpha
        denominator = sentence_length**self.alpha
        raw_prediction = LL / denominator
        return raw_prediction

    def forward(self, LL, sentence_length):
        raw_prediction = self.raw_predictions(LL, sentence_length)
        return torch.sigmoid(raw_prediction * self.slope + self.intercept) * 3 + 1


def calibrate_model(
    human_ratings_df,
    model_log_likehoods,
    measure_sentences_by_words=True,
    verbose=False,
):
    """Calibrate a model to human acceptability ratings.

    args:
        human_ratings_df: pandas.DataFrame, human acceptability ratings
        model_log_likehoods: torch.tensor, log-likelihoods of the model
        measure_sentences_by_words: bool, if True, sentences are measured by words, otherwise by characters.
        verbose: bool
    returns:
        readout_model (torch.nn.Module), r (float) 
    """

    # start with calibrating the slope and intercept, using the default values of alpha and B.
    human_ratings = torch.tensor(
        list(human_ratings_df["mean-ratings"]), dtype=torch.float32
    )
    n_human_ratings = torch.tensor(
        list(human_ratings_df["n_ratings"]), dtype=torch.float32
    )

    sentence_length_in_words = torch.tensor(
        [len(sentence.split()) for sentence in human_ratings_df["sentence"]],
        dtype=torch.float32,
    )
    sentence_length_in_characters = torch.tensor(
        [len(sentence) for sentence in human_ratings_df["sentence"]],
        dtype=torch.float32,
    )

    if measure_sentences_by_words:
        sentence_length = sentence_length_in_words
    else:
        sentence_length = sentence_length_in_characters

    def weighted_MSE_loss(model_predictions, human_ratings, n_human_ratings):
        """Compute weighted loss between model log-likelihoods and human ratings."""
        return torch.sum(
            n_human_ratings * (model_predictions - human_ratings) ** 2
        ) / torch.sum(n_human_ratings)

    def MSE_loss(model_predictions, human_ratings):
        """Compute loss between model log-likelihoods and human ratings."""
        return torch.mean((model_predictions - human_ratings) ** 2)
        
    def optimize(
        readout_model,
        model_log_likehoods,
        human_ratings,
        n_human_ratings,
        sentence_length,
        parameter_names,
        verbose=True,
    ):
        parameters = [
            getattr(readout_model, parameter_name) for parameter_name in parameter_names
        ]

        # turn on requires_grad for the parameters we want to optimize
        for parameter in parameters:
            parameter.requires_grad = True

        optimizer = torch.optim.LBFGS(
            parameters,
            lr=0.1,
            max_iter=100,
            max_eval=100,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            model_predictions = readout_model(model_log_likehoods, sentence_length)
            loss = MSE_loss(model_predictions, human_ratings)
            loss.backward()
            return loss

        converged = False
        i_iter = 0
        n_iterations_without_improvement = 0
        best_loss = closure()
        if verbose:
            print(f"initial loss: {best_loss}")
        while not converged:
            optimizer.step(closure)
            cur_loss = closure()
            if verbose:
                print(
                    f"iteration {i_iter}, loss: {cur_loss}, iterations without improvement: {n_iterations_without_improvement}"
                )
            i_iter += 1
            if cur_loss < best_loss:
                best_loss = cur_loss
                n_iterations_without_improvement = 0
            else:
                n_iterations_without_improvement += 1
            converged = n_iterations_without_improvement > 5

        # turn off requires_grad for the parameters we want to optimize
        for parameter in parameters:
            parameter.requires_grad = False
        return best_loss, readout_model

    def fit_readout_model(
        model_log_likehoods,
        human_ratings,
        n_human_ratings,
        sentence_length,
        slope=1.0,
        alpha=0.8,
        # B=5.0,
        verbose=True,
    ):

        readout_model = LogLikelihoodToAcceptabilityRating(
            intercept=-model_log_likehoods.mean(), slope=slope, alpha=alpha,# B=B
        )
        mean_raw_prediction = (
            readout_model.raw_predictions(model_log_likehoods, sentence_length)
            .mean()
            .detach()
        )
        readout_model.intercept.data = -mean_raw_prediction

        best_loss, readout_model = optimize(
            readout_model,
            model_log_likehoods,
            human_ratings,
            n_human_ratings,
            sentence_length,
            parameter_names=["slope", "intercept"],
            verbose=verbose,
        )

        # # now, calibrate alpha and B:
        best_loss, readout_model = optimize(
            readout_model,
            model_log_likehoods,
            human_ratings,
            n_human_ratings,
            sentence_length,
            parameter_names=[
                "slope",
                "intercept",
                "alpha",
            ],
            verbose=verbose,
        )

        # evaluate Pearson correlation
        model_predictions = readout_model(model_log_likehoods, sentence_length)
        pearson_correlation = scipy.stats.pearsonr(
            model_predictions.detach().numpy(), human_ratings.detach().numpy()
        )[0]
        if verbose:
            print(f"Pearson correlation: {pearson_correlation}")

        return pearson_correlation, readout_model

    r, readout_model = fit_readout_model(
        model_log_likehoods,
        human_ratings,
        n_human_ratings,
        sentence_length,
        slope=1.0,
        alpha=0.8,
        # B=5,
        verbose=verbose,
    )

    return readout_model, r

def _test_absolute_rating_controversiality():
    model_accept_name = 'bert_has_a_mouth'
    model_reject_name = 'electra_has_a_mouth'

    human_ratings_df = import_Lau_2020_data()

    model_save_folder = os.path.join("absolute_rating_models", "calibrated_on_full_ratings_n_chars_normalized")

    model_accept_rating_module= load_rating_model(model_accept_name)
    model_reject_rating_module = load_rating_model(model_reject_name)

    model_accept_log_likelihoods =  get_model_loglikelihoods(human_ratings_df, model_name=model_accept_name)
    model_reject_log_likelihoods =  get_model_loglikelihoods(human_ratings_df, model_name=model_reject_name)


    sentence_length_in_chars = torch.tensor(
        [len(sentence) for sentence in human_ratings_df["sentence"]], dtype=torch.float32)

    absolute_rating_controversiality = acquire_absolute_rating_controversiality_fun(
        model_accept_rating_module,
        model_reject_rating_module)

    sentence_log_p = torch.stack([model_accept_log_likelihoods, model_reject_log_likelihoods],axis=1).unsqueeze(-1)
    # shape - (n_sentences, n_models, 1)
    controversiality = absolute_rating_controversiality(
        sentence_log_p,
        n_characters=sentence_length_in_chars)

    assert controversiality.shape == (len(human_ratings_df),)

    # generate a dataframe sorted by controversiality
    controversiality_df = pd.DataFrame({
        "sentence": human_ratings_df["sentence"],
        model_accept_name + "_log_p": model_accept_log_likelihoods,
        model_reject_name + "_log_p": model_reject_log_likelihoods,
        model_accept_name + "_predicted_rating": model_accept_rating_module(model_accept_log_likelihoods, sentence_length_in_chars),
        model_reject_name + "_predicted_rating": model_reject_rating_module(model_reject_log_likelihoods, sentence_length_in_chars),
        "controversiality": controversiality,}
    )

    controversiality_df.sort_values("controversiality", ascending=False, inplace=True)

    # save the dataframe
    controversiality_df.to_csv(os.path.join(model_save_folder, "controversiality.csv"), index=False)

    # display all df columns
    pd.set_option('display.max_columns', None)
    print(controversiality_df.head(10))
    print(controversiality_df.tail(10))

    import matplotlib.pyplot as plt

    # plot the predicted ratings as a scatter plot, with the controversiality as the color
    plt.scatter(
        controversiality_df[model_accept_name + "_predicted_rating"],
        controversiality_df[model_reject_name + "_predicted_rating"],
        c=controversiality_df["controversiality"],
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        s=10,
    )
    plt.xlabel(model_accept_name + " predicted rating")
    plt.ylabel(model_reject_name + " predicted rating")
    plt.xlim(1,4)
    plt.ylim(1,4)
    cbar = plt.colorbar()
    cbar.set_label("controversiality")
    plt.show()

def synthesize_one_absolute_rating_controversial_sentence(model_accept_name, model_reject_name, initial_sentence):
    """
    
    Synthesize a sentence to maximize the controversiality between two models' predicted ratings

    """

    # load LM models
    print(f"loading model {model_accept_name}")
    model_accept = model_factory(model_accept_name, 0)
    print(f"loading model {model_reject_name}")
    model_reject = model_factory(model_reject_name, 0)
    print("done loading models")

    # load rating models
    model_accept_rating_module = load_rating_model(model_accept_name)
    model_reject_rating_module = load_rating_model(model_reject_name)

    # get the controversiality function
    absolute_rating_controversiality = acquire_absolute_rating_controversiality_fun(
        model_accept_rating_module,
        model_reject_rating_module)
    
    def loss_fun(*args, **kwargs):
        return -absolute_rating_controversiality(*args, **kwargs)

    def monitoring_func(sentences, sentences_log_p):
        print('\n')
        print(f"{model_accept_name} ratings:", model_accept_rating_module(
            torch.tensor(sentences_log_p[0], dtype=torch.float32),
            torch.tensor([len(sentences[0])], dtype=torch.float32)).item(),
            f"log_p: {sentences_log_p[0]}")            
            
        print(f"{model_reject_name} ratings:", model_reject_rating_module(
            torch.tensor(sentences_log_p[1], dtype=torch.float32),
            torch.tensor([len(sentences[0])], dtype=torch.float32)).item(),
            f"log_p: {sentences_log_p[1]}")
        
    results = optimize_sentence_set(
        n_sentences=1,
        models=[model_accept, model_reject],
        loss_func=loss_fun,
        sentences_to_change=None,
        sent_length_in_words=len(initial_sentence.split(' ')),
        replacement_strategy="exhaustive",
        sentences=[initial_sentence],
        max_steps=10000,
        internal_stopping_condition=lambda loss: False,
        external_stopping_check=lambda: False,
        max_replacement_attempts_per_word=50,
        max_non_decreasing_loss_attempts_per_word=5,
        keep_words_unique=True, 
        allowed_repeating_words=None,
        monitoring_func=monitoring_func,
        save_history=False,
        model_names=[model_accept_name, model_reject_name],
        do_pass_n_characters=True,
        verbose=3,
    )
    return results

if __name__ == "__main__":

    initial_sentence="This is the greatest paper I ever read in my whole life"
        
    # synthesize_one_absolute_rating_controversial_sentence(model_accept_name="bert_has_a_mouth", model_reject_name="electra_has_a_mouth", initial_sentence=initial_sentence)
    # prepare_all_rating_models(measure_sentences_by_words=False)
    # _test_rating_models_loading()
    # _test_absolute_rating_controversiality()

    # model_name = 'trigram'
    human_ratings_df = import_Lau_2020_data()
    # # # human_ratings_df['n_ratings'] = 1.0
    # # print(human_ratings_df['sentence'])
    # model_log_likelihoods = get_model_loglikelihoods(human_ratings_df, model_name=model_name)
    # readout_model, r = calibrate_model(human_ratings_df, model_log_likelihoods, measure_sentences_by_words=False)
    # sentence_length = [len(sentence.split(' ')) for sentence in human_ratings_df['sentence']]
    # model_predicted_ratings = readout_model(
    #     torch.tensor(model_log_likelihoods, dtype=torch.float32),
    #     torch.tensor(sentence_length, dtype=torch.float32))
    # # print all readout_model parameters:
    # for name, param in readout_model.named_parameters():
    #     print(f'{name}: {param.data}')
