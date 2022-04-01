import requests
import hashlib
import pathlib
import os

import tqdm

files = [
    {
        "uri": "https://zenodo.org/record/6403789/files/bigram.model.dict?download=1",
        "md5": "e97e05bbf895d2e7cbad930c9710e059",
    },
    {
        "uri": "https://zenodo.org/record/6403789/files/bigram.model.mdl?download=1",
        "md5": "e8d6500b17b9727dced13b3dcbbf3c3b",
    },
    {
        "uri": "https://zenodo.org/record/6403789/files/trigram.model.dict?download=1",
        "md5": "e97e05bbf895d2e7cbad930c9710e059",
    },
    {
        "uri": "https://zenodo.org/record/6403789/files/trigram.model.mdl?download=1",
        "md5": "97a68114276179df46707738b6bc194b",
    },
    {
        "uri": "https://zenodo.org/record/6403789/files/bilstm_state_dict.pt?download=1",
        "md5": "26ac3b99913adaae0f18a885de782a88",
    },
    {
        "uri": "https://zenodo.org/record/6403789/files/lstm_state_dict.pt?download=1",
        "md5": "53fe015aef0f770a2f8e398d8ca04995",
    },
    {
        "uri": "https://zenodo.org/record/6403789/files/rnn_state_dict.pt?download=1",
        "md5": "ea1267b10232e66328aa85d69da7914e",
    },
    {
        "uri": "https://zenodo.org/record/6403789/files/neuralnet_word2id_dict.pkl?download=1",
        "md5": "2b46111a675f86b749f1a06682ed3051",
    },
]


def download_big_file_and_check_md5(uri, md5sum, local_path):
    """Download a big file with requests with a tqdm progress bar"""
    r = requests.get(uri, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024
    t = tqdm.tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(local_path, "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        raise Exception("Downloaded file size mismatch")
    md5 = hashlib.md5()
    with open(local_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    if md5.hexdigest() != md5sum:
        raise Exception("MD5 mismatch")


def uri_to_filename(uri):
    return uri.split("/")[-1].replace("?download=1", "")


if __name__ == "__main__":

    # create folder
    pathlib.Path("model_checkpoints").mkdir(parents=True, exist_ok=True)

    # Download files and check MD5 checksums
    for file in files:
        download_big_file_and_check_md5(
            file["uri"],
            file["md5"],
            os.path.join("model_checkpoints", uri_to_filename(file["uri"])),
        )