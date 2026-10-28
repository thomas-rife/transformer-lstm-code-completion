# Transformer vs LSTM on Code Completion

Our project compares the results of an LSTM vs a Transformer on Python code completion. The dataset we are using can be found [here](https://huggingface.co/datasets/codeparrot/github-code).

Developed by Brian Bowers, August Wetterau and Thomas Rife

## Downloading the data

Simply run from the home repository path: `python -m utils.download_data`

## Running the transformer

1. Train the model: `python -m transformer.train_transformer`
2. Evaluate the model: `python -m transformer.evaluate_transformer`

## Running the LSTM (CUDA and Windows Machines)

1. Train/evaluate the model: `py -m lstm.lstm`

## References

For a complete list of references and citation please look at our paper, found under the folder docs and linked [here](https://github.com/LMU-CMSI-Korpusik/projects-thomas_brian_august/blob/main/docs/Final_Paper.pdf).

This project used many Python packages. Many of these are cited in the paper, but here is a complete list: Pytorch, TQDM, Transformers, Scikit-learn, and several modules from the Python standard library. Hugging Face was especially helpful, as we used it both for our models and tokenizers. Finally, the paper style is from the ACL 2023 conference, and the template along with directions can be found here: <https://2023.aclweb.org/calls/style_and_formatting/>.

[Microsoft’s IntelliCode Paper](https://ar5iv.labs.arxiv.org/html/2005.08025)

[Stanford’s LSTM Automatic Code Completer](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2760954.pdf)

[GPT-2 Tokenizer](https://huggingface.co/docs/transformers/v4.51.3/model_doc/gpt2)

[CodeBERT Tokenizer](https://huggingface.co/microsoft/codebert-base)
