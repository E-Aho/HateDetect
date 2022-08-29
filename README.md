# HateDetect
## A project investigating methods of improving hate speech detection methods for social media platforms.

#### NB: As part of this package, some unit tests and string conversion methods require the usage of harmful and hateful language. This is only done to try and advance hate speech modelling and detection, with the goal of reducing the impact of hate online.
#### As part of this project, files which contain hateful words or terms are marked with `CW_` (for "content warning") as part of the title, and contain a forward indicating the presence of hateful language.
#### Files beginning with `CW_` (or `test_CW_` in the case of pytest files) contain hateful slurs and terms within them.

[The written report for the project is viewable here.](https://github.com/E-Aho/HateDetect/blob/main/MainReport.pdf)

## Running the code
### Prerequesites
* Make sure you have Python > 3.9 and TF > 2.8 installed
* Navigate to the root of this repository
* Install the project requirements with ```pip install -r"requirements.txt"```
* To run a script, you will need to call the following command: `python3 src/path/to/script.py`

### Processing datasets
* First you need to collect and process the datasets, so run 
  * `datasets/collect_datasets.py`
  * `datasets/preprocess_hatexplain.py`
    * You may need a HuggingFace login to download the datasets using their API

### Training models
To train models, the `models/hyperparameter_sweep.py` file can be called. This will iterate over the different model configs set out in that file.

Otherwise, you can call each of the base files (e.g `bert_with_attention_entropy.py`) and adjust parameter settings inside that to change your model configs.

The other files contain helper functions and structures to minimise the amount of code that clutters up the main files, in an effort to make the logic more readable.

### Running Tests
To run the tests, from the root of the repo, you can just call 
```pytest tests```


## TODOs
* Adjust main files so that they can be called with different configs from command line.
* Refactor hyperparam sweep to do runs based off of single JSON/XML file elsewhere in repo.
* Fill out some more test cases & update old ones
