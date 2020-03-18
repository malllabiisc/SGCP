# Syntax-Guided Controlled Generation of Paraphrases

Source code for [TACL 2020](https://transacl.org/) paper: Syntax-Guided Controlled Generation of Paraphrases

<p align="center">
  <img align="center" src="https://github.com/malllabiisc/SGCP/blob/master/images/SGCP.png" alt="Image" height="420" >
</p>

- Overview: Architecture of SGCP (proposed method). SGCP aims to paraphrase an input sentence, while conforming to the syntax of an exemplar sentence (provided along with the input). The input sentence is encoded using the Sentence Encoder to obtain a semantic signal c<sub>t</sub> . The Syntactic Encoder takes a constituency parse tree (pruned at height H) of the exemplar sentence as an input, and produces representations for all the nodes in the pruned tree. Once both of these are encoded, the Syntactic Paraphrase Decoder uses pointer-generator network, and at each time step takes the semantic signal c<sub>t</sub> , the decoder recurrent state s<sub>t</sub> , embedding of the previous token and syntactic signal h<sup>Y</sup><sub>t</sub> to generate a new token. Note that the syntactic signal remains the same for each token in a span (shown in figure above curly braces). The gray shaded region (not part of the model) illustrates a qualitative comparison of the exemplar syntax tree and the syntax tree obtained from the generated paraphrase.

## Dependencies

- Compatible with Pytorch 1.3.0 and Python 3.x
- The necessary packages can be install through requirements.txt

## Setup

To get the project's source code, clone the github repository:

```shell
$ git clone https://github.com/malllabiisc/SGCP
```

Install VirtualEnv using the following (optional):

```shell
$ [sudo] pip install virtualenv
```

Create and activate your virtual environment (optional):

```shell
$ virtualenv -p python3 venv
$ source venv/bin/activate
```

Install all the required packages:

```shell
$ pip install -r requirements.txt
```

Create essential folders in the repository using:

```shell
$ chmod a+x setup.sh
$ ./setup.sh
```

## Resources

#### Dataset

Download the following dataset(s):

[Data]()

Extract and place them in the `SGCP/data` directory

Path: `SGCP/data/<dataset-folder-name>`.

A sample dataset folder might look like
```
data/QQPPos/<train/test/val>/<src.txt/tgt.txt/refs.txt/src.txt-corenlp-opti/tgt.txt-corenlp-opti/refs.txt-corenlp-opti>
```

#### Pre-trained Models:

Download the following pre-trained models for both QQPPos and ParaNMT50m datasets:

[Models]()

Extract and place them in the `SGCP/Models` directory

Path: `SGCP/Models/<dataset_Models>`

#### Evaluation Essentials

Download the evaluation file and place it in `SGCP/src/evaluation` directory

Path: `SGCP/src/evaluation/<apps/data/ParaphraseDetection>`

This contains all the necessary files needed to evaluate the model. It also contains the Paraphrase Detection Score Models for Model-based evaluation.

## Training the model

- For training the model with default hyperparameter settings, execute the following command:
  ```
  python -m src.main -run_name test_run -dataset <DatasetName> -gpu <GPU-ID> -bpe
  ```
  - -run_name: To specify the name of the run for storing model parameters
  - -dataset: Which dataset to train the model on, choose from QQPPos and ParaNMT50m
  - -gpu: For a multi-GPU machine, specify the id of the gpu where you wish to run the code. For a single GPU machine simply use 0 as the ID
  - -bpe: To enable byte-pair encoding for tokenizing data.

- Other hyperparameters can be viewed in src/args.py

## Generation and Evaluation

- For generating paraphrases on the QQPPos dataset, execute the following command:
  ```
  python -m src.main -mode decode -dataset QQPPos -run_name QQP_Models -gpu <gpu-num> -beam_width 1 -max_length 60 -res_file generations.txt
  ```

- Similarly for ParaNMT dataset:
  ```
  python -m src.main -mode decode -dataset ParaNMT50m -run_name ParaNMT_Models -gpu <gpu-num> -beam_width 1 -max_length 60 -res_file generations.txt
  ```

- To evaluate BLEU, ROUGE, METEOR, TED and Prec. scores, first clean the generations:
  - For QQPPos:
  ```
  python -m src.utils.clean_generations -gen_dir Generations/QQP_Models -data_dir data/QQPPos/test
  -gen_file generations.txt
  ```

  - For ParaNMT50m
  ```
  python -m src.utils.clean_generations -gen_dir Generations/ParaNMT_Models -data_dir data/ParaNMT50m/test
  -gen_file generations.txt
  ```

- Since our model generates multiple paraphrases corresponding to different heights of the syntax tree, to select a single generation:
  ```
  python -m src.utils.candidate_selection -gen_dir Generations/QQP_Models
  -clean_gen_file clean_generations.csv -res_file final_paraphrases.txt -crt <SELECTION CRITERIA>
  ```

  - -crt: Criteria to use for selecting a single generation from the given candidates. Choose 'rouge' for ROUGE based selection as given in paper (SGCP-R) and 'maxht' for selecting the generation corresponding to the full height of the tree (SGCP-F)

- Finally, to obtain the scores, run:
  - For QQPPos:
  ```
  python -m src.evaluation.eval -i Generations/QQP_Models/final_paraphrases.txt
  -r data/QQPPos/test/ref.txt -t data/QQPPos/test/tgt.txt
  ```

  - For ParaNMT50m:
  ```
  python -m src.evaluation.eval -i Generations/ParaNMT_Models/final_paraphrases.txt
  -r data/ParaNMT50m/test/ref.txt -t data/ParaNMT50m/test/tgt.txt
  ```

For any clarification, comments, or suggestions please create an issue or contact [ashutosh@iisc.ac.in](http://ashutoshml.github.io)
