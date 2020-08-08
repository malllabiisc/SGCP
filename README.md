# Syntax-Guided Controlled Generation of Paraphrases

Source code for [TACL 2020](https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00318) paper: [Syntax-Guided Controlled Generation of Paraphrases](https://arxiv.org/pdf/2005.08417.pdf)

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

- Download the following dataset(s): [Data](https://indianinstituteofscience-my.sharepoint.com/:u:/g/personal/ashutosh_iisc_ac_in/ER-roD8qRXFCsyJwbOHOVPgBs-VTKNmkNLzQvM0cLtvBhw?e=a0dOid)
- Extract and place them in the `SGCP/data` directory

Path: `SGCP/data/<dataset-folder-name>`.

A sample dataset folder might look like
```
data/QQPPos/<train/test/val>/<src.txt/tgt.txt/refs.txt/src.txt-corenlp-opti/tgt.txt-corenlp-opti/refs.txt-corenlp-opti>
```

#### Pre-trained Models:

- Download the following pre-trained models for both QQPPos and ParaNMT50m datasets: [Models](https://indianinstituteofscience-my.sharepoint.com/:u:/g/personal/ashutosh_iisc_ac_in/Ed5IT05LTaFNhVFweWuUE8MBnRCkSAJwSotrAhzT_2lL5w?e=3hOrSI)
- Extract and place them in the `SGCP/Models` directory

Path: `SGCP/Models/<dataset_Models>`

#### Evaluation Essentials

- Download the evaluation file: [evaluation](https://indianinstituteofscience-my.sharepoint.com/:u:/g/personal/ashutosh_iisc_ac_in/EQVo8LOkzlFKhAlfjMnZc20BEFfAzvemc9TdBONtBSpmGQ?e=q3J4NS)
- Extract and place it in `SGCP/src/evaluation` directory
- Give executable permissions to `SGCP/src/evaluation/apps/multi-bleu.perl`

Path: `SGCP/src/evaluation/<apps/data/ParaphraseDetection>`

This contains all the necessary files needed to evaluate the model. It also contains the Paraphrase Detection Score Models for Model-based evaluation.

## Training the model

- For training the model with default hyperparameter settings, execute the following command:
  ```
  python -m src.main -mode train -run_name testrun -dataset <DatasetName> -gpu <GPU-ID> -bpe
  ```
  - -run_name: To specify the name of the run for storing model parameters
  - -dataset: Which dataset to train the model on, choose from QQPPos and ParaNMT50m
  - -gpu: For a multi-GPU machine, specify the id of the gpu where you wish to run the code. For a single GPU machine simply use 0 as the ID
  - -bpe: To enable byte-pair encoding for tokenizing data.

- Other hyperparameters can be viewed in src/args.py

## Generation and Evaluation

- For generating paraphrases on the QQPPos dataset, execute the following command:
  ```
  python -m src.main -mode decode -dataset QQPPos -run_name QQP_Models -gpu <gpu-num> -beam_width 10 -max_length 60 -res_file generations.txt
  ```

- Similarly for ParaNMT dataset:
  ```
  python -m src.main -mode decode -dataset ParaNMT50m -run_name ParaNMT_Models -gpu <gpu-num> -beam_width 10 -max_length 60 -res_file generations.txt
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

## Custom Dataset Processing
  Preprocess and parse the data using the following steps.

  1. Move the contents of your custom dataset in the data/ directory, with files arranged something like this:
      - data
        - Custom_Dataset
          - train
              - src.txt
              - tgt.txt
          - val
              - src.txt
              - tgt.txt
              - ref.txt
          - test
              - src.txt
              - tgt.txt
              - ref.txt

       Here, src.txt contains the source sentences, tgt.txt contains exemplars and ref.txt contains the paraphrases.
  2. Construct a byte-pair codes file which will be used to generate byte pair encodings of the dataset. From the main directory of this repo, run:
    ```
    subword-nmt learn-bpe  <data/Custom_Dataset/train/src.txt> data/Custom_Dataset/train/codes.txt
    ```
    Note: [Optional] Generate codes from both src.txt and tgt.txt - For that first concatenate the two files and replace src.txt with the name of the concatenated file in the command.

  3. Parse the data files using stanford corenlp. First start a corenlp server by executing the following commands:
  ```
  cd src/evaluation/apps/stanford-corenlp-full-2018-10-05
  java -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse -parse.model /edu/stanford/nlp/models/srparser/englishSR.ser.gz -status_port <PORT_NUMBER> -port <PORT_NUMBER> -timeout 15000
  ```

  4. Finally run the parser on the text files.
  ```
  cd <PATH_TO_THIS_REPO>
  python -m src.utils.con_parser -infile data/Custom_Dataset/train/src.txt -codefile data/Custom_Dataset/train/codes.txt -port <PORT_NUMBER (where the corenlp server is running, from step 3)> -host localhost
  ```
  This will generate a file in train folder called src.txt-corenlp-opti
  Run this for all other files i.e. tgt.txt in train folder, src.txt, tgt.txt, ref.txt in val folder and similarly for the files in test folder.

### Citing:
Please cite the following paper if you use this code in your work.

```bibtex
@article{sgcp2020,
author = {Kumar, Ashutosh and Ahuja, Kabir and Vadapalli, Raghuram and Talukdar, Partha},
title = {Syntax-Guided Controlled Generation of Paraphrases},
journal = {Transactions of the Association for Computational Linguistics},
volume = {8},
number = {},
pages = {330-345},
year = {2020},
doi = {10.1162/tacl\_a\_00318},
URL = { https://doi.org/10.1162/tacl_a_00318 },
eprint = { https://doi.org/10.1162/tacl_a_00318 }
}
```

For any clarification, comments, or suggestions please create an issue or contact [ashutosh@iisc.ac.in](http://ashutoshml.github.io)
