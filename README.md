---
library_name: transformers
tags: []
---

# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->



## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This is the model card of a 🤗 transformers model that has been pushed on the Hub. This model card has been automatically generated.

- **Developed by:** Tanaya Atmaram Kambli
- **Model type:** ASR (Wave2Vec 960H)
- **Language(s) (NLP):** English
- **License:** Unsure
- **Finetuned from model [optional]:** Wave2Vec 960H

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** https://huggingface.co/Tanaya25/ASR-Common_speech_11_10/tree/main


## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
The model is an ASR model for converting Speech to Text. Trained with English speech using CommonVoice 11.10

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->



### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

Transcibing english lectures, Voice assistants

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

SHould not be used for medical or legal transcriptions

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The model has been fine tune on 10% of CommonVoice 11.10 and hence, contains biases. WER is at 26%.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

This model can be used a a strating point to further train on the rest of the CommonVoice data

## How to Get Started with the Model

Use the code below to get started with the model.


```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Trainer, TrainingArguments

# Load the model and processor
processor = Wav2Vec2Processor.from_pretrained('Tanaya25/ASR-Common_speech_11_10')
model = Wav2Vec2ForCTC.from_pretrained('Tanaya25/ASR-Common_speech_11_10')

# Set up the training arguments
eval_args = TrainingArguments(
    output_dir="./output",  # Output directory for evaluation results
    per_device_eval_batch_size=8,  # Batch size for evaluation per device
    logging_dir='./logs',  # Directory for logging
    logging_steps=50,  # Log every 50 steps
    do_eval=True,  # Ensure evaluation occurs
    run_name="my_evaluation_run",
    report_to="none"  # Add this to disable wandb reporting
)

trainer = Trainer(
    model=model,
    args=eval_args,
    processor=processor.feature_extractor,
    compute_metrics=compute_metrics  # Custom metrics function if needed
)

# Evaluate on test dataset
results = trainer.evaluate(test_dataset)
print(results)


## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

Dataset: CommonVoice 11.10 English (10% subset after shuffling)
Common Voice 11.10 is a large, open-source speech dataset released by Mozilla, designed for training and improving automatic speech recognition (ASR) systems. It includes over **30,000 hours of recorded speech** from contributors worldwide, covering **100+ languages**. The dataset consists of **audio clips**, corresponding **text transcriptions**, **speaker demographics** (such as age, gender, and accent), and **metadata**. It is widely used for multilingual ASR, speech-to-text models, and linguistic research.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

- 

#### Preprocessing [optional]

- Resampling to 16kHz
- Populating audio data 'array' to 'input_values' col
- Populating transcription data 'sentence' to 'labels' col
- data cleaning: rows in which audio all values are below 1e-6 and any transciptions which are null are removed
- converting labels to upper case
- removing punctuations (only have [A-Z' ])
- split to train (80%), val (10%), test (10%)
- created vocabulary using training data lables: 26 characters (A-Z), apostrophe ('), space change to '|' for easier identification, PAD token and UNK token. So total 30 tokens
- Normalizing during feature extraction


#### Training Hyperparameters

- **Training regime:**

1. output_dir: /content/drive/MyDrive/Speech_to_text/model_checkpoints
2. batch_size: 8
3. train_epochs: 10
4. fp16: True
5. gradient_checkpointing: True
6. logging_dir: ./logs
7. learning_rate: 3e-5
8. weight_decay: 0.005
9. dataloader_num_workers: 8
10. Optimizer: AdamW
11. metric: WER
    
(set when using Early Stopping)
13. metric_for_best_model: eval_loss 
14. greater_is_better: False
15. load_best_model_at_end: True

<!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

1. Dataset size: Train = 75898
3. Speed: Training = 3.5 hours on A100 with 40 GB GPU RAM. 3.8 Epcochs trained before reaching convergence. 
         
   
## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

1. Data =  Val = 9487
2. Metrics = WER (around 2.6 achieved)
   
#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

1. Data =  Test = 9488
2. Metrics = WER (around 2.6 achieved)
3. Time: around 10 minutes

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

The evaluation of this model may be disaggregated by the following factors:

Accent or Dialect: Performance might vary depending on the speaker's accent or regional dialect. The model may perform better on certain accents and less well on others.

Audio Quality: The model's performance can be affected by the quality of the audio input, including noise, distortion, or low-quality recordings.

Speaker Demographics: Variations in the performance may be observed based on speaker demographics such as age, gender, or background.

Environmental Noise: The presence of background noise or echo in the audio could affect the model's ability to accurately transcribe speech.

Speech Speed: The model might have different performance levels for fast versus slow speakers.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

The primary evaluation metric used in this model is Word Error Rate (WER).

WER is a common metric for evaluating speech-to-text models, as it measures the difference between the predicted transcript and the ground truth. It is calculated by comparing the number of substitutions, deletions, and insertions required to transform the predicted output into the reference text. A lower WER indicates better model performance, as it signifies fewer errors in the transcription process.

This metric is chosen because it directly reflects the model's ability to correctly transcribe speech into text, which is the core task of the model.

### Results

Here's a table comparing the predictions (`pred_str`) and the labels (`label_str`):

| **Index** | **Prediction**                                                      | **Label**                                                             |
|-----------|---------------------------------------------------------------------|-----------------------------------------------------------------------|
| 1         | THE HEROES CRESSE TO THE WHIM SUCCEEDSS                              | THE HERO'S QUEST TO THE RIM SUCCEEDS                                  |
| 2         | SHORTLY AFTER BEING DRAFTED HE WAS RELEASED FROM THE COMPANYS        | SHORTLY AFTER BEING DRAFTED HE WAS RELEASED FROM THE COMPANY          |
| 3         | THUS SHE WAS ARRESTED AND STAYED IMPRISONED FOR A MONTHS             | THUS SHE WAS ARRESTED AND STAYED IN PRISON FOR A MONTH                |
| 4         | THE CHURCH WAS REBUILT ON THE PRESENT SITES                          | THE CHURCH WAS REBUILT ON THE PRESENT SITE                            |
| 5         | THE COMMON NAME OF THESE FISH IS RELATED TO THEIR BIZAR TUBILAR EYESS | THE COMMON NAME OF THESE FISH IS RELATED TO THEIR BIZARRE TUBULAR EYES|
| 6         | A DEFHJAM RECORDINGSLOGO WAS ALSO PRESENT ON ITS FOLLOWER ALBUMS     | A DEF JAM RECORDINGS LOGO WAS ALSO PRESENT ON ITS FOLLOWUP ALBUM      |
| 7         | BY SOME ACCOUNTS THE PROBLEM IS NAMED AFTER A SOLITURGANES           | BY SOME ACCOUNTS THE PROBLEM IS NAMED AFTER A SOLITAIRE GAME         |
| 8         | THE CONSTRUCTION WHILE NOT FUNCTORIAL IS A FUNDAMENTAL TOOL IN SCHEME THEORYS | THE CONSTRUCTION WHILE NOT FUNCTORIAL IS A FUNDAMENTAL TOOL IN SCHEME THEORY |
| 9         | HE WAS IMPRISOND AT CAMP PADAWAWA IN ONTARIO UNTIL THE END OF THE WARS | HE WAS IMPRISONED AT CAMP PETAWAWA IN ONTARIO UNTIL THE END OF THE WAR|
| 10        | HE WRITES NEARLY TWENTY OPERAS COMIC OPERAS AND OPERATUSS            | HE WRITES NEARLY TWENTY OPERAS COMICOPERAS AND OPERETTAS              |

This table shows the predictions alongside the correct labels, which will help in evaluating model performance by comparing the errors.

{'eval_loss': 0.25646209716796875, 'eval_model_preparation_time': 0.0032, 'eval_wer': 0.26040177331591563, 'eval_runtime': 573.1095, 'eval_samples_per_second': 16.555, 'eval_steps_per_second': 2.069}

#### Summary

This model is a speech-to-text model trained using the CommonVoice dataset. It converts spoken language into written text using a Wav2Vec2-based architecture, which is fine-tuned to handle a wide range of speakers and accents.

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

This model leverages Wav2Vec2, a transformer-based architecture, to extract speech features and make predictions. It is particularly effective for tasks involving noisy environments and diverse accents due to its ability to learn from large, unlabelled audio datasets.

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA A100-SXM4-40GB
- **Hours used:** Around 45 hours for the entire preprocessing till training to get expected results
- **Cloud Provider:** Google Cloud
- **Compute Region:** us-east1
- **Carbon Emitted:**  6.66 kg CO2 eq.

## Technical Specifications [optional]

### Model Architecture and Objective

The model is based on the Wav2Vec2 architecture, which uses a pre-trained feature extractor to encode raw audio into high-level features, followed by a transformer decoder to produce text sequences. The primary objective is to transcribe audio to text with high accuracy across various accents and environments.

### Compute Infrastructure

[More Information Needed]

#### Hardware

GPU: NVIDIA A100-SXM4-40GB
RAM: System: 83.5 GB, GPU: 40 GB
Storage: SSDs (around 250 GB)

#### Software

Libraries: Hugging Face Transformers, PyTorch, Datasets
Python Version: 3.8 or higher
Operating System: Ubuntu 20.04

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

@inproceedings{baevski2020wav2vec,
  title={Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations},
  author={Alexei Baevski and Henry Zhou and Abdelrahman Mohamed and Michael Auli},
  booktitle={NeurIPS},
  year={2020}
}


