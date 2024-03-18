# Voice Transcriber with Lightning.AI

Welcome to my project! I am leveraging the power of Lightning.AI, a state-of-the-art cloud computing and IDE solution, to develop a voice transcriber model. Our model takes French speech as input and generates text files as output.

## Our Journey

Our journey begins with an educational exploration of the Lightning PyTorch library. We'll dive deep into its functionalities and capabilities to understand how we can best utilize it for our project.

Next, we'll decide on a model architecture that is compatible with our requirements. We're looking for a model that not only transcribes speech to text but does so in real-time, handling the speed and flow of natural conversation.

Finally, we'll choose a pre-trained model that's suitable for our task. The goal is to leverage existing knowledge and avoid reinventing the wheel, speeding up our development process and improving the accuracy of our transcriptions.

## Multi-Speaker Differentiation

A unique feature of our model is its ability to differentiate between multiple speakers. This is a complex task, but we're up for the challenge. We'll be using the open-source library PyAnote to assist us in this endeavor.

Stay tuned for more updates as we progress in our exciting journey!

---

To start your project, you can follow these steps:

1. **Environment Setup**: Set up a Python environment using  conda. Install necessary libraries like PyTorch Lightning, PyTorch, and any other libraries you think you might need.

2. **Data Collection**: Collect or find a dataset of French speeches. You might need to look for datasets online or create one yourself.
Biggest Non-Commercial French Language Dataset The SIWIS French Speech Synthesis Database includes high-quality French speech recordings and associated text files, aimed at building TTS systems, investigating multiple styles, and emphasis. Various sources such as parliament debates and novels were uttered by a professional French voice talent. A subset of the database contains emphasized words in many different contexts.

== Syntax of file names: ==
* For part 1,2,3 and 5:
  [style]_[type]_[session]_[id].[ext] where:
  - style = neutral (neut) / emphasised (emph) / expressive (expr)
  - type = parliament (parl) / book (book) / siwis (siwis) / sus (sus) /
chapter (chap)
  - session = 2 digits
  - id = 4 digits
  - ext = wav / txt / lab
* part4:
  chap_full.[ext]
  
```
SiwisFrenchSpeechSynthesisDatabase
├── wavs
│   ├── part1
│   │   ├── neut_parl_s01_0001.wav
│   │   ├── ...
│   ├── part2
│   │   ├── ...
│   ├── part3
│   │   ├── ...
│   ├── part4
│   │   ├── chap_full.wav
│   ├── part5
│   │   ├── ...
└── text
    ├── part1
    │   ├── neut_parl_s01_0001.txt
    │   ├── ...
    ├── part2
    │   ├── ...
    ├── part3
    │   ├── ...
    ├── part4
    │   ├── chap_full.txt
    └── part5
        ├── ...
```

Features: 9750 utterances from various sources more than ten hours of speech data freely available
3. **Data Preprocessing**: Preprocess your data. This might involve converting audio files to a suitable format, normalizing audio levels, segmenting speeches into smaller chunks, etc.
Pytorch dataloader streaming data as solution and HDF5 as data compressor for both voice and text.

4. **Model Selection**: Research and select a suitable pre-trained model for speech recognition. You might want to look into models specifically trained for French language. Speech Brain was selected

5. **Training**: Use PyTorch Lightning to train your model on your dataset.

6. **Evaluation**: Evaluate your model's performance. This might involve creating a test set of speeches and checking the accuracy of the transcriptions.

7. **Integration of PyAnote**: Integrate PyAnote for speaker diarization (differentiating between speakers).

8. **Testing**: Test the entire system together. Make sure the transcription is fast enough for real-time speech and that the speaker diarization is working correctly.

9. **Iteration**: Based on your tests, you might need to go back to previous steps and make improvements.

10. **Documentation**: Document your code and your process. This will be helpful for others who want to understand your project, and for you if you need to revisit the project in the future.

Remember, this is a complex project and it's okay to take it one step at a time. Good luck!
