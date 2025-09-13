# MultiLang-ASR-Gh  
Automatic Speech Recognition (ASR) for Ghanaian Languages  

This project develops an ASR system for Ghanaian languages (Akan, Ewe, Dagbani, Dagaare, and Ikposo) using the SpeechData dataset. The work covers dataset preprocessing, deep learning model design, training, and evaluation.  

## Key Steps  

- **Dataset** → SpeechData audio + transcripts (structured across five languages).  
- **Preprocessing** → normalization, MFCC feature extraction, augmentation, and text-to-numeric conversion.  
- **Modeling** → CNNs, RNNs, LSTMs, and Transformer-based architectures with attention.  
- **Training** → hyperparameter tuning, monitoring loss/accuracy, and iterative improvements.  
- **Evaluation** → metrics include Word Error Rate (WER), Character Error Rate (CER), and Accuracy.  
- **Reporting** → visualizations (loss curves, confusion matrices), results, challenges, and future directions.  

The repo provides the full pipeline: from raw audio to a trained ASR model, with documentation to replicate and build on this work.  