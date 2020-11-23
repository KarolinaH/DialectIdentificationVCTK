# DialectIdentificationVCTK
End-to-end CNN-LSTM English dialect identification (DID) on VCTK dataset

If running for the first time, ensure you have downloaded all VCTK data.

The code is run from home/scripts/, assuming all files above are saved in home/scripts/ and data (audio files) are in home/data/.

First, run `python feature_extraction.py` to extract either mel filterbank (fbank) or MFCC features (specify in main part of script).
Once all wavs are padded and features have been extracted, run `python train.py`. Hyperparameters for this can be changed in hparams.py, and changes to the model architecture can be made in models.py.

Best performance on VCTK dataset using this code was 48% error. I'd recommend using a different dataset and/or data augmentation on the VCTK dataset to provide more training data.

The dissertation this was for may be uploaded at a later point.
