## Data Preprocessing: ACDC

DataPreprocessing/image_paths.txt: files with paths to the files in the each patient's directory
I distribute the ACDC dataset to 80%, 10%, 10% for training, validation and testing respectively.
Additionnally I make sure that each partition has an equal distribution of patients according to their pathology (NOR, DCM, HCM, MINF, RV). Check ACDC dataset to learn more about this.

First download the ACDC dataset from this link and create the following structure.

```Shell
├── datasets
    ├── ACDC
    ├── ACDC_processed
        ├── training
        ├── validation
        ├── testing
```

```Shell
 mkdir datasets
 cd datasets/
 mkdir ACDC_processed
 cd ACDC_processed/
 mkdir training/
 mkdir validation/
 mkdir testing/
 cd ../../DataPreprocessing/
```
ACDC_Preprocess_Script.py: converts 4D MRI cardiac volumes to temporal sequences. --acdc_folder argument takes the full path of the folder where ACDC is stored, --acdc_processed_folder takes the full path of the folder where to store processed dataset (ACDC_processed).
```Shell
python3 ACDC_Preprocess_Script.py  --acdc_folder=/home/data/ACDC/ --acdc_processed_folder=/home/kevin/manal/RAFT/datasets/ACDC_processed/
```