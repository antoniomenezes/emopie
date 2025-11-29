# EMOPIE

### an EMOtion classifier from the Perspective of Intratextual Entities

EMOPIE is a **multi-label transformer-based classifier** designed to
identify **emotions expressed by specific entities (experiencers)**
within a text.\
Traditional emotion classification typically focuses on sentence-level
or author-level sentiment. EMOPIE instead models **entity-dependent
emotional viewpoints**, enabling emotion detection even when multiple
entities appear in the same sentence or passage.

This repository contains the first public implementation of EMOPIE,
including preprocessing, model architecture, training scripts, and
evaluation pipeline (the main classifier was configured for the **x-enVENT** dataset).\
Additional code was tested with preprocessed versions of ABBE and REMAN, 
respectively emopie_clf_abbe.py and emopie_clf_reman.py.

------------------------------------------------------------------------

## Features

-   **Entity-aware emotion classification**
-   **Multi-label support**
-   **Transformer-based architecture**
-   **Reusable and extensible pipeline**
-   **Clean training script**

------------------------------------------------------------------------

## Repository Structure

    emopie/
    ├── data/
    ├── image/
    ├── emopie_clf_abbe.py
    ├── emopie_clf_reman.py
    ├── emopie_clf_x_envent.py
    ├── README.md
    ├── LICENSE
    └── .gitignore

------------------------------------------------------------------------

## Installation

``` bash
git clone https://github.com/antoniomenezes/emopie
cd emopie
python -m venv emopie-env
source emopie-env/bin/activate   # Windows: emopie-env\Scripts\activate
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Model Architecture

EMOPIE merges:

1.  **Global \[CLS\] pooled output** (Global Context)
2.  **Entity span embedding** (Localized Context)

The representations are concatenated and passed to a multi-label
classifier.

![EMOPIE model architecture](image/architecture.png)

------------------------------------------------------------------------

## Dataset

The implementation supports the **x-enVENT** dataset.

------------------------------------------------------------------------

## Usage

### Train

``` bash
python emopie_clf_x_envent.py
```

### Evaluate

Automatically performed after training.

------------------------------------------------------------------------

## Metrics (results are presented)

-   Micro/Macro F1\
-   Precision & Recall\
-   Per-emotion results

------------------------------------------------------------------------

## Citation

    Menezes, A.M.A. Moreira, V.P. Emotion Classification from the Perspective of Intratextual Entities. (2025).

------------------------------------------------------------------------

## License

Apache 2.0 License.
