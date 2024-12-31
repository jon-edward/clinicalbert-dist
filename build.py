import pathlib
import shutil

import transformers

MODEL_IDENT = "emilyalsentzer/Bio_ClinicalBERT"

MODEL_DIR = pathlib.Path(__file__).parent / "bio_clinicalbert-model"
TOKENIZER_DIR = pathlib.Path(__file__).parent / "bio_clinicalbert-tokenizer"


def build():

    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)

    if TOKENIZER_DIR.exists():
        shutil.rmtree(TOKENIZER_DIR)

    model = transformers.AutoModel.from_pretrained(MODEL_IDENT)
    model.save_pretrained(MODEL_DIR)

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_IDENT)
    tokenizer.save_pretrained(TOKENIZER_DIR)

    shutil.make_archive(MODEL_DIR.name, "zip", MODEL_DIR)
    shutil.make_archive(TOKENIZER_DIR.name, "zip", TOKENIZER_DIR)

if __name__ == "__main__":
    build()
