import pathlib
import shutil

import transformers

MODEL_IDENT = "emilyalsentzer/Bio_ClinicalBERT"

OUT_DIR = pathlib.Path(__file__).parent / "bio_clinicalbert"


def build():

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    model = transformers.AutoModel.from_pretrained(MODEL_IDENT)
    model.save_pretrained(OUT_DIR)

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_IDENT)
    tokenizer.save_pretrained(OUT_DIR)

    shutil.make_archive(OUT_DIR.name, "zip", OUT_DIR)

if __name__ == "__main__":
    build()
