import pathlib
import shutil

import transformers

out_root = pathlib.Path(__file__).parent

models = [
    ("emilyalsentzer/Bio_ClinicalBERT", out_root / "bio_clinicalbert", transformers.AutoModel),
    ("jon-t/Bio_ClinicalBERT_QA", out_root / "bio_clinicalbert_qa", transformers.AutoModelForQuestionAnswering),
]


def build():
    for model_ident, out_dir, model_type in models:
        if out_dir.exists():
            shutil.rmtree(out_dir)

        model = model_type.from_pretrained(model_ident)
        model.save_pretrained(out_dir)

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_ident)
        tokenizer.save_pretrained(out_dir)

        shutil.make_archive(out_dir.name, "zip", out_dir)

if __name__ == "__main__":
    build()
