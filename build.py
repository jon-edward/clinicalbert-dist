import pathlib
import shutil

import sentence_transformers

out_root = pathlib.Path(__file__).parent



models = [
    ("NeuML/pubmedbert-base-embeddings", out_root / "pubmedbert-base-embeddings", sentence_transformers.SentenceTransformer),
]


def build():
    for model_ident, out_dir, model_type in models:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        
        model = model_type(model_ident)
        model.save(out_dir)
        
        shutil.make_archive(out_dir.name, "zip", out_dir)

if __name__ == "__main__":
    build()
