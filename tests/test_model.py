from __future__ import annotations

import scvi

from embedding_scvi import EmbeddingSCVI


def test_model_init(n_genes: int = 100, n_batches: int = 2):
    adata = scvi.data.synthetic_iid(n_genes=n_genes, n_batches=n_batches)
    EmbeddingSCVI.setup_anndata(
        adata,
        categorical_covariate_keys=["batch", "labels"],
    )

    model = EmbeddingSCVI(adata)
    model.train(max_epochs=1)

    bdata = scvi.data.synthetic_iid(n_genes=n_genes - 10, n_batches=n_batches + 2)
    bdata = EmbeddingSCVI.prepare_finetune_anndata(bdata, model)
    model = EmbeddingSCVI.load_finetune_anndata(bdata, model)
    model.initialize_finetune_embeddings()

    model.train(max_epochs=1, accelerator="cpu")
