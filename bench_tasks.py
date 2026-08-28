"""Flow tasks for the benchmark, derived from
embeddings_pipeline/scripts/cad_tasks_embeddings.py.

Two deliberate differences from the original:

1. `gather_cad_files` reads a frozen file list (env var BENCH_FILELIST) instead
   of walking the directory and shuffling. This makes every configuration
   encode exactly the same parts in the same order, and removes the ~55 s of
   directory-walk time from the measurement.
2. Flow name and output dir come from env vars (BENCH_FLOW_NAME / BENCH_OUT_DIR)
   so concurrent/consecutive configurations never overwrite each other's
   artifacts, and nothing is written into the existing
   GPU1.1/embeddings_pipeline/out tree.

This module MUST stay importable in a fresh interpreter with no side effects
beyond the license call, because HOOPS AI re-imports it in every worker process
(Windows spawn).
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import List

import hoops_ai
from hoops_ai.cadaccess import HOOPSLoader
from hoops_ai.flowmanager import flowtask
from hoops_ai.ml.EXPERIMENTAL import EmbeddingFlowModel
from hoops_ai.storage import DataStorage, PyGGraphStoreHandler
from hoops_ai.storage.helpers import generate_unique_id_from_path

# ---------------------------------------------------------------- license
license_key = os.environ.get("HOOPS_AI_LICENSE")
if not license_key:
    sys.exit("HOOPS_AI_LICENSE environment variable is required.")
hoops_ai.set_license(license_key, validate=True, silent=True)

# ---------------------------------------------------------------- config
FILELIST = os.environ.get("BENCH_FILELIST", "")
flows_outputdir = pathlib.Path(
    os.environ.get("BENCH_OUT_DIR", str(pathlib.Path.cwd() / "out")))
_FLOW_NAME = os.environ.get("BENCH_FLOW_NAME", "BENCH_Embedding")


def get_flow_name() -> str:
    return _FLOW_NAME


flow_name = get_flow_name()

EmbeddingModel = EmbeddingFlowModel(
    result_dir=str(flows_outputdir / "flows" / flow_name),
    log_file=str(flows_outputdir / "flows" / flow_name / "flow.log"),
)


# ---------------------------------------------------------------- tasks
@flowtask.extract(
    name="Gather CAD files from datasources",
    inputs=["cad_datasources"],
    outputs=["cad_dataset"],
    parallel_execution=True,
)
def gather_cad_files(source: str) -> List[str]:
    """Return the frozen benchmark file list (order preserved, no shuffle).

    `source` is ignored when BENCH_FILELIST is set; it is kept in the signature
    because the flow framework passes it positionally.
    """
    if FILELIST:
        listing = pathlib.Path(FILELIST)
        # utf-8-sig, not utf-8: PowerShell 5.1's "Set-Content -Encoding UTF8"
        # prepends a BOM, which would otherwise glue ﻿ onto the first path
        # and make it silently unopenable.
        raw = [ln.strip().lstrip("﻿")
               for ln in listing.read_text(encoding="utf-8-sig").splitlines()
               if ln.strip()]
        if not raw:
            raise RuntimeError(f"BENCH_FILELIST is empty: {listing}")
        # Portable file lists store paths RELATIVE to the corpus root so the same
        # list works on Windows and on the AWS Ubuntu box: set BENCH_FILE_ROOT to
        # wherever the mechcad tree lives. Absolute entries are used as-is.
        file_root = os.environ.get("BENCH_FILE_ROOT", "").strip()
        files = []
        for entry in raw:
            p = pathlib.Path(entry)
            if not p.is_absolute() and file_root:
                p = pathlib.Path(file_root) / entry
            files.append(str(p))
        missing = [f for f in files if not pathlib.Path(f).is_file()]
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {len(files)} files in {listing} do not exist, "
                f"first: {missing[0]!r}")
        return files

    # Fallback: behave like the original task.
    from hoops_ai.storage import CADFileRetriever, LocalStorageProvider
    retriever = CADFileRetriever(
        storage_provider=LocalStorageProvider(directory_path=source),
        formats=[".stp", ".step", ".iges", ".igs", ".sldprt", ".SLDPRT"],
    )
    return retriever.get_file_list()


@flowtask.transform(
    name="Extracting CAD ML input for EmbeddingFlowModel",
    inputs=["cad_dataset"],
    outputs=["cad_files_encoded"],
    parallel_execution=True,
)
def encode_data_for_ml_training(cad_file: str,
                                cad_loader: HOOPSLoader,
                                storage: DataStorage) -> str:
    """Identical to the shipped demo task -- do not optimise, we are timing it."""
    facecount, edgecount = EmbeddingModel.encode_cad_data(cad_file, cad_loader, storage)

    graph_storage = PyGGraphStoreHandler()

    item_no_suffix = pathlib.Path(cad_file).with_suffix("")
    hash_id = generate_unique_id_from_path(str(item_no_suffix))
    graph_output_path = flows_outputdir / "flows" / flow_name / "graph_data" / f"{hash_id}.pt"
    graph_output_path.parent.mkdir(parents=True, exist_ok=True)

    EmbeddingModel.convert_encoded_data_to_graph(storage, graph_storage, str(graph_output_path))

    storage.save_metadata("Item", str(cad_file))
    storage.save_metadata("source", "SCREW_BENCH")

    return storage.get_file_path("")
