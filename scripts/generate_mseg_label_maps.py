"""Generate label maps using a MSeg semantic segmentation model.

MSeg label maps are only used if `filter_ground_with_semantics` is true in the BEVRenderingConfig.
"""

import logging
from pathlib import Path

import click

import tbv.utils.mseg_interface as mseg_interface


def generate_mseg_label_maps(data_root: Path, mseg_semantic_repo_root: Path) -> None:
    """

    Args:
        data_root: Path to local directory where the TbV logs are stored.
        mseg_semantic_repo_root: Path to where mseg_semantic repo is stored.
    """
    logging.info("Running semantic segmentation inference...")
    log_id = []

    for log_id in log_ids:
        mseg_interface.run_semantic_segmentation(
            log_id=log_id, log_dir=log_dir, mseg_semantic_repo_root=mseg_semantic_repo_root
        )


@click.command(help="Generate label maps using a MSeg semantic segmentation model.")
@click.option(
    "--mseg_semantic_repo_root",
    help="Path to where mseg_semantic repo is stored.",
    default="/Users/johnlambert/Downloads/mseg-semantic",
    type=click.Path(exists=True),
)
@click.option(
    "--data-root",
    required=True,
    help="Path to local directory where the TbV logs are stored.",
    type=click.Path(exists=True),
)
def run_generate_mseg_label_maps(mseg_semantic_repo_root: str, data_root: str) -> None:
    """ """
    generate_mseg_label_maps(data_root=Path(data_root), mseg_semantic_repo_root=Path(mseg_semantic_repo_root))


if __name__ == "__main__":
    run_generate_mseg_label_maps()
