


import tbv.utils.mseg_interface as mseg_interface


def main() -> None:
	""" """

    logging.info("Running semantic segmentation inference...")

    # cmd = f'bash {mcd_repo_root}/run_ground_imagery.sh {subsampled_slice_extraction_dir} {mseg_semantic_repo_root}'
    # stdout, stderr = run_command(cmd, return_output=True)
    # dump_txt_report(stdout, reports_dir, log_id, report_type_id='MSegInference')
    mseg_interface.run_semantic_segmentation(log_id, slice_extraction_dir, mseg_semantic_repo_root)


if __name__ == "__main__":
	main()