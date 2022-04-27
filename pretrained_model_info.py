
"""Metadata for pretrained models, trained on TbV data."""

# each model is named as train_ckpt.pth
MODEL_METADATA_DICT = {

    'b153d124-814c-4beb-b832-999a5f582c5e': {
        "release_notes": "jan 13, 2021, fix data augmentation bug, 20 epochs, early fusion",
        "yaml_filename": "train_2021_01_11_egoview_config_earlyfusion.yaml",
        "train_results_json_filename": "results-2021_01_13_12_56_06-train_2021_01_11_egoview_config_earlyfusion.json",
        "prior_dirname": "2021_01_13_12_56_06"
    },

    '9b170dcf-6ba8-41d0-9ff2-fc3faf92e514': {
        "release_notes": "jan 13, fixed data augmentation bug, 60 epochs, early fusion",
        "yaml_filename": "train_2021_01_12_egoview_config_earlyfusion_v1.yaml",
        "train_results_json_filename": "results-2021_01_13_14_58_48-train_2021_01_12_egoview_config_earlyfusion_v1.json",
        "prior_dirname": "2021_01_13_14_58_48"
    },

    '18394fe1-c544-45c4-a23d-4448d891d029': {
        "release_notes": "jan 14, fixed data aug bug, 60 epochs, late fusion",
        "yaml_filename": "train_2021_01_14_egoview_config_earlyfusion_v1.yaml",
        "train_results_json_filename": "results-2021_01_14_01_31_25-train_2021_01_14_egoview_config_earlyfusion_v1.json",
        "prior_dirname": "2021_01_14_01_31_25"
    },

    '1a090b7b-29aa-4799-a5d8-ebe7f5b82a83': {
        "release_notes": "jan 15, multiple negatives per, 60 epoch, early fusion",
        "yaml_filename": "train_2021_01_14_egoview_config_earlyfusion_v2.yaml",
        "train_results_json_filename": "results-2021_01_14_23_35_59-train_2021_01_14_egoview_config_earlyfusion_v2.json",
        "prior_dirname": "2021_01_14_23_35_59"
    },

    'b5a2198c-5368-4feb-85c5-843a5646ecfa': {
        "release_notes": "Ego-view, map-only",
        "yaml_filename": "train_2021_01_16_egoview_config_earlyfusion_v1.yaml",
        "train_results_json_filename": "results-2021_01_16_01_51_45-train_2021_01_16_egoview_config_earlyfusion_v1.json",
        "prior_dirname": "2021_01_16_01_51_45"
    },

    '36b24988-5c54-46b4-9c22-cd48f70ae9f6': {
        "release_notes": "jan 17, early fusion, 120 epochs",
        "yaml_filename": "train_2021_01_17_egoview_config_earlyfusion_v1.yaml",
        "train_results_json_filename": "results-2021_01_17_03_30_13-train_2021_01_17_egoview_config_earlyfusion_v1.json",
        "prior_dirname": "2021_01_17_03_30_13"
    },

    'd70cefa6-ab67-4aa7-b288-c8baa471a04d': {
        "release_notes": "jan 20 , early fusion, BEV fixed bug, 20 epochs, resnet 18",
        "yaml_filename": "train_2021_01_20_lidar_rgb_corr_config_earlyfusion_resnet18_fixeddataug_bug.yaml",
        "train_results_json_filename": "results-2021_01_20_01_24_50-train_2021_01_20_lidar_rgb_corr_config_earlyfusion_resnet18_fixeddataug_bug.json",
        "prior_dirname": "2021_01_20_01_24_50"
    },

    '0589cca2-72aa-4626-9a05-af60eeea5fb6': {
        "release_notes": "jan 20, early fusion, resnet 50, 30 epochs, BEV, fixed bug",
        "yaml_filename": "train_2021_01_17_lidar_rgb_corr_config_earlyfusion_resnet18_fixeddataug_bug.yaml",
        "train_results_json_filename": "results-2021_01_20_04_25_30-train_2021_01_17_lidar_rgb_corr_config_earlyfusion_resnet18_fixeddataug_bug.json",
        "prior_dirname": "2021_01_20_04_25_30"
    },

    '71f2cc98-b634-4e1e-aefc-7826b4fc8b6f': {
        "release_notes": "jan 21, see if rotation helps improve generalization at all",
        "yaml_filename": "train_2021_01_21_lidar_rgb_corr_config_earlyfusion_resnet18_fixeddataug_bug.yaml",
        "train_results_json_filename": "results-2021_01_21_01_25_17-train_2021_01_21_lidar_rgb_corr_config_earlyfusion_resnet18_fixeddataug_bug.json",
        "prior_dirname": "2021_01_21_01_25_17"
    },

    '6b75002f-9b38-4eaf-a017-d8ae30d1bd9e': {
        "release_notes": "448 x 448 , training was aborted at 24 epochs",
        "yaml_filename": "train_2021_01_21_lidar_rgb_corr_config_earlyfusion_resnet18_fixeddataug_bug_v2.yaml",
        "train_results_json_filename": "results-2021_01_21_18_17_26-train_2021_01_21_lidar_rgb_corr_config_earlyfusion_resnet18_fixeddataug_bug_v2.json",
        "prior_dirname": "2021_01_21_18_17_26"
    },

    'e3411e4e-87a6-4539-9ebb-1fcd6f99e601': {
        "release_notes": "resnet50 on egoview, early fusion of sensor + map",
        "yaml_filename": "train_2021_01_22_egoview_config_earlyfusion_v1.yaml",
        "train_results_json_filename": "results-2021_01_22_20_30_02-train_2021_01_22_egoview_config_earlyfusion_v1.json",
        "prior_dirname": "2021_01_22_20_30_02"
    },

    '8ee8e4e8-c1be-489e-bfa8-8cdce4de0706': {
        "release_notes": "448 x 448 with resnet-50, bev",
        "yaml_filename": "train_2021_01_21_lidar_rgb_corr_config_earlyfusion_resnet18_fixeddataug_bug_v2.yaml",
        "train_results_json_filename": "results-2021_01_24_04_11_27-train_2021_01_21_lidar_rgb_corr_config_earlyfusion_resnet18_fixeddataug_bug_v2.json",
        "prior_dirname": "2021_01_24_04_11_27"
    },

    '9dcfaa3f-a6af-4295-93ca-24a84d6b9c2d': {
        "release_notes": "egoview 224x224 with semantic early fusion, larger batch size",
        "yaml_filename": "train_2021_01_25_egoview_w_labelmap_config_earlyfusion_v1.yaml",
        "train_results_json_filename": "results-2021_01_25_19_03_19-train_2021_01_25_egoview_w_labelmap_config_earlyfusion_v1.json",
        "prior_dirname": "2021_01_25_19_03_19"
    },

    '8e92e79b-22fa-42bb-b23e-39b79a949e3f': {
        "release_notes": "egoview 224x224 with semantic early fusion, smaller batch size",
        "yaml_filename": "train_2021_01_24_egoview_w_labelmap_config_earlyfusion_v1.yaml",
        "train_results_json_filename": "results-2021_01_25_15_07_25-train_2021_01_24_egoview_w_labelmap_config_earlyfusion_v1.json",
        "prior_dirname": "2021_01_25_15_07_25"
    },

    'b3ef41e8-db72-4e12-808d-353e4cd54280': {
        "release_notes": "egoview 224x224 with only semantic + map, no sensor info",
        "yaml_filename": "train_2021_01_30_egoview_labelmap_map_pair_config_earlyfusion_v1.yaml",
        "train_results_json_filename": "results-2021_01_31_01_51_50-train_2021_01_30_egoview_labelmap_map_pair_config_earlyfusion_v1.json",
        "prior_dirname": "2021_01_31_01_51_50"
    },

    '2ab39838-e4ac-45c0-9dc5-92ed63295966': {
        "release_notes": "bev with semantics, NOT CONVERGED ALL THE WAY!",
        "yaml_filename": "train_2021_02_08_seamseg_bev_config_train_v1.yaml",
        "train_results_json_filename": "results-2021_02_09_03_30_43-train_2021_02_08_seamseg_bev_config_train_v1.json",
        "prior_dirname": "2021_02_09_03_30_43"
    },

    '6d3bfc13-1da4-49f0-bf6d-f6b1fc9647d8': {
        "release_notes": "dropout either map or semantics, 100% prob? unintuitive Experiment (best model)",
        "yaml_filename": "train_2021_03_01_egoview_w_labelmap_config_earlyfusion_dropout_v1.yaml",
        "train_results_json_filename": "results-2021_03_01_19_06_29-train_2021_03_01_egoview_w_labelmap_config_earlyfusion_dropout_v1.json",
        "prior_dirname": "2021_03_01_19_06_29"
    },

    'eb046d7a-95d7-463c-9f90-f5ac7f72b8c2': {
        "release_notes": "dropout either semantics or sensor, 100% prob",
        "yaml_filename": "train_2021_03_02_egoview_w_labelmap_config_earlyfusion_dropout_v1_dropmaxonemodality.yaml",
        "train_results_json_filename": "results-2021_03_02_05_08_18-train_2021_03_02_egoview_w_labelmap_config_earlyfusion_dropout_v1_dropmaxonemodality.json",
        "prior_dirname": "2021_03_02_05_08_18"
    },

    '774d6826-763c-4ca4-85d3-608ecc554c62': {
        "release_notes": "dropout either semantics or sensor, 50% prob",
        "yaml_filename": "train_2021_03_01_egoview_w_labelmap_config_earlyfusion_dropout_v1_dropmaxonemodality.yaml",
        "train_results_json_filename": "results-2021_03_01_21_47_43-train_2021_03_01_egoview_w_labelmap_config_earlyfusion_dropout_v1_dropmaxonemodality.json",
        "prior_dirname": "2021_03_01_21_47_43"
    },

    '2a3550a4-7b3d-4ab1-8165-e20d7cb069c9': {
        "release_notes": "dropout either semantics or sensor, 0% prob",
        "yaml_filename": "train_2021_03_02_egoview_w_labelmap_config_earlyfusion_dropout_v2_dropmaxonemodality.yaml",
        "train_results_json_filename": "results-2021_03_02_14_54_24-train_2021_03_02_egoview_w_labelmap_config_earlyfusion_dropout_v2_dropmaxonemodality.json",
        "prior_dirname": "2021_03_02_14_54_24"
    },

    'fe460247-d73f-4519-8643-ff38f95fb3b7': {
        "release_notes": "0% dropout, and with blur",
        "yaml_filename": "train_2021_03_02_egoview_w_labelmap_config_earlyfusion_dropout_v3_dropmaxonemodality.yaml",
        "train_results_json_filename": "results-2021_03_03_01_34_05-train_2021_03_02_egoview_w_labelmap_config_earlyfusion_dropout_v3_dropmaxonemodality.json",
        "prior_dirname": "2021_03_03_01_34_05"
    },

    '615683e4-8431-4b31-ba1d-3523e6165fa9': {
        "release_notes": "late fusion, march 3, sensor + map",
        "yaml_filename": "train_2021_03_03_egoview_config_latefusion_v1.yaml",
        "train_results_json_filename": "results-2021_03_04_05_10_28-train_2021_03_03_egoview_config_latefusion_v1.json",
        "prior_dirname": "2021_03_04_05_10_28"
    },

    '17fd2c0a-fee5-47c5-92cc-8b37f4479a8b': {
        "release_notes": "independent_semantics_dropout_prob: 0.75, independent_map_dropout_prob: 0.50, v1",
        "yaml_filename": "train_2021_03_07_egoview_config_dropout_v1.yaml",
        "train_results_json_filename": "results-2021_03_07_05_49_46-train_2021_03_07_egoview_config_dropout_v1.json",
        "prior_dirname": "2021_03_07_05_49_46"
    },

    '88f42746-3374-40f7-a015-625652ca62c8': {
        "release_notes": "all 3 inputs, blurred, v2, independent_semantics_dropout_prob: 0.25, independent_map_dropout_prob: 0.25",
        "yaml_filename": "train_2021_03_07_egoview_config_dropout_v2.yaml",
        "train_results_json_filename": "results-2021_03_07_16_31_18-train_2021_03_07_egoview_config_dropout_v2.json",
        "prior_dirname": "2021_03_07_16_31_18"
    },

    '44e55ee6-76da-4995-8fff-f4a2a4c3a8af': {
        "release_notes": "v3, independent_semantics_dropout_prob: 0.75, independent_map_dropout_prob: 0.75",
        "yaml_filename": "train_2021_03_07_egoview_config_dropout_v3.yaml",
        "train_results_json_filename": "results-2021_03_08_03_05_21-train_2021_03_07_egoview_config_dropout_v3.json",
        "prior_dirname": "2021_03_08_03_05_21"
    },
    
    '98e50a71-c7b2-411d-99f4-781826488a26': {
        "release_notes": "blurred input, independent_map_dropout_prob: 0.5, independent_semantics_dropout_prob: 0.0, 3/08 v1",
        "yaml_filename": "train_2021_03_08_egoview_config_dropout_v1.yaml",
        "train_results_json_filename": "results-2021_03_08_14_37_03-train_2021_03_08_egoview_config_dropout_v1.json",
        "prior_dirname": "2021_03_08_14_37_03"
    },

    '0061c32d-da98-4583-a311-8f2fc37b6655': {
        "release_notes": "all 3 input modalities, 3/08 v2, independent_map_dropout_prob: 0.0, independent_semantics_dropout_prob: 0.5",
        "yaml_filename": "train_2021_03_08_egoview_config_dropout_v2.yaml",
        "train_results_json_filename": "results-2021_03_09_01_13_03-train_2021_03_08_egoview_config_dropout_v2.json",
        "prior_dirname": "2021_03_09_01_13_03"
    },

    'e0ac04a5-c883-4497-81ef-e89ef3d23fdb': {
        "release_notes": "3/09 late fusion, BEV, reproduce previous numbers",
        "yaml_filename": "train_2021_03_09_lidar_rgb_corr_config_earlyfusion_v1.yaml",
        "train_results_json_filename": "results-2021_03_10_03_40_09-train_2021_03_09_lidar_rgb_corr_config_earlyfusion_v1.json",
        "prior_dirname": "2021_03_10_03_40_09"
    },

    '19180eb3-94a7-4d9c-b3c2-2ddc22cf9704': {
        "release_notes": "bev, early fusion, semantic map sensor, dropout map or semantics",
        "yaml_filename": "train_2021_03_10_lidar_rgb_corr_config_earlyfusion_v1.yaml",
        "train_results_json_filename": "results-2021_03_11_03_00_32-train_2021_03_10_lidar_rgb_corr_config_earlyfusion_v1.json",
        "prior_dirname": "2021_03_11_03_00_32"
    },

    '4d4f41a2-4bfe-42f2-88d0-1db253eeb9be': {
        "release_notes": "ego-view high res, all 3 modalities w/ dropout, 448x448",
        "yaml_filename": "train_2021_03_14_egoview_midres_config_dropout_v1.yaml",
        "train_results_json_filename": "results-2021_03_14_07_38_54-train_2021_03_14_egoview_midres_config_dropout_v1.json",
        "prior_dirname": "2021_03_14_07_38_54"
    },

    '52c611bd-50d1-47ee-aedf-3879958b243c': {
        "release_notes": "BEV, semantics + map only",
        "yaml_filename": "train_2021_03_14_bev_nosensor_config_dropout_v2.yaml",
        "train_results_json_filename": "results-2021_03_15_18_00_30-train_2021_03_14_bev_nosensor_config_dropout_v2.json",
        "prior_dirname": "2021_03_15_18_00_30"
    },

    'de62fc14-defa-4b5e-ab7c-0bd95dc52fb3': {
        "release_notes": "BEV, map-only, 03/17/2021",
        "yaml_filename": "train_2021_03_16_bev_maponly_config_nodropout_v1.yaml",
        "train_results_json_filename": "results-2021_03_16_16_24_08-train_2021_03_16_bev_maponly_config_nodropout_v1.json",
        "prior_dirname": "2021_03_16_16_24_08"
    }
}

