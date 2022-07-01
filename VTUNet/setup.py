from setuptools import setup, find_namespace_packages

setup(name='vtunet',
      packages=find_namespace_packages(include=["vtunet", "vtunet.*"]),
      install_requires=[
            "torch>=1.6.0a",
            "tqdm",
            "dicom2nifti",
            "scikit-image>=0.14",
            "medpy",
            "scipy",
            "batchgenerators>=0.21",
            "numpy",
            "sklearn",
            "SimpleITK",
            "pandas",
            "requests",
            "nibabel", 'tifffile'
      ],
      entry_points={
          'console_scripts': [
              'vtunet_convert_decathlon_task = vtunet.experiment_planning.vtunet_convert_decathlon_task:main',
              'vtunet_plan_and_preprocess = vtunet.experiment_planning.vtunet_plan_and_preprocess:main',
              'vtunet_train = vtunet.run.run_training:main',
              'vtunet_train_DP = vtunet.run.run_training_DP:main',
              'vtunet_train_DDP = vtunet.run.run_training_DDP:main',
              'vtunet_predict = vtunet.inference.predict_simple:main',
              'vtunet_ensemble = vtunet.inference.ensemble_predictions:main',
              'vtunet_find_best_configuration = vtunet.evaluation.model_selection.figure_out_what_to_submit:main',
              'vtunet_print_available_pretrained_models = vtunet.inference.pretrained_models.download_pretrained_model:print_available_pretrained_models',
              'vtunet_print_pretrained_model_info = vtunet.inference.pretrained_models.download_pretrained_model:print_pretrained_model_requirements',
              'vtunet_download_pretrained_model = vtunet.inference.pretrained_models.download_pretrained_model:download_by_name',
              'vtunet_download_pretrained_model_by_url = vtunet.inference.pretrained_models.download_pretrained_model:download_by_url',
              'vtunet_determine_postprocessing = vtunet.postprocessing.consolidate_postprocessing_simple:main',
              'vtunet_export_model_to_zip = vtunet.inference.pretrained_models.collect_pretrained_models:export_entry_point',
              'vtunet_install_pretrained_model_from_zip = vtunet.inference.pretrained_models.download_pretrained_model:install_from_zip_entry_point',
              'vtunet_change_trainer_class = vtunet.inference.change_trainer:main',
              'vtunet_evaluate_folder = vtunet.evaluation.evaluator:vtunet_evaluate_folder',
              'vtunet_plot_task_pngs = vtunet.utilities.overlay_plots:entry_point_generate_overlay',
          ],
      },
      
      )
