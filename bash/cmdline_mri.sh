python3 -m ktc meta_learner --whichos remote --config "configfiles/extra/stacked_data_options.yaml" --max_steps 30 --level0 vgg16 --level1 xgb

#python -m ktc train_stacked --whichos windows --method MRI --config "configfiles/extra/stacked_data_options.yaml" --max_steps 3