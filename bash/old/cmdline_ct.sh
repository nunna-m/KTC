python3 -m ktc meta_learner --whichos remote --config "configfiles/extra/stacked_data_options.yaml" --max_steps 30 --level0 cnn --level1 fc --level0_filename stacked_test

#python -m ktc train_stacked --whichos windows --method MRI --config "configfiles/extra/stacked_data_options.yaml" --max_steps 1

#python -m ktc meta_learner --whichos windows --config "configfiles/extra/stacked_data_options.yaml" --max_steps 1 --level0 cnn --level1 fc

# python3 -m ktc train_stacked --whichos linux --method MRI --level_0 oldcnn --config "configfiles/extra/stacked_data_options.yaml" --max_steps 1 --filename stacked_temp
# python3 -m ktc train_stacked --whichos linux --method CT --level_0 oldcnn --config "configfiles/extra/stacked_data_options.yaml" --max_steps 1 --filename stacked_temp
# python3 -m ktc meta_learner --whichos linux --config "configfiles/extra/stacked_data_options.yaml" --max_steps 1 --level0 oldcnn --level1 fc --level0_filename stacked_temp

# python3 -m ktc meta_learner --whichos remote --config "configfiles/extra/stacked_data_options.yaml" --max_steps 1 --level0 cnn --level1 fc

# python3 -m ktc meta_learner --whichos linux --config "configfiles/extra/stacked_data_options.yaml" --max_steps 5 --level0 cnn --level1 fc --level0_filename stacked_test
