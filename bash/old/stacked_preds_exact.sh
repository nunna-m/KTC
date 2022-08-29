python3 -m ktc train_stacked --whichos remote --method MRI --level_0 oldcnn --config "configfiles/extra/stacked_data_options.yaml" --max_steps 50 --filename oldcnn_fc_preds_exactTumor 
python3 -m ktc train_stacked --whichos remote --method CT --level_0 oldcnn --config "configfiles/extra/stacked_data_options.yaml" --max_steps 50 --filename oldcnn_fc_preds_exactTumor   
python3 -m ktc meta_learner --whichos remote --config "configfiles/extra/stacked_data_options.yaml" --max_steps 30 --level0 oldcnn --level1 fc --level0_filename oldcnn_fc_preds_exactTumor
python3 -m ktc train_stacked --whichos remote --method MRI --level_0 vgg16_lastblocktrain --config "configfiles/extra/stacked_data_options.yaml" --max_steps 50 --filename vgg16_lblck_fc_preds_exactTumor 
python3 -m ktc train_stacked --whichos remote --method CT --level_0 vgg16_lastblocktrain --config "configfiles/extra/stacked_data_options.yaml" --max_steps 50 --filename vgg16_lblck_fc_preds_exactTumor   
python3 -m ktc meta_learner --whichos remote --config "configfiles/extra/stacked_data_options.yaml" --max_steps 30 --level0 vgg16_lastblocktrain --level1 fc --level0_filename vgg16_lblck_fc_preds_exactTumor
