python3 -m ktc train --whichos remote --fold 1 --network cnn --modalities dc --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --fold 1 --network cnn --modalities ec --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --fold 1 --network cnn --modalities pc --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --fold 1 --network cnn --modalities am tm --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
