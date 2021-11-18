python3 -m ktc train --whichos remote --fold 1 --network resnet --modalities dc --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --fold 1 --network resnet --modalities ec --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --fold 1 --network resnet --modalities pc --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --fold 1 --network resnet --modalities dc ec pc --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --fold 1 --network resnet --modalities am --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --fold 1 --network resnet --modalities tm --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --fold 1 --network resnet --modalities am tm --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate