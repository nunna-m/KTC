python3 -m ktc train --whichos remote --fold 1 --network alexnet --modalities dc --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --fold 1 --network alexnet --modalities ec --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --fold 1 --network alexnet --modalities pc --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --fold 1 --network alexnet --modalities dc ec pc --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --fold 1 --network alexnet --modalities am --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --fold 1 --network alexnet --modalities tm --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --fold 1 --network alexnet --modalities am tm --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate