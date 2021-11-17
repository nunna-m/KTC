python3 -m ktc train --whichos remote --modalities tm ec pc --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities tm dc pc --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities am tm dc --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities am tm ec --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities am tm pc --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities am tm --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities dc ec pc --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities am tm dc ec pc --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate