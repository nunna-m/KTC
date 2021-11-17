python3 -m ktc train --whichos remote --modalities am pc --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities dc tm --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities ec tm --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities pc tm --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities am dc ec --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities am ec pc --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities am dc pc --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities tm dc ec --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate