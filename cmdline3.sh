python3 -m ktc train --whichos remote --modalities am dc tm --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities am dc ec pc tm --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities am ec tm --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities am pc tm --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities dc ec tm --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities dc pc tm --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities ec pc tm --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate