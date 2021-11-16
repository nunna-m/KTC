python3 -m ktc train --whichos remote --modalities am --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --modalities am tm --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --modalities am dc --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --modalities am pc --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --modalities am ec --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --modalities tm --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --modalities dc tm --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --modalities ec tm --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate
python3 -m ktc train --whichos remote --modalities pc tm --config "configfiles/extra/data_options.yaml" --max_steps 50 --validate