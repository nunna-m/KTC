python3 -m ktc train --whichos remote --modalities am --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities tm --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities dc --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities ec --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities pc --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities dc ec --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities ec pc --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities dc pc --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities am dc --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate
python3 -m ktc train --whichos remote --modalities am ec --config "configfiles/extra/data_options.yaml" --max_steps 100 --validate