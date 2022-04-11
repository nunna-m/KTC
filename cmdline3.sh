python3 -m ktc train --whichos remote --modalities dc ec tm --method both --network cnn --config "configfiles/extra/data_options.yaml" --max_steps 30
python3 -m ktc train --whichos remote --modalities ec pc tm --method both --network cnn --config "configfiles/extra/data_options.yaml" --max_steps 30
python3 -m ktc train --whichos remote --modalities dc pc tm --method both --network cnn --config "configfiles/extra/data_options.yaml" --max_steps 30
python3 -m ktc train --whichos remote --modalities am dc tm --method both --network cnn --config "configfiles/extra/data_options.yaml" --max_steps 30
python3 -m ktc train --whichos remote --modalities am ec tm --method both --network cnn --config "configfiles/extra/data_options.yaml" --max_steps 30
python3 -m ktc train --whichos remote --modalities am pc tm --method both --network cnn --config "configfiles/extra/data_options.yaml" --max_steps 30
python3 -m ktc train --whichos remote --modalities am tm --method both --network cnn --config "configfiles/extra/data_options.yaml" --max_steps 30
python3 -m ktc train --whichos remote --modalities dc ec pc --method both --network cnn --config "configfiles/extra/data_options.yaml" --max_steps 30
python3 -m ktc train --whichos remote --modalities am dc ec pc tm --method both --network cnn --config "configfiles/extra/data_options.yaml" --max_steps 30