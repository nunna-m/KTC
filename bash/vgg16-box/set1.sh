python3 -m ktc train --whichos remote --modalities am --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box
python3 -m ktc train --whichos remote --modalities tm --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box
python3 -m ktc train --whichos remote --modalities dc --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box
python3 -m ktc train --whichos remote --modalities ec --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box
python3 -m ktc train --whichos remote --modalities pc --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box
python3 -m ktc train --whichos remote --modalities dc ec --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box
python3 -m ktc train --whichos remote --modalities ec pc --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box
python3 -m ktc train --whichos remote --modalities dc pc --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box