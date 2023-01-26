python3 -m ktc train --whichos remote --modalities am dc --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box
python3 -m ktc train --whichos remote --modalities am ec --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box
python3 -m ktc train --whichos remote --modalities am pc --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box
python3 -m ktc train --whichos remote --modalities dc tm --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box
python3 -m ktc train --whichos remote --modalities ec tm --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box
python3 -m ktc train --whichos remote --modalities pc tm --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box
python3 -m ktc train --whichos remote --modalities am dc ec --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box
python3 -m ktc train --whichos remote --modalities am ec pc --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box
python3 -m ktc train --whichos remote --modalities am dc pc --method both --network vgg16 --config "configfiles/extra/data_options_box.yaml" --max_steps 50 --filename vgg16_box