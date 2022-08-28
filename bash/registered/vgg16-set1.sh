python3 -m ktc train_registered --whichos remote --modalities am tm --method both --network vgg16 --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg16_boxCrop
python3 -m ktc train_registered --whichos remote --modalities am ec --method both --network vgg16 --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg16_boxCrop
python3 -m ktc train_registered --whichos remote --modalities am pc --method both --network vgg16 --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg16_boxCrop
python3 -m ktc train_registered --whichos remote --modalities am dc --method both --network vgg16 --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg16_boxCrop
python3 -m ktc train_registered --whichos remote --modalities dc tm --method both --network vgg16 --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg16_boxCrop