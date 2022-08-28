python3 -m ktc train_registered --whichos remote --modalities ec tm --method both --network vgg19 --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg19_boxCrop
python3 -m ktc train_registered --whichos remote --modalities pc tm --method both --network vgg19 --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg19_boxCrop
python3 -m ktc train_registered --whichos remote --modalities dc ec --method both --network vgg19 --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg19_boxCrop
python3 -m ktc train_registered --whichos remote --modalities ec pc --method both --network vgg19 --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg19_boxCrop
python3 -m ktc train_registered --whichos remote --modalities dc pc --method both --network vgg19 --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg19_boxCrop