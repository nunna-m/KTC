python3 -m ktc train_registered --whichos remote --modalities am tm --method both --network vgg16_lastblocktrain --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg16_lastblocktrain_boxCrop
python3 -m ktc train_registered --whichos remote --modalities am ec --method both --network vgg16_lastblocktrain --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg16_lastblocktrain_boxCrop
python3 -m ktc train_registered --whichos remote --modalities am pc --method both --network vgg16_lastblocktrain --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg16_lastblocktrain_boxCrop
python3 -m ktc train_registered --whichos remote --modalities am dc --method both --network vgg16_lastblocktrain --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg16_lastblocktrain_boxCrop
python3 -m ktc train_registered --whichos remote --modalities dc tm --method both --network vgg16_lastblocktrain --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg16_lastblocktrain_boxCrop
python3 -m ktc train_registered --whichos remote --modalities ec tm --method both --network vgg16_lastblocktrain --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg16_lastblocktrain_boxCrop
python3 -m ktc train_registered --whichos remote --modalities pc tm --method both --network vgg16_lastblocktrain --config "configfiles/extra/data_options_registered.yaml" --max_steps 50 --filename vgg16_lastblocktrain_boxCrop