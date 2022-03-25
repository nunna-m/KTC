#!/bin/bash
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities am --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities tm --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities dc --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities ec --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities pc --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities dc ec --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities ec pc --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities dc pc --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities am dc --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities am ec --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities am pc --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities dc tm --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities ec tm --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities pc tm --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities am dc ec --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities am ec pc --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities am dc pc --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities dc ec tm --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities ec pc tm --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities dc pc tm --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities am dc tm --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities am ec tm --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities am pc tm --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities am tm --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities dc ec pc --kfolds 5
python -m pre  crossvalDataGen --whichos linux --path pre/paths.yaml --modalities am dc ec pc tm --kfolds 5
