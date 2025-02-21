#!/bin/bash

# Evaluate translation quality on BeaverTails dataset with different reference-free models

# MetricX-24 Hybrid XXL
python evaluate.py --config ./configs/quality_evaluation/beavertails_it_metricx-24.yaml --log-file ./logs/evaluation_metricx-24.log

# Cometkiwi 23 XXL
python evaluate.py --config ./configs/quality_evaluation/beavertails_it_cometkiwi-23.yaml --log-file ./logs/evaluation_cometkiwi-23.log

# XCOMET XXL
python evaluate.py --config ./configs/quality_evaluation/beavertails_it_xcomet.yaml --log-file ./logs/evaluation_xcomet.log
