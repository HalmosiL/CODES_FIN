#!/bin/bash

#CIRA+AP
python cosine_combination_test.py ../FIN_CONFIG/PSPnet_CIRA+3_CIRA+AP_EPS_0.03.json > /dev/null &
python cosine_combination_test.py ../FIN_CONFIG/PSPnet_SAT2_CIRA+AP_EPS_0.03.json > /dev/null &

#PGD
python pgd_test.py ../FIN_CONFIG/PSPnet_CIRA+3_PDG_EPS_0.03.json > /dev/null &
python pgd_test.py ../FIN_CONFIG/PSPnet_SAT2_PDG_EPS_0.015.json > /dev/null &
