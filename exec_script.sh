#!/bin/bash
# for workload in "archeology" "astronomy" "legal" "biomedical"; do
for workload in "biomedical"; do

for sut_name in "BaselineLLMSystemDeepseekR1"; do
#  for config in "Naive" "OneShot" "FewShot"; do
 for config in "Naive"; do
    echo "Running workload: $workload with SUT: $sut_name$config"
    python evaluate.py --sut $sut_name$config --workload $workload.json --dataset_name $workload
done
done
done