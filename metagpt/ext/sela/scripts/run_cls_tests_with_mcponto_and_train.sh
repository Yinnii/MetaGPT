#!/bin/bash

# List of tasks to run
tasks=(
  # Clickpredictionsmall
  creditg
  # GesturePhaseSegmentationProcessed
  icr
  jasmine
  kc1
  kick
  mfeatfactors
  segment
  smokerstatus
  softwaredefects
  titanic
  winequalitywhite
)

# Number of rollouts
rollouts=10

# Run experiment for each task
for task in "${tasks[@]}"; do
  echo "Running experiment for task: $task"
  python run_experiment.py \
    --exp_mode mcts \
    --task "$task" \
    --rollouts "$rollouts" \
    --from_scratch \
    --with_pre_insights \
    --with_pretraining
done

echo "All experiments completed."

for task in "${tasks[@]}"; do
  echo "Check if task resulted in mcts results: $task"

  if find /home/yin/Projects/MetaGPT/metagpt/ext/sela/results/mcts -type f -name "*${task}*" | grep -q .; then
    echo "Results found for $task."
  else
    echo "$task" >> ./scripts/restarts_qwen2.5_malex.txt
    echo "$task needs restart."
  fi
done