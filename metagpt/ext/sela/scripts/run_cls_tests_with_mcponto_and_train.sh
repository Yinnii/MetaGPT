#!/bin/bash

# List of tasks to run
tasks=(
  Clickpredictionsmall
  creditg
  GesturePhaseSegmentationProcessed
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

