# Run each python script in the examples dir through a few test cases
set -e
for FILE in ./examples/*.py
do
  for CURRICULUM in SimpleCartPole SimpleMiniGrid
  do
    for NENV in 1 3
    do
      CMD="python $FILE --curriculum $CURRICULUM --num-parallel-envs $NENV --curriculum-seed 0"
      echo "$CMD"
      eval "$CMD"
    done
  done
done
