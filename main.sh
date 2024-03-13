params=(0.00 0.45 0.90)
number_of_players=2
for par in ${params[@]} ; do
    python PureContinuious_with_DQN.py -alpha ${par} -n ${number_of_players}
    sleep 5
done