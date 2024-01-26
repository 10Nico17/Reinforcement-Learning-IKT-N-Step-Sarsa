# RL_Team_03

## How to run our Code:

```
$ cd code
$ source make_venv.sh
$ python3 reinforce.py
```

Sometimes, after the parallel learning stage, not all parallel learning processes stop and lock the system, even though all sections report to be „done“.
If this happens, press ctrl+c to stop the program and restart it with the flag ’-s‘ to skip the parallel learning stage:

```
$ python3 reinforce.py -s
```

