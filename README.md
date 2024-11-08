# Multi-Hop Knowledge Graph Reasoning with Reward Shaping

Based off the work by [salesforce](https://github.com/salesforce/MultiHopKG) but heavily edited.


# Run

## Poetry (Optional)

Not necessary but you might find it convenient to install poetry:

```sh
sudo apt install poetry
```

Then install all packages here

```sh
poetry install --no-root
```

Then enter the environment

```sh
poetry shell
```

## Data

Make sure you've got your hands on `data-release.tgz` and, while in the repo root, decompress it with:

```sh
tar -xvf data-release.tgz
```

## Actually Running it

```sh
python mlm_training.py
```
