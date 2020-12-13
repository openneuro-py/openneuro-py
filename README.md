openneuro-py
============

A leightweight Python client for accessing [OpenNeuro](https://openneuro.org)
datasets.

How to use
----------

Download an entire dataset:

```shell
openneuro-py download --dataset=ds000246
```

Exclude a directory:

```shell
openneuro-py download --dataset=ds000246 \
                      --exclude=sub-emptyroom
```

Download only a single file (plus a few essential BIDS files):

```shell
openneuro-py download --dataset=ds000246 \
                      --include=sub-0001/meg/sub-0001_coordsystem.json
```

`--include` and `--exclude` can be passed mulitple times:

```shell
openneuro-py download --dataset=ds000246 \
                      --include=sub-0001/meg/sub-0001_coordsystem.json \
                      --include=sub-0001/meg/sub-0001_acq-LPA_photo.jpg
```
