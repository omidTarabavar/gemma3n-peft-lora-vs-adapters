# Datasets

The course datasets are not redistributed. Place them here:

```text
datasets/
├── in_domain/    # Persian news
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
└── out_domain/   # Wikipedia articles
    ├── train.jsonl
    ├── dev.jsonl
    └── test.jsonl
```

Each line is a JSON object with a `text` field (`content`, `article` or `body` also work).
