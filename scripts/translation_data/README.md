# 1M translation dataset composition
| Dataset | Size (sentence pairs) |
| --- |-----------------------|
| CCMatrix | 500000                |
| WikiMatrix | 400000                |
| Europarl | 50000                 |
| OpenSubtitles | 50000                 |

---

The datasets have been filtered, deduplicated, test-set (incl. FLORES-200 and WMT18) overlap removed.

OpusFilter config:
```
steps:
  - type: opus_read
    parameters:
      corpus_name: !varstr "{corpus}"
      suppress_prompts: true
      source_language: !varstr "{original_src}"
      target_language: !varstr "{original_tgt}"
      release: latest
      preprocessing: raw
      src_output: !varstr "{src}-{tgt}.{corpus}.raw.{src}.txt"
      tgt_output: !varstr "{src}-{tgt}.{corpus}.raw.{tgt}.txt"

  - type: remove_duplicates
    parameters:
      inputs: [ !varstr "{src}-{tgt}.{corpus}.raw.{src}.txt", !varstr "{src}-{tgt}.{corpus}.raw.{tgt}.txt" ]
      outputs: [ !varstr "{src}-{tgt}.{corpus}.dedup.{src}.txt", !varstr "{src}-{tgt}.{corpus}.dedup.{tgt}.txt" ]

  - type: filter
    parameters:
      inputs: [ !varstr "{src}-{tgt}.{corpus}.dedup.{src}.txt", !varstr "{src}-{tgt}.{corpus}.dedup.{tgt}.txt" ]
      outputs: [ !varstr "{src}-{tgt}.{corpus}.filter.{src}.txt", !varstr "{src}-{tgt}.{corpus}.filter.{tgt}.txt" ]
      filters:
        - LongWordFilter: { }
        - LengthFilter:
            name: char
            unit: char
            min_length: 10
            max_length: 1000
        - LengthFilter:
            name: word
            unit: word
            min_length: 2
            max_length: 100
        - LengthRatioFilter:
            name: word
            unit: word
            threshold: 3
        - CharacterScoreFilter:
            scripts: [ !varstr "{src_script}", !varstr "{tgt_script}" ]
        - LanguageIDFilter:
            name: fasttext
            id_method: fasttext
            fasttext_model_path: /gpfs/space/projects/nlpgroup/modular-nmt/mtee-13/fasttext/lid.176.bin
            languages: [ !varstr "{src}", !varstr "{tgt}" ]
        - TerminalPunctuationFilter: { }
        - NonZeroNumeralsFilter: { }
```

The final deduplication after combining removed 2288 sentence-pairs (the final dataset has 997712 sentence-pairs).

The dataset is shuffled.

---

Instructions created with
```
bitext_to_instructions.py \
    --src-path en-et.combined-v1.shuf.en.txt \
    --tgt-path en-et.combined-v1.shuf.et.txt \
    --src-lang English \
    --tgt-lang Estonian \
    --out-path en-et.v1.json \
    --p-has-input 0.85 \
    --p-has-estonian-instruction 0.05
```