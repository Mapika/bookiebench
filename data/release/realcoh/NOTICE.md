# realcoh v2 release: sources, licences and notices

`ids_only` sources ship without their text; rebuild it from the Hugging Face datasets with
`python -m honest.realcoh.build_v2 rebuild --release <this dir> --out <dir>` (needs the datasets in the local HF cache,
at the revisions pinned in manifest.json `revisions`: `huggingface-cli download <repo> --repo-type dataset --revision <rev>`).

E-mail addresses in all state texts are masked as `[EMAIL]`, in every source (`rebuild` applies the same masking).

| source | dataset | licence | release mode | in decider training | likely in pretraining |
|---|---|---|---|---|---|
| belebele_es | facebook/belebele (spa_Latn) | cc-by-sa-4.0 | keep | False | True |
| bgl_logs | logfit-project/BGL | LogHub licence (research use), re-published | ids_only | True | True |
| climate | tdiggelm/climate_fever | not declared | ids_only | True | True |
| code_defects | google/code_x_glue_cc_defect_detection | c-uda (CodeXGLUE); code LGPL-2.1+ (FFmpeg) / GPL-2.0 (QEMU) | keep | True | True |
| eurlex_fr | coastalcph/multi_eurlex (fr) | cc-by-sa-4.0 | keep | False | True |
| finance_news | abisee/cnn_dailymail | apache-2.0 (dataset scripts); article text copyright CNN / Daily Mail | ids_only | False | True |
| forecast_news | abisee/cnn_dailymail | apache-2.0 (dataset scripts); article text copyright CNN / Daily Mail | ids_only | False | True |
| gold_news | ChanceFocus/flare-headlines | not declared (FLARE / gold-news headlines) | ids_only | True | True |
| hdfs_logs | logfit-project/hdfsv1-grouped-labeled | LogHub licence (research use), re-published | ids_only | True | True |
| ledgar | coastalcph/lex_glue (ledgar) | cc-by-4.0 | keep | True | True |
| math_problems | DigitalLearningGmbH/MATH-lighteval | mit | keep | False | True |
| med_ru | AlucardV/medical-specialty-classification | not declared | ids_only | True | True |
| mmlu_pro | TIGER-Lab/MMLU-Pro | mit | keep | False | True |
| news | abisee/cnn_dailymail | apache-2.0 (dataset scripts); article text copyright CNN / Daily Mail | ids_only | False | True |
| reviews | HuggingFaceFW/fineweb-edu | odc-by-1.0 (dataset); page text copyright of each site | ids_only | False | True |
| sci_claims | copenlu/scientific-exaggeration-detection | gpl-3.0 | ids_only | False | True |
| sports | abisee/cnn_dailymail | apache-2.0 (dataset scripts); article text copyright CNN / Daily Mail | ids_only | False | True |
| ssh_logs | bolu61/loghub_2 | LogHub licence (research use); IPs, hosts and users masked | ids_only | False | True |
| stackexchange | HuggingFaceFW/fineweb-edu | odc-by-1.0 (dataset); page text copyright of each site | ids_only | False | True |
| student_answers | nkazi/SciEntsBank | cc-by-4.0 | keep | True | True |
| support_chat | Salesforce/APIGen-MT-5k | cc-by-nc-4.0 | ids_only | False | False |
| swe_issues | princeton-nlp/SWE-bench_Verified | mit (dataset); issue text by GitHub users | ids_only | False | True |
| symptoms | gretelai/symptom_to_diagnosis | apache-2.0 | keep | True | False |
| thunderbird_logs | logfit-project/Thunderbird | LogHub licence (research use), re-published | ids_only | False | True |
| tos | coastalcph/lex_glue (unfair_tos) | cc-by-4.0 | keep | True | True |

**code_defects**: Functions are from FFmpeg (LGPL-2.1-or-later) and QEMU (GPL-2.0); see https://ffmpeg.org/legal.html and https://wiki.qemu.org/License. Redistributed via CodeXGLUE (Devign) under C-UDA.
