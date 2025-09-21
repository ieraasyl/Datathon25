import os
import json
import warnings

import pandas as pd

from argparse import ArgumentParser

from src.moderation import moderate_df
from src.classification import classify_df
from src.asnwer_generation import generate_answers_df
from src.preprocesing import flatten_our_json, preprocess_df

warnings.filterwarnings(
    "ignore",
    message=r".*does not return WordVectorModel or SupervisedModel.*",
    category=UserWarning,
    module=r"fasttext(\.|$)"
)


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file")
    # parser.add_argument("--output", type=str, required=True, help="Path to the output CSV file")
    # args = parser.parse_args()

    # input_path = args.input
    # output_path = args.output

    # if not os.path.exists(input_path):
    #     raise FileNotFoundError(f"Input file '{input_path}' does not exist.")

    # Parser code
    # parser has as an output parsed_comments_path file
    parsed_comments_path = "examples/instagram_comments_20250921_003125.json"
    
    
    # Preprocessor code
    with open(parsed_comments_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    df_flat = flatten_our_json(obj)
    df_prep = preprocess_df(df_flat, text_col="text")  # adds text_norm, lang_primary, emojis_count...
    df_mod  = moderate_df(df_prep, text_col="text_norm", lang_col="lang_primary")
    df_cls  = classify_df(df_mod, text_col="text_norm", lang_col="lang_primary", emoji_col="emoji_sent")
    df_gen  = generate_answers_df(df_cls, text_col="text_norm", lang_col="lang_primary")

    output_path = "examples/instagram_comments_20250921_003125_final.csv"
    df_gen.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Moderated comments saved to '{output_path}'.")
