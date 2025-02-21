from IPython.display import display
from pandas import DataFrame


def question_preserved(src: str, mt: str) -> bool:
    return src.count("?") == mt.count("?")


def exclamation_preserved(src: str, mt: str) -> bool:
    return src.count("!") == mt.count("!")


def token_ratio(src_tokens: list[str], mt_tokens: list[str]) -> float:
    return len(src_tokens) / len(mt_tokens)


def length_ratio(src: str, mt: str) -> float:
    return len(src) / len(mt)


def get_tokens(example: dict[str, str], src_field: str, mt_field: str, tokenizer) -> dict[str, str | list[str]]:
    return {
        "src": example[src_field],
        "mt": example[mt_field],
        "src_tokens": tokenizer.tokenize(example[src_field]),
        "mt_tokens": tokenizer.tokenize(example[mt_field]),
    }


def compute_stats(df: DataFrame):
    # length and token ratios
    df["length_ratio"] = df.apply(lambda x: length_ratio(x["src"], x["mt"]), axis=1)
    df["token_ratio"] = df.apply(lambda x: token_ratio(x["src_tokens"], x["mt_tokens"]), axis=1)

    # token lengths
    df["src_token_len"] = df.apply(lambda x: len(x["src_tokens"]), axis=1)
    df["mt_token_len"] = df.apply(lambda x: len(x["mt_tokens"]), axis=1)

    # qestion marks
    df["contains_question"] = df.apply(lambda x: "?" in x["src"], axis=1)
    df["question_preserved"] = df.apply(lambda x: question_preserved(x["src"], x["mt"]), axis=1)

    # exclamation marks
    df["contains_exclamation"] = df.apply(lambda x: "!" in x["src"], axis=1)
    df["exclamation_preserved"] = df.apply(lambda x: exclamation_preserved(x["src"], x["mt"]), axis=1)

    return df


def check_truncation(df: DataFrame, new_max_tokens: int, field: str = "prompt"):
    truncated = df[df["mt_token_len"] >= new_max_tokens]
    print(f"Number of truncated {field} translations: ", len(truncated))

    if len(truncated) > 0:
        display(truncated[["src", "mt", "mt_token_len"]].sort_values("mt_token_len", ascending=False))
        print()


def check_questions(df: DataFrame, field: str = "prompt"):
    not_preserved = df[df["contains_question"] & ~df["question_preserved"]]
    print(f"Number of {field}s with question but not preserved: ", len(not_preserved))

    if len(not_preserved) > 0:
        return not_preserved

    return None


def check_exclamations(df: DataFrame, field: str = "prompt"):
    not_preserved = df[df["contains_exclamation"] & ~df["exclamation_preserved"]]
    print(f"Number of {field}s with exclamation but not preserved: ", len(not_preserved))

    if len(not_preserved) > 0:
        return not_preserved

    return None


def display_length_ratio(df: DataFrame, sort_by: str = "length_ratio", ascending: bool = False, field: str = "prompt"):
    df = df[["src", "mt", "src_token_len", "mt_token_len", "length_ratio", "token_ratio"]].copy()

    print(f"{field.capitalize()} Length and Token Ratio Analysis")
    display(df.sort_values(sort_by, ascending=ascending))
    print()
