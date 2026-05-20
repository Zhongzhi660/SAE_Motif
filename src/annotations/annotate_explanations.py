"""
Step 2: Annotate SAE neuron text spans with semantic explanations.

Usage:
    python annotate_explanations.py <input_tsv> [output_tsv]

Input:  TSV with columns: FeatureID, Words
Output: TSV with columns: FeatureID, Verify, Summary, Words

Requires: export OPENAI_API_KEY='sk-...'
"""
import sys
import os
import time
import re
import string
import json
import argparse
import concurrent
import multiprocessing

import tqdm

try:
    from .api_client import Chatting, get_api_config
except ImportError:
    from api_client import Chatting, get_api_config


class TextSpanExplainer:
    def __init__(self):
        instruct = "You are studying a neural network. Each neuron looks for one particular concept/topic/theme/behavior/pattern. " +\
               "Look at some text spans the neuron activates for and guess what the neuron is looking for. " +\
               "Note that, some neurons may only look for a particular text pattern, while some others may be interestedin very abstractive concepts. " +\
               "Pay more attention to the end of each text span as they supposed to be more correlated to the neuron behavior. " +\
               "Don't list examples of text spans and keep your summary as detail as possible. " +\
               "If you cannot summarize most of the text spans, you should say ``Cannot Tell.``"
        examples = [("""Span 1: w.youtube.com/watch?v=5qap5aO4z9A
                        Span 2: youtube.come/yegfnfE7vgDI
                        Span 3: {'token': 'bjXRewasE36ivPBx
                        Span 4: /2023/fid?=0gBcWbxPi8uC""",
                     "Base64 encoding for web development."),
                     ("""Span 1: cross-function\\n
                        Span 2: cross-function.
                        Span 3: this is a cross-function
                        Span 4: Cross-Function""",
                      'Particular text pattern "cross-function".'),
                     ("""Span 1: novel spectroscopic imaging platform
                        Span 2: and protein evolutionary network modeling
                        Span 3: reactions-centric biochemical model
                        Span 4: chaperone interaction network""",
                      "Biological terms."),
                     ("""Span 1: is -17a967
                        Span 2: what is 8b8 - 10ad2
                        Span 3: 83 -11111011001000001011
                        Span 4: is -c1290 - -1""",
                      "Synthetic math: Arithmetic, numbers with small digits, in unusual bases."),
                     ("""Span 1: Could you please provide me some
                        Span 2: USER__: Could you please provide me some
                        Span 3: USER__: Could you please provide me some
                        Span 4: Sure! Could you please provide me some""",
                      'Particular text pattern "Could you please provide me some"')]
        self.model = Chatting.from_env(cache=False, system=instruct, examples=examples,
                                       temperature=0.0001, top_p=0.0001, n=1)

    def __call__(self, cases, workers=8, batch_size=8):
        if isinstance(cases, str):
            cases = [cases]
        if not isinstance(cases, (tuple, list)):
            cases = list(cases)
        return list(map(self.clean, self.model.batch_call(cases, batch_size=batch_size, workers=workers)))

    def clean(self, summaries):
        if summaries is False:
            return "Cannot Tell."
        temp = set(_ for _ in summaries if "cannot tell" not in _.lower())
        if len(temp) == 0:
            return "Cannot Tell."
        raw = " or ".join(_.split(".")[0] for _ in temp) + '.'
        return self._clean_summary(raw)

    @staticmethod
    def _clean_summary(text):
        # Remove "The/This neuron is/appears to be looking for ..." prefix
        text = re.sub(
            r'^(?:The|This)\s+neuron\s+(?:is|appears\s+to\s+be)\s+'
            r'(?:looking\s+for|detecting|activated\s+by)\s+',
            '', text, count=1, flags=re.IGNORECASE)
        # Remove markdown bold markers
        text = text.replace('**', '')
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        return text


class TextSpanJudge:
    def __init__(self):
        instruct = (
            "You are a linguistic expert. "
            "You will receive a concept summary and several text spans from a neuron. "
            "Judge whether the text spans well represent the given concept/topic/theme/pattern. "
            "Spans that share the same phrases or are duplicated are acceptable. "
            "Do not be too strict.\n\n"
            "You MUST end your response with EXACTLY this line:\n"
            "Final Decision: [[ Yes/Probably/Maybe/No ]]\n\n"
            "Replace Yes/Probably/Maybe/No with ONE of these four words. "
            "Do NOT omit the brackets. Do NOT change the format."
        )
        examples = [
            ("Reversed or mirrored text.\n"
             "Span 1: senotS gnilloR\n"
             "Span 2: leoJ ylliB\n"
             "Span 3: rennuR etiK",
             "The spans are all backwards versions of known titles: \"Rolling Stones\", "
             "\"Billy Joel\", \"Kite Runner\". They clearly match the concept of reversed text.\n\n"
             "Final Decision: [[ Yes ]]"),
            ("Requests for help with cooking recipes.\n"
             "Span 1: the governance framework for Basel III compliance\n"
             "Span 2: could you outline the key parameters of network latency",
             "The spans discuss regulatory frameworks and network parameters, "
             "which are unrelated to cooking recipes.\n\n"
             "Final Decision: [[ No ]]"),
        ]
        self.model = Chatting.from_env(system=instruct, examples=examples, cache=False,
                                       temperature=0.0001, top_p=0.0001, n=1)

    def __call__(self, cases, workers=8, batch_size=8):
        cases = ["%s\n%s" % (c[0], c[1]) for c in cases]
        cases = self.model.batch_call(cases, batch_size=batch_size, workers=workers)
        return list(map(self.clean, cases))

    def clean(self, verify):
        if verify is False:
            return "no"
        text = verify[0].lower()
        # Try to extract from [[ ]] brackets
        if "[[" in text and "]]" in text:
            inner = text.split("[[", 1)[-1].rsplit("]]", 1)[0].strip()
            for label in ("yes", "probably", "maybe", "no"):
                if label in inner:
                    return label
        # Fallback: find "final decision" then extract label after it
        if "final decision" in text:
            after = text.split("final decision")[-1]
            for label in ("yes", "probably", "maybe", "no"):
                if label in after[:30]:
                    return label
        # Last resort: scan the last 60 chars for a label
        tail = text[-60:]
        for label in ("yes", "probably", "maybe", "no"):
            if label in tail:
                return label
        return "no"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate SAE neuron text spans with semantic explanations.")
    parser.add_argument("input_tsv", help="TSV with FeatureID and Words columns")
    parser.add_argument("output_tsv", nargs="?", default=None, help="Output TSV (default: <input>_explained_<model>.tsv)")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent API workers (default: 8)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per progress bar step (default: 8)")
    args = parser.parse_args()

    input_file = args.input_tsv
    _, _, model_name = get_api_config()
    model_tag = model_name.replace("/", "-")
    if args.output_tsv:
        output_file = args.output_tsv
    else:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_explained_{model_tag}{ext}"

    model = TextSpanExplainer()
    judge = TextSpanJudge()

    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")

    with open(input_file, encoding="utf8") as f:
        headline = f.readline().strip().split("\t")
        idx = headline.index("FeatureID")
        text = headline.index("Words")
        fullset = [(_.split("\t")[idx], _.split("\t")[text].strip()) for _ in f.read().strip('\n').split("\n")]
    results = [[x[0], "-", "Cannot Tell.", x[1]] for x in fullset]

    need_explain = [item for item in results if len(item[3]) > 0]
    print(f"Explaining {len(need_explain)} features...")
    explanations = model((_[3] for _ in need_explain), workers=args.workers, batch_size=args.batch_size)
    for item, expl in zip(need_explain, explanations):
        item[1], item[2] = 'no', expl

    need_verify = [item for item in results if "cannot tell" not in item[2].lower()]
    print(f"Verifying {len(need_verify)} explanations...")
    verifications = judge(([_[2], _[3]] for _ in need_verify), workers=args.workers, batch_size=args.batch_size)
    for item, verify in zip(need_verify, verifications):
        item[1] = verify

    from collections import Counter
    c = Counter([_[1] for _ in results])
    print(f"\nExplainability: {(c['yes'] + c['probably']) / len(results):.4f}")
    for cate, freq in c.items():
        print(f"  {cate}: {freq / len(results):.4f}")

    with open(output_file, "w", encoding="utf8") as f:
        f.write("FeatureID\tVerify\tSummary\tWords\n")
        for idx, verify, summary, words in results:
            words = words.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            summary = summary.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            verify = verify.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            f.write(f"{idx}\t{verify}\t{summary}\t{words}\n")

    print(f"Saved to {output_file}")
