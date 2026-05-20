import concurrent
import json
import math
import os
import random
import time

import filelock
import openai
import tqdm


STORED_FILE = "./cache_api_calls.txt"


def synchronize(func, iters, batch_size=None, workers=2):
    if workers is None:
        workers = 2
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for batch in batchit(iters, batch_size):
            yield pool.map(func, batch)


def batchit(corpus, size):
    assert hasattr(corpus, "__iter__")
    assert size is None or isinstance(size, int) and size > 0
    batch = []
    for row in corpus:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch.clear()
    if len(batch) > 0:
        yield batch


def get_api_config():
    key = os.environ.get("API_KEY")
    if not key:
        raise ValueError(
            "API_KEY environment variable not set.\n"
            "  export API_KEY='...'\n"
            "  export BASE_URL='https://api.example.com/v1/'  (optional)\n"
            "  export MODEL='gpt-4o-mini'  (optional)"
        )
    base_url = os.environ.get("BASE_URL", None)
    model = os.environ.get("MODEL", "gpt-4o-mini")
    return key, base_url, model


class _APISetup:
    def __init__(self, secret_key, engine, function, do_cache=False, max_retry=6, cool_down=1.0, base_url=None):
        assert isinstance(secret_key, str)
        assert isinstance(engine, str)
        assert isinstance(max_retry, int) or max_retry is None
        assert isinstance(cool_down, (float, int)) and cool_down > 0.0
        self._client = openai.OpenAI(api_key=secret_key, base_url=base_url, timeout=60.0)
        self._model = engine
        self._retry = max_retry
        self._cool = cool_down
        self._lock = filelock.FileLock(STORED_FILE + ".lock") if do_cache else None

    def __call__(self, *args, **kwrds):
        inputs = self.preprocess(*args, **kwrds)
        internals = self.create(**inputs)
        if self._lock:
            with self._lock:
                with open(STORED_FILE, "a+") as f:
                    store = {"INPUTS": inputs, "OUTPUTS": internals, "TIME": time.asctime()}
                    f.write(json.dumps(store) + "\n")
        return self.postprocess(internals)

    def batch_call(self, queries, batch_size=8, workers=8, bar=True):
        if bar:
            queries = list(queries)
            bar = tqdm.tqdm(total=math.ceil(len(queries) / batch_size))
        results = []
        for batch_result in synchronize(self.__call__, queries, batch_size, workers):
            results.extend(batch_result)
            if bar:
                bar.update(1)
        return results

    def create(self, *args, **kwrds):
        tries = 0
        max_retry = self._retry if self._retry is not None else 6
        while True:
            try:
                return self._client.chat.completions.create(model=self._model, **kwrds)
            except Exception as e:
                msg = str(e)
                print(("Unknown Error: %s" % msg).replace("\n", "\\n"))
                tries += 1
                if tries > max_retry:
                    return False
                sleep_time = min(self._cool * (2 ** (tries - 1)), 30.0) + random.uniform(0, 0.5)
                if "rate limit" in msg.lower() or "429" in msg.lower():
                    sleep_time = max(sleep_time, 2.0 + random.uniform(0, 0.5))
                time.sleep(sleep_time)

    def preprocess(self, **kwrds):
        return kwrds

    def postprocess(self, outputs):
        return outputs


class Chatting(_APISetup):
    def __init__(
        self,
        secret_key,
        model,
        system=None,
        examples=None,
        cache=False,
        temperature=1.0,
        top_p=0.1,
        n=1,
        seed=42,
        max_tokens=120,
        base_url=None,
    ):
        _APISetup.__init__(self, secret_key, model, "ChatCompletion", base_url=base_url)
        self._params = {
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "seed": seed,
            "max_tokens": max_tokens,
        }
        self.system = system
        self.examples = examples
        self._history = [] if cache else None

    @property
    def system(self):
        return self._instruct

    @system.setter
    def system(self, prompt):
        assert isinstance(prompt, str) or prompt is None
        self._instruct = []
        if prompt is not None:
            self._instruct.append({"role": "system", "content": prompt})

    @property
    def examples(self):
        return self._examples

    @examples.setter
    def examples(self, samples):
        if samples is None:
            samples = []
        if isinstance(samples, str):
            samples = [samples]
        new_examples = []
        for sample in samples:
            assert len(sample) == 2 and all(map(lambda _: isinstance(_, str), sample))
            new_examples.append({"role": "system", "name": "example_user", "content": sample[0]})
            new_examples.append({"role": "system", "name": "example_assistant", "content": sample[1]})
        self._examples = new_examples

    def preprocess(self, new_query):
        new_query = {"role": "user", "content": new_query}
        inputs = self._instruct + self._examples
        if self._history is not None:
            inputs.extend(self._history)
            self._history.append(new_query)
        return {"messages": inputs + [new_query]} | self._params

    def postprocess(self, response):
        if response is False:
            if self._history:
                self._history.pop(-1)
            return False
        out_text = [c.message.content for c in response.choices]
        if self._history:
            self._history.append({"role": "assistant", "content": out_text[0]})
        return out_text

    @classmethod
    def from_env(cls, system=None, examples=None, cache=False, temperature=0.0001, top_p=0.001, n=1, max_tokens=120):
        key, base_url, model = get_api_config()
        return cls(
            secret_key=key,
            model=model,
            system=system,
            examples=examples,
            cache=cache,
            temperature=temperature,
            top_p=top_p,
            n=n,
            max_tokens=max_tokens,
            base_url=base_url,
        )
