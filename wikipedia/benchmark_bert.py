from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments
from pprint import pprint

args = PyTorchBenchmarkArguments(models=["bert-base-uncased"], batch_sizes=[1], sequence_lengths=[100])
benchmark = PyTorchBenchmark(args)
results = benchmark.run()
pprint(results)
