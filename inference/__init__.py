"""DeepTune's Test/Inference feature: load a previously trained model (from
a completed run's checkpoint) and run it on new data, producing a
per-sample predicted-class/value plus confidence, and comparing against
ground-truth labels when the new data has them.

Deliberately separate from trainers/, evaluators/, and embed/: those all
assume the labels DeepTune's own split_dataset.py produced are present and
correct (they're built for measuring a model DeepTune just trained), while
this package's job is running an already-trained model against data it has
never seen, which may or may not have labels at all. Each submodule here
(vision, video, text, tabular, timeseries) writes its own small inference
loop rather than reusing evaluators/*/evaluate*.py's, because every one of
those hard-requires labels and discards the per-sample predictions it
computes internally, keeping only aggregate metrics -- see infer.py for the
CLI entry point that ties these together with desktop/runner.py the same
way deeptune.py does for training.
"""
