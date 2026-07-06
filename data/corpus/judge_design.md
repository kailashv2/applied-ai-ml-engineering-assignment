# Judge Pipeline Notes

The judge pipeline accepts JSON or YAML test suites. It builds a structured prompt, calls the judge model, parses a structured verdict, and writes a suite-level report.

For malformed JSON, the parser first tries strict schema validation, then applies a JSON repair fallback, and finally validates the repaired object against the same schema.

Position bias is measured by running pairwise comparisons in both A/B and B/A order. A high flip rate means the judge may be preferring position rather than quality.

Verbosity bias is tested by adding verbose-but-wrong probes and terse-but-correct probes. The rubric penalizes unsupported length and requires per-criterion evidence.
