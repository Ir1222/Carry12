"""CSV logging for clean CarryBox perturbation evaluation."""

import csv
import os


class EvaluationCsvLogger:
    def __init__(self, output_dir):
        self.output_dir = os.path.abspath(output_dir)
        self.trace_dir = os.path.join(self.output_dir, "traces")
        self.summary_path = os.path.join(self.output_dir, "summary.csv")
        os.makedirs(self.trace_dir, exist_ok=True)
        self._summary_fields = None

    def write_trace(self, trial_id, rows):
        path = os.path.join(self.trace_dir, f"{trial_id}.csv")
        if not rows:
            with open(path, "w", newline="") as file:
                file.write("")
            return path

        fields = list(rows[0].keys())
        for row in rows[1:]:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with open(path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def append_summary(self, row):
        if self._summary_fields is None:
            self._summary_fields = list(row.keys())
            write_header = not os.path.exists(self.summary_path) or os.path.getsize(
                self.summary_path
            ) == 0
            if not write_header:
                with open(self.summary_path, newline="") as file:
                    existing_fields = next(csv.reader(file), [])
                if existing_fields != self._summary_fields:
                    raise ValueError(
                        "Existing summary.csv schema does not match evaluator CSV v2. "
                        f"existing={existing_fields}, expected={self._summary_fields}. "
                        "Use a new --output_dir."
                    )
        else:
            write_header = False
            if list(row.keys()) != self._summary_fields:
                raise ValueError(
                    "Summary row schema changed during evaluation: "
                    f"got={list(row.keys())}, expected={self._summary_fields}"
                )
        with open(self.summary_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self._summary_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
