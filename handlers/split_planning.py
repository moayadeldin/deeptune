"""Split size calculations shared by dataset preparation and desktop checks."""
import math

DEFAULT_SPLIT_RATIOS = (0.7, 0.1, 0.2)


def allocate_split_counts(count: int, ratios=DEFAULT_SPLIT_RATIOS) -> tuple[int, int, int]:
    """Round ratios to three nonempty partitions without losing any samples."""
    if len(ratios) != 3 or any(not math.isfinite(r) or r <= 0 for r in ratios) or not math.isclose(sum(ratios), 1):
        raise ValueError('Train, validation, and test proportions must be positive and sum to 1.')
    if count < 3:
        raise ValueError(f'Three separate splits need at least 3 samples; found {count}.')
    desired = [count * ratio for ratio in ratios]
    sizes = [max(1, math.floor(value)) for value in desired]
    while sum(sizes) > count:
        index = max((i for i in range(3) if sizes[i] > 1), key=lambda i: sizes[i] - desired[i])
        sizes[index] -= 1
    while sum(sizes) < count:
        index = max(range(3), key=lambda i: desired[i] - sizes[i])
        sizes[index] += 1
    return tuple(sizes)


def classification_split_issues(class_counts, unit='samples') -> list[str]:
    counts = {str(label): int(count) for label, count in class_counts.items() if count > 0}
    issues = []
    if len(counts) < 2:
        issues.append(f'Classification needs at least 2 classes; found {len(counts)}.')
    short = [(label, count) for label, count in counts.items() if count < 3]
    if short:
        details = '; '.join(f'{label!r}: {count} (add {3 - count})' for label, count in short[:10])
        if len(short) > 10:
            details += f'; and {len(short) - 10} more classes'
        issues.append(f'Each class needs at least 3 different {unit} for separate training, validation, and test sets. {details}.')
    return issues
