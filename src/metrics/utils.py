"""
Helper functions for computing character and word error rates between a
prediction and its target
"""


def _levenshtein_distance(reference, hypothesis) -> int:
    """
    Levenshtein distance between two sequences

    Args:
        reference (Sequence): reference sequence, such as a list of
            characters or words
        hypothesis (Sequence): sequence to compare against the
            reference
    """
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def calc_cer(target_text, predicted_text) -> float:
    """
    Measures how many single character edits separate the prediction
    from the target

    Args:
        target_text (str): ground-truth transcription
        predicted_text (str): predicted transcription
    """
    if len(target_text) == 0:
        return 1.0 if len(predicted_text) > 0 else 0.0
    distance = _levenshtein_distance(target_text, predicted_text)
    return distance / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    """
    Measures how many whole word edits separate the prediction from the
    target

    Args:
        target_text (str): ground-truth transcription
        predicted_text (str): predicted transcription
    """
    target_words = target_text.split()
    predicted_words = predicted_text.split()
    if len(target_words) == 0:
        return 1.0 if len(predicted_words) > 0 else 0.0
    distance = _levenshtein_distance(target_words, predicted_words)
    return distance / len(target_words)
