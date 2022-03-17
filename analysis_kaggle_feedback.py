import matplotlib.pyplot as plt
import csv


def wordcount_prediction_string(p_str: str) -> int:
    return len(p_str.split())


def normalize(a: list) -> list:
    return [f / sum(a) for f in a]


def analyse_cluster(
        filename: str,
        labels: list[str],
        column_class=lambda header: header.index("class") if "class" in header else header.index("discourse_type"),
        column_prediction_string=lambda header: header.index("predictionstring"),
):
    label_indices = dict([(label, i) for (i, label) in enumerate(labels)])
    # label_indices := {'Lead': 0, 'Position': 1, 'Claim': 2, 'Counterclaim': 3, ...}

    label_counts = [0] * len(labels)
    # label counts = [0, 0, 0, 0, 0, ...]

    word_counts = {}
    for label in labels:
        word_counts[label] = []
    # wordcounts := {'Lead': [], 'Position': [], 'Claim': [], 'Counterclaim': [], ...}

    with open(filename, newline='') as csvfile:
        label_reader = csv.reader(csvfile)
        header = label_reader.__next__()
        column_class = column_class(header)
        column_prediction_string = column_prediction_string(header)
        count = 0
        for row in label_reader:
            count += 1
            label = row[column_class]
            if label in labels:
                label_counts[label_indices[label]] += 1
                word_counts[label].append(wordcount_prediction_string(row[column_prediction_string]))
        print("Read", count, "rows of", filename)

    return label_counts, word_counts


def plot_lf_wc(label_frequencies, word_counts, labels):
    _, (ax, axb) = plt.subplots(1, 2, figsize=(12, 5))
    ax.set_title("label-distribution")
    ax.bar(labels, label_frequencies)
    axb.boxplot(word_counts.values())
    axb.set_xticklabels(labels, rotation=45)
    axb.set_title("# tokens")
    plt.show()


def plot_all_clusters(data, labels):
    _, ax = plt.subplots(len(data), len(data[0]), figsize=(12, 100))
    for i, d in enumerate(data):
        ax[i][0].set_title("cluster " + str(i))
        ax[i][0].bar(labels, normalize(d[0]))
        ax[i][1].boxplot(d[1].values())
        ax[i][1].set_xticklabels(labels, rotation=45)
