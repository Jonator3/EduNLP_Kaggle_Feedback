import matplotlib.pyplot as plt
import csv


def wordcount_prediction_string(p_str: str) -> int:
    return len(p_str.split(" "))


def normalize(a: list) -> list:
    return [f / sum(a) for f in a]


labels = ['Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal', 'Evidence', 'Concluding Statement']
labels_short = ['Lead', 'Pos', 'Claim', 'Co-claim', 'Rb', 'Ev', 'CS']
label_indices = dict([(label, i) for (i, label) in enumerate(labels)])
# label_indices := {'Lead': 0, 'Position': 1, 'Claim': 2, 'Counterclaim': 3, ...}
fig = plt.figure()


def analyse_cluster(filename: str):
    label_counts = [0] * len(labels)
    # label counts = [0, 0, 0, 0, 0, ...]

    word_counts = {}
    for label in labels:
        word_counts[label] = []
    # wordcounts := {'Lead': [], 'Position': [], 'Claim': [], 'Counterclaim': [], ...}

    with open(filename, newline='') as csvfile:
        label_reader = csv.reader(csvfile)
        label_reader.__next__()
        count = 0
        for row in label_reader:
            discourse_type = row[5]
            count += 1
            label_counts[label_indices[discourse_type]] += 1
            word_counts[discourse_type].append(wordcount_prediction_string(row[7]))
        print("Read", count, "rows of", filename)

    return label_counts, word_counts


def plot_lf_wc(label_frequencies, word_counts, title):
    _, (ax, axb) = plt.subplots(1, 2, figsize=(16, 8))
    ax.set_title(title)
    ax.bar(labels_short, label_frequencies)
    axb.boxplot(word_counts.values())
    axb.set_xticklabels(labels_short, rotation=45)


def plot_all_clusters(data):
    _, ax = plt.subplots(len(data), len(data[0]), figsize=(12, 100))
    for i, d in enumerate(data):
        ax[i][0].set_title("cluster "+str(i))
        ax[i][0].bar(labels_short, normalize(d[0]))
        ax[i][1].boxplot(d[1].values())
        ax[i][1].set_xticklabels(labels_short, rotation=45)

