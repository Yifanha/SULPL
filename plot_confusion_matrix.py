import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

normailized = True
csv_path = 'results/raf across_fer/confusion_matrix.csv'
js = json.load(open('models/fer2013+/cls2id.json'))

labels = [k for k in js]
# labels = [str(i) for i in range(0, 29)]
tick_marks = np.array(range(len(labels))) + 0.5
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = np.loadtxt(csv_path, delimiter=',')
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(14, 10), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    if normailized:
        c = cm_normalized[y_val][x_val]
    else:
        c = cm[y_val][x_val]
    if c > 0.01:
        if normailized:
            plt.text(x_val, y_val, "%.2f" % (c,), color='red', fontsize=7, va='center' , ha='center')
        else:
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=7, va='center', ha='center')

plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
plot_confusion_matrix(cm)
if normailized:
    plt.savefig(csv_path.replace('.csv', 'normalized.png'), format='png')
else:
    plt.savefig(csv_path.replace('.csv', '.png'), format='png')