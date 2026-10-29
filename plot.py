from pathlib import Path
import os
import matplotlib.pyplot as plt

def plot_chart(train_list, test_list,x_label, y_label,x_legend,y_legend, title, save_path):
  path = Path(save_path)
  os.makedirs(path.parent, exist_ok=True)
  plt.plot(train_list,label=x_legend)
  plt.plot(test_list,label=y_legend)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)
  plt.legend()
  plt.savefig(save_path)
  plt.close()
  
