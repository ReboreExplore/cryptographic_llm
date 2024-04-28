# Generate dataset statistics for the given dataset
import matplotlib.pyplot as plt
import pandas as pd

def generate(field):
    field_count = crypto_dataset[field].value_counts()
    field_count.plot(kind='pie',autopct='%1.1f%%',figsize=(5, 5))
    plt.title(f'Percentage distribution of {field} in the dataset')
    # Save the pie chart as a PNG file
    plt.savefig(f'./assets/dataset_analysis/{field}_pie_chart.png')
    plt.close()

if __name__ == "__main__":
    crypto_dataset = pd.read_csv('./dataset/crypto_dataset_v1.csv', sep=',',quotechar='"',skipinitialspace=True,encoding='utf-8')
    plot_stat = ['topic','category','type']
    for field in plot_stat:
        generate(field)
    print("Dataset statistics generated successfully!")

