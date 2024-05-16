# Generate dataset statistics for the given dataset
import matplotlib.pyplot as plt
import pandas as pd
import os
import pylab



def my_autopct(pct):
    return f'{pct:.1f}%'

# Generate pie chart for the given field
def generate(field):
    '''
    This function generates a pie chart for the given field

    Input:  field - The field for which the pie chart is to be generated

    '''
    field_count = crypto_dataset[field].value_counts()
    plt.figure(figsize=(8, 8)) 
    field_count.plot(kind='pie',autopct=my_autopct, startangle=0, colors=['#D6E3F8','#FFD23F','#0CCE6B','#FE938C'],fontsize=13)
    # Set a title for the pie chart
    plt.title(f'Percentage distribution of {field} in the dataset', fontsize=15, loc='center')
    plt.gcf().set_size_inches(8, 8)
    # Set a legend
    pylab.ylabel('')
    plt.legend(loc='lower right', bbox_to_anchor=(1,0), bbox_transform=plt.gcf().transFigure, title=f'{field} values', title_fontsize=15,fontsize=15)
    # Save the pie chart as a PNG file
    save_path = f'dataset_analysis/{field}_pie_chart.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def print_stat_for_field(field):
    '''
    This function prints the statistics for the given field
    '''
    print(f"Statistics for {field} field:")
    print(crypto_dataset[field].value_counts())
    print("\n")


if __name__ == "__main__":
    crypto_dataset = pd.read_csv('train/all/train_all-v1510.csv', sep=',',quotechar='"',skipinitialspace=True,encoding='utf-8')
    plot_stat = ['topic','category','type']
    for field in plot_stat:
        generate(field)
        print_stat_for_field(field)
    print("Dataset statistics generated successfully!")

