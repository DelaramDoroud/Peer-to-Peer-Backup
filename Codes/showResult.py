from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def restored_to_backed_up_ratio_plot(backup: list, restore: list):
    block_data, block_time = zip(*backup)
    restore_data, restore_time = zip(*restore)
    # Use defaultdict to accumulate data for each year
    year_data1 = defaultdict(list)
    year_data2 = defaultdict(list)
    
    for year, value in zip(map(int, block_time), block_data):
        year_data1[year].append(value)
            
    for year, value in zip(map(int, restore_time), restore_data):
        year_data2[year].append(value)
    
    years = sorted(set(year_data1.keys()).union(set(year_data2.keys())))  # سال‌های مشترک
    restore_to_backup_ratio = []

    for year in years:
        total_backup = sum(year_data1.get(year, [0]))  # مقدار کل بکاپ در سال
        total_restore = sum(year_data2.get(year, [0]))  # مقدار کل بازیابی در سال
        
        if total_backup > 0:  # جلوگیری از تقسیم بر صفر
            ratio = total_restore / total_backup
        else:
            ratio = 0  # اگر بکاپی وجود نداشته باشد، نسبت را صفر قرار می‌دهیم
        
        restore_to_backup_ratio.append(ratio)
    #print("ratiooooooooo", restore_to_backup_ratio)
    plt.figure()
    plt.plot(years, restore_to_backup_ratio, color="magenta")
    plt.xlabel('years')
    plt.ylabel('the ratio of blocks')
    plt.title("The ratio of Restored to Backed-up blocks over time")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.5)
    plt.xlim(0, 120)
    plt.show()

def blocks_count_plot(local_or_backup_blocks: list, titlee: str, lablee: str, y1: int, y2: int):
        data, time = zip(*local_or_backup_blocks)

        # Use defaultdict to accumulate data for each year
        year_data = defaultdict(list)
        for year, value in zip(map(int, time), data):
            year_data[year].append(value)

        # Calculate the mean for each year
        mean_data = [sum(values) / len(values) for values in year_data.values()]
        years = list(year_data.keys())
        plt.plot(years, mean_data)
        plt.xlabel("years")
        plt.ylabel(f"Number of {lablee}")
        plt.title(titlee)
        plt.xlim(0, 120)  # Set x-axis limits from 0 to 100 years
        plt.ylim(y1, y2)  # Set y-axis limits from 0 to 100 Blocks
        plt.show()
        
def selfish_node_count(selfish_node: int):
        print(selfish_node)
        plt.figure(figsize=(8, 4))
        plt.bar(selfish_node, 1, color='lightblue')
        plt.xlabel("index of selfish node")
        plt.ylabel("status")
        plt.title("Selfish Node Index")
        plt.xlim(0,10)  # Set x-axis limits from 0 to 100 years
        plt.ylim(0,1)  # Set y-axis limits from 0 to 100 Blocks
        plt.xticks(range(0, 22, 2))
        plt.yticks([0, 1])
        plt.show()