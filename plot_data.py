import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("micropattern_analysis.csv")

    # analysis steps:
    # 1. determine the dencity of cells/um in each sample,
    # 2. calculate the expected total number of cells in the hole area based on the diameter
    # 3. calculate the fraction of fositive cells within the hole
    # 4. plot the data in a box plot - y axis is positive fraction and x axis is hole diameter

    df["dencity"] = df["total"] / 500
    df["total_cells_in_hole_diameter"] = df["hole_diameter"] * df["dencity"]
    df["positive_fraction"] = df["positive"] / df["total_cells_in_hole_diameter"]

    sns.set_style("whitegrid")

    # Create the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="hole_diameter", y="positive_fraction", data=df)

    plt.xlabel("Hole Diameter")
    plt.ylabel("Positive Fraction")

    # Save the plot as a PNG file
    plt.savefig("positive_fraction_vs_hole_diameter.png")


if __name__ == "__main__":
    main()
