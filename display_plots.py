import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image, display

# Function to display a plot from the saved directory
def display_plot(plot_path):
    if os.path.exists(plot_path):
        img = mpimg.imread(plot_path)
        plt.figure(figsize=(14, 7))
        plt.imshow(img)
        plt.axis('off') 
        plt.show()
    else:
        print(f"Plot {plot_path} does not exist.")

# List of plot filenames to display 
tickers = ['AAPL', 'GOOG', 'MSFT']  

# Display all the saved plots
for ticker in tickers:
    # Display Train Loss vs Validation Loss plot
    display_plot(f'plots/{ticker}_train_loss_vs_val_loss.png')

    # Display Actual vs Predicted Prices plot
    display_plot(f'plots/{ticker}_actual_vs_predicted_prices.png')
